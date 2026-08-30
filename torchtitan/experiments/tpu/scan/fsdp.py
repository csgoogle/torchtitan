# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In-body FSDP all-gather for `ScannedLayerStack`.

Why this file exists
--------------------
SimpleFSDP installs its all-gather as a *class property* on the parameter name,
so reading a stacked parameter gathers the **whole** `[n_layers, ...]` tensor
once, at graph-build time — necessarily before the `stablehlo.while` that
consumes it. Measured on `debugmodel`: 10 `all-gather`s in `ENTRY`, zero in the
loop body. At LLaMA-3 70B that is 136.9 GB of gathered bf16 weights resident
simultaneously against a 96 GB/device budget, i.e. OOM before the first
activation. Per-layer gathering is 1.71 GB. The 80x is the whole reason
scan-over-layers is worth doing at 70B at all.

The fix has two halves, and **both** are required — measured, not assumed:

1. Move the collective into the combine_fn. A
   `torch.distributed._functional_collectives` call does trace into a `scan`
   body and does lower into the `stablehlo.while` body region on torch_tpu.
   The stacked parameter stays a DTensor sharded on dim 1 (so initialization
   keeps DTensor's offset-based RNG and stays bit-identical to a single-device
   run); `forward` unwraps it with `.to_local()` and the body gathers the one
   `[shard, ...]` slice it needs.
2. Force the gather to be **recomputed** in backward. Without this the min-cut
   partitioner saves the gathered per-layer weight as a scan `ys` residual and
   the full 136.9 GB comes straight back, just materialized layer-by-layer.
   `_functorch.config.unsafe_allow_optimization_of_collectives` looks like the
   knob and measurably does nothing (the collective is still off the
   recomputable-op allowlist); only an explicit `CheckpointPolicy.MUST_RECOMPUTE`
   tag gets through, because `ban_recomputation_if_allowed` honours a user tag
   ahead of both the collective ban and the allowlist.

   `ScannedLayerStack.forward` sets that tag the ordinary way -- an ambient
   `fx_traceback.current_meta["recompute"]` window around the `scan` call, which
   reaches every node the body traces. Tagging the whole body is also what
   removes the gathered weight from the saved set: with no untagged view op left
   in the body, min-cut has nothing to save in the gathered tensor's place, so
   the *shards* cross the cut instead. `install_scan_body_ac_shim` below is the
   only machinery required, and it exists solely to work around two upstream
   nodes the ambient channel over-tags.

Safety of forcing recompute of a collective: upstream's *Note [Recomputing
collectives in the partitioner]* bans it because a pass that
reorders/deletes/duplicates a collective **inconsistently across ranks**
deadlocks. Here every rank compiles the same SPMD graph and the duplication is
identical on all of them.
"""

import collections
import dataclasses
import os
from typing import Any

import torch
from torch import nn
import torch.distributed._functional_collectives as funcol
from torch.distributed.tensor import DTensor, Shard

from torchtitan.tools.logging import logger


@dataclasses.dataclass(frozen=True)
class ScanFSDPSpec:
  """How `ScannedLayerStack.forward` should gather one stacked parameter.

  Attributes:
    group_name: the resolved process-group name funcol collectives take.
    param_dtype: compute dtype for the gathered weight (SimpleFSDP's
      `forward_dtype`); `None` means leave it alone.
    reduce_dtype: dtype the gradient is reduced in (SimpleFSDP's
      `backward_dtype`); `None` means reduce in `param_dtype`.
    sharded: `False` for a parameter SimpleFSDP left replicated (its dim-1
      extent did not divide the mesh) — cast it, do not gather it.
  """

  group_name: str
  param_dtype: torch.dtype | None
  reduce_dtype: torch.dtype | None
  sharded: bool


class _AllGatherReduceScatter(torch.autograd.Function):
  """`all_gather` forward / `reduce_scatter` backward with SimpleFSDP's dtypes.

  A plain `funcol.all_gather_tensor` already differentiates to a
  `reduce_scatter`, but it reduces in whatever dtype the forward gathered in.
  SimpleFSDP's `redistribute(forward_dtype=param_dtype,
  backward_dtype=reduce_dtype)` is deliberately asymmetric — gather in bf16 to
  halve comm volume, reduce in fp32 to keep the gradient accurate — and no
  composition of casts around a symmetric collective reproduces that. Hence an
  explicit `autograd.Function`.
  """

  @staticmethod
  def forward(  # pylint: disable=arguments-differ
      ctx: Any,
      shard: torch.Tensor,
      group_name: str,
      param_dtype: torch.dtype | None,
      reduce_dtype: torch.dtype | None,
  ) -> torch.Tensor:
    ctx.group_name = group_name
    ctx.reduce_dtype = reduce_dtype
    ctx.shard_dtype = shard.dtype
    if param_dtype is not None:
      shard = shard.to(param_dtype)
    return funcol.all_gather_tensor(shard, gather_dim=0, group=group_name)

  @staticmethod
  def backward(ctx: Any, grad: torch.Tensor):  # pylint: disable=arguments-differ
    if ctx.reduce_dtype is not None:
      grad = grad.to(ctx.reduce_dtype)
    out = funcol.reduce_scatter_tensor(
        grad, reduceOp="sum", scatter_dim=0, group=ctx.group_name
    )
    return out.to(ctx.shard_dtype), None, None, None


def gather_slice(
    slice_: torch.Tensor, spec: ScanFSDPSpec | None
) -> torch.Tensor:
  """Turns one layer's local shard into that layer's full weight, in the body."""
  if spec is None:
    return slice_
  if not spec.sharded:
    return slice_ if spec.param_dtype is None else slice_.to(spec.param_dtype)
  return _AllGatherReduceScatter.apply(
      slice_, spec.group_name, spec.param_dtype, spec.reduce_dtype
  )


def plan_stacked_gathers(
    stack: nn.Module,
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
) -> dict[str, ScanFSDPSpec]:
  """Reads the DTensor placements SimpleFSDP installed and derives the plan.

  Called once, after `data_parallel(...)` has run over the stack. Returns a
  mangled-name -> spec mapping; a parameter absent from the mapping is used
  as-is (no cast, no gather).

  Args:
    stack: the `ScannedLayerStack`, post-`data_parallel`.
    param_dtype: SimpleFSDP's `MixedPrecisionPolicy.param_dtype`.
    reduce_dtype: SimpleFSDP's `MixedPrecisionPolicy.reduce_dtype`.

  Raises:
    ValueError: on a mesh or placement the in-body path cannot express.
  """
  plan: dict[str, ScanFSDPSpec] = {}
  n_sharded = 0
  for name, param in stack._parameters.items():  # pylint: disable=protected-access
    if param is None:
      continue
    if not isinstance(param, DTensor):
      # Not under SimpleFSDP at all (single-device run): nothing to gather, but
      # still honour the compute dtype so scanned and unrolled paths agree.
      plan[name] = ScanFSDPSpec("", param_dtype, reduce_dtype, sharded=False)
      continue

    mesh = param.device_mesh
    if mesh.ndim != 1:
      raise ValueError(
          "scan in-body FSDP supports a 1-D dp mesh only (got"
          f" {mesh.ndim}-D for {name}); hybrid_shard would need one collective"
          " per mesh dim inside the loop body."
      )
    placement = param.placements[0]
    if isinstance(placement, Shard):
      if placement.dim != 1:
        raise ValueError(
            f"{name} is sharded on dim {placement.dim}; the scan axis is dim 0"
            " and the in-body gather assumes dim 1 (= dim 0 of the per-layer"
            " weight). See parallelize_llama's shard_dim=1."
        )
      sharded = True
      n_sharded += 1
    else:
      # SimpleFSDP auto-downgrades Shard -> Replicate when the extent does not
      # divide the mesh (`p.shape[dim] < device_mesh.size(i)`); mirror that.
      sharded = False
    plan[name] = ScanFSDPSpec(
        group_name=funcol._resolve_group_name(mesh),  # pylint: disable=protected-access
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        sharded=sharded,
    )

  logger.info(
      "scan in-body FSDP: %d/%d stacked parameters gathered per layer inside"
      " the scan body (the rest are replicated).",
      n_sharded,
      len(plan),
  )
  return plan


def log_saved_set(fwd: torch.fx.GraphModule, bwd: torch.fx.GraphModule) -> None:
  """Logs what the partitioner decided to save across the scan body's cut.

  This is the cheap way to audit the property this module exists to guarantee —
  that the *shards*, not the *gathered* weights, cross from forward to backward
  — without dumping and reading HLO. Enable with `SCAN_FSDP_AUDIT_SAVED_SET=1`.
  A healthy scanned LLaMA block shows only shard-shaped `_to_copy` weights plus
  activations, and a backward that contains as many `all_gather_into_tensor`
  calls as the forward.
  """

  def collectives(gm):
    return collections.Counter(
        node.target._schema.name.split("::")[-1]  # pylint: disable=protected-access
        for node in gm.graph.nodes
        if isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace == "_c10d_functional"
    )

  outputs = next(n for n in fwd.graph.nodes if n.op == "output").args[0]
  logger.info(
      "scan in-body FSDP saved-set audit: %d tensors cross the cut;"
      " fwd collectives=%s bwd collectives=%s",
      len(outputs),
      dict(collectives(fwd)),
      dict(collectives(bwd)),
  )
  for out in outputs:
    val = getattr(out, "meta", {}).get("val", None)
    shape = tuple(val.shape) if val is not None and hasattr(val, "shape") else "?"
    logger.info(
        "scan in-body FSDP saved-set audit:   %-18s %-22s %s",
        getattr(out, "name", out),
        shape,
        getattr(out, "target", None),
    )


_ORIGINAL_MIN_CUT = None


def install_scan_body_ac_shim() -> None:
  """Lets a caller's ambient AC tags reach the scan body's partitioner.

  `fx_traceback.current_meta["recompute"]`, set around a `scan(...)` call inside
  `preserve_node_meta()`, reaches every node of the body's joint graph -- so the
  caller can express "recompute this whole region" with the ordinary AOT
  mechanism instead of a graph pass that has to recognise collectives by name.
  `ScannedLayerStack.forward` opens that window when `recompute_body` is set.

  Two properties of the ambient channel make the shim necessary:

  1. It also tags `placeholder` nodes, which upstream's dispatch-mode tagger
     never does and `must_recompute()` does not exclude. Min-cut then builds a
     flow network referencing nodes it does not own and raises
     `NetworkXError: node source not in graph`. A graph input is by definition
     not recomputable, so stripping the key from placeholders cannot change any
     partition upstream's own tagger would produce.
  2. The window is necessarily open while the *outer* tracer creates the `scan`
     node itself. Left tagged, min-cut recomputes the whole layer loop in
     backward -- the forward runs twice per step, silently, and the saved-set
     metric *improves* while doing it. Clearing the tag from the `scan` node is
     enough; its `getitem` users then inherit nothing.

  Both are local stands-in for one-line upstream guards in
  `torch/_functorch/partitioners.py`. Wrapping
  `min_cut_rematerialization_partition` is what reaches the scan body: the HOP
  partitions its body with that same function
  (`torch/_higher_order_ops/partitioner.py`, `HopGraphMinCutPartitioner`), and
  there is no per-HOP partitioner hook to use instead.

  Idempotent.
  """
  global _ORIGINAL_MIN_CUT
  if _ORIGINAL_MIN_CUT is not None:
    return

  import torch._functorch.partitioners as partitioners  # pylint: disable=g-import-not-at-top
  # `torch._higher_order_ops.scan` the attribute is the exported `scan`
  # *function*, which shadows the submodule of the same name -- so the class has
  # to come in by `from`-import, not by attribute access off the package.
  from torch._higher_order_ops.scan import ScanOp  # pylint: disable=g-import-not-at-top

  _ORIGINAL_MIN_CUT = partitioners.min_cut_rematerialization_partition

  def strip_untaggable_then_partition(joint_module, joint_inputs, **kwargs):
    stripped = 0
    for node in joint_module.graph.nodes:
      if "recompute" not in node.meta:
        continue
      if node.op == "placeholder" or isinstance(node.target, ScanOp):
        node.meta.pop("recompute", None)
        stripped += 1
    if stripped:
      logger.info(
          "scan in-body FSDP: stripped ambient recompute tags from %d"
          " placeholder/scan node(s) before min-cut.",
          stripped,
      )
    fwd, bwd = _ORIGINAL_MIN_CUT(joint_module, joint_inputs, **kwargs)
    if os.environ.get("SCAN_FSDP_AUDIT_SAVED_SET") == "1":
      log_saved_set(fwd, bwd)
    return fwd, bwd

  partitioners.min_cut_rematerialization_partition = (
      strip_untaggable_then_partition
  )
  logger.info(
      "scan in-body FSDP: ambient-AC shim installed"
      " (min_cut_rematerialization_partition wrapped)."
  )


def uninstall_scan_body_ac_shim() -> None:
  """Restores the stock partitioner. Idempotent; safe to call when not installed.

  The install wraps a module-level function in `torch._functorch.partitioners`,
  which is process-global: without this, a test that installs it leaks the
  wrapper into every later partition in the same process.
  """
  global _ORIGINAL_MIN_CUT
  if _ORIGINAL_MIN_CUT is None:
    return

  import torch._functorch.partitioners as partitioners  # pylint: disable=g-import-not-at-top

  partitioners.min_cut_rematerialization_partition = _ORIGINAL_MIN_CUT
  _ORIGINAL_MIN_CUT = None
  logger.info("scan in-body FSDP: restored the stock min-cut partitioner.")


def shard_stacked_layers(
    stack: nn.Module,
    *,
    dp_mesh: Any,
    dp_mode: str,
    mp_policy: Any,
    recompute_body: bool,
) -> None:
  """Shards the stacked parameters on dim 1 and wires the in-body gather.

  Shared by the graph_trainer and fullgraph lanes; runs after SimpleFSDP has
  been applied to the rest of the model with the stacked parameters ignored.

  Dim 0 of a stacked parameter is the *layer* axis that `scan` walks; sharding
  it would make every iteration slice a sharded axis. Dim 1 — the first real
  weight axis — is sharded instead.

  `data_parallel` is used here for its *sharding* only, not its gather: it turns
  each stacked parameter into a dim-1-sharded DTensor, which is what keeps
  initialization bit-identical to a single-device run (`distribute_tensor` +
  `OffsetBasedRNGTracker`). The all-gather it installs as a parametrization
  property is deliberately never read — `ScannedLayerStack.forward` takes
  `.to_local()` and gathers one layer at a time *inside* the scan body. Reading
  the property instead gathers all 80 layers at graph-build time: 136.9 GB
  resident at 70B, measured.

  Recompute of the in-body gather is not optional either: without it the
  min-cut partitioner saves the *gathered* per-layer weight as a scan `ys`
  residual and the 136.9 GB comes straight back, just materialized
  layer-by-layer. `recompute_body` is what asks for it -- the stack opens an
  ambient AC window around its `scan` call, and `install_scan_body_ac_shim`
  makes that window survive to the body's partitioner.
  """
  from torchtitan.experiments.graph_trainer.simple_fsdp import (  # pylint: disable=g-import-not-at-top
      data_parallel as simple_fsdp_data_parallel,
  )

  simple_fsdp_data_parallel(
      stack,
      dp_mesh,
      mode=dp_mode,
      mp_policy=mp_policy,
      shard_dim=1,
  )
  stack.set_fsdp_plan(
      plan_stacked_gathers(
          stack,
          param_dtype=mp_policy.param_dtype,
          reduce_dtype=mp_policy.reduce_dtype,
      )
  )
  stack.set_recompute_body(recompute_body)
  install_scan_body_ac_shim()
