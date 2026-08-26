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
_INSTALLED_RECOMPUTE_BODY = None

# Ops that are free (or near-free) to recompute and that therefore must not be
# allowed to act as a saved stand-in for a tensor tagged MUST_RECOMPUTE. The
# functionalized `*_copy` spellings are what actually reach the partitioner
# inside a scan body; the aliasing spellings are listed for the non-HOP path.
_VIEW_LIKE_OPS = frozenset({
    torch.ops.aten.t_copy.default,
    torch.ops.aten.t.default,
    torch.ops.aten.view_copy.default,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.transpose_copy.int,
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute_copy.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.expand_copy.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.squeeze_copy.dim,
    torch.ops.aten.unsqueeze_copy.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.slice_copy.Tensor,
    torch.ops.aten.select_copy.int,
    torch.ops.aten.alias_copy.default,
    torch.ops.aten.alias.default,
    torch.ops.aten.detach.default,
    torch.ops.aten._to_copy.default,
})


def _forward_region(graph) -> list:
  """The joint graph's forward nodes — everything the tangents cannot reach.

  A joint graph is topologically ordered fwd-then-bwd, so one forward sweep
  marking each user of a `tangents_*` placeholder classifies the backward
  region exactly. Tagging a backward node `MUST_RECOMPUTE` is mostly inert
  (min-cut handles `required_bw_nodes` before it consults the policy), but it
  is inert by accident, and a tagged nondeterministic op there would be
  rewritten through `run_with_rng_state`. Cheaper to not tag them.
  """
  backward = set()
  for node in graph.nodes:
    if node.op == "placeholder":
      if node.name.startswith("tangents"):
        backward.add(node)
    elif any(inp in backward for inp in node.all_input_nodes):
      backward.add(node)
  return [node for node in graph.nodes if node not in backward]


def install_scan_collective_recompute(recompute_body: bool = False) -> None:
  """Makes the scan body's partitioner recompute its all-gathers in backward.

  With `recompute_body`, every value-producing node in the body is tagged too,
  which is what full activation checkpointing means for a scanned stack: the
  only thing that crosses the cut is the loop carry, so the `ys` residual stops
  growing with the width of the block. `torch.utils.checkpoint` cannot express
  this — the HOP re-traces the body through `materialize_as_graph` and drops the
  recompute tags before this partitioner ever runs, silently.

  Idempotent. The scan HOP partitions its body with the ordinary
  `min_cut_rematerialization_partition`
  (`torch/_higher_order_ops/partitioner.py`, `HopGraphMinCutPartitioner`), so
  wrapping that function is what reaches the scan body — there is no per-HOP
  partitioner hook to use instead.

  `all_gather*`, the `wait_tensor` that consumes it, and every *view-like*
  descendant of that wait (transpose / view / reshape / dtype cast) are tagged.
  The descendants matter as much as the gather: tagging the gather alone leaves
  min-cut free to save `t_copy(gathered_weight)` instead — a full-size tensor of
  exactly the shape we were trying not to keep — and it will, because a
  transpose is free to save and an all-gather is expensive to recompute.
  Measured on `debugmodel`: gather-only tagging left 5 × `t_copy` of the
  gathered `[768,256]` / `[256,768]` / `[256,256]` weights in the saved set.

  `reduce_scatter` is left alone: it lives in the backward region already and
  tagging it would only confuse the cut.
  """
  global _ORIGINAL_MIN_CUT, _INSTALLED_RECOMPUTE_BODY
  if _ORIGINAL_MIN_CUT is not None:
    if _INSTALLED_RECOMPUTE_BODY != recompute_body:
      # The wrapper closes over `recompute_body`, so the first install wins and
      # the second caller silently gets a different activation-memory policy
      # from the one it asked for. Two models in one process with different
      # `memory_policy` settings is the way to hit this.
      logger.warning(
          "scan in-body FSDP: install_scan_collective_recompute(%s) ignored;"
          " already installed with recompute_body=%s. Call"
          " uninstall_scan_collective_recompute() first to change it.",
          recompute_body,
          _INSTALLED_RECOMPUTE_BODY,
      )
    return

  import torch._functorch.partitioners as partitioners  # pylint: disable=g-import-not-at-top

  _ORIGINAL_MIN_CUT = partitioners.min_cut_rematerialization_partition
  _INSTALLED_RECOMPUTE_BODY = recompute_body

  def tag_then_partition(joint_module, joint_inputs, **kwargs):
    tagged = set()
    if recompute_body:
      tagged.update(
          node
          for node in _forward_region(joint_module.graph)
          if node.op == "call_function" and node.meta.get("val") is not None
      )
    for node in joint_module.graph.nodes:
      if not isinstance(node.target, torch._ops.OpOverload):  # pylint: disable=protected-access
        continue
      if node.target.namespace != "_c10d_functional":
        continue
      opname = node.target._schema.name.split("::")[-1]  # pylint: disable=protected-access
      if opname.startswith("all_gather"):
        tagged.add(node)
      elif opname == "wait_tensor" and any(
          inp in tagged for inp in node.all_input_nodes
      ):
        tagged.add(node)

    # Close the set over view-like users. `graph.nodes` is topologically
    # ordered, so a single forward sweep reaches every descendant.
    for node in joint_module.graph.nodes:
      if node in tagged or node.op != "call_function":
        continue
      if node.target not in _VIEW_LIKE_OPS:
        continue
      if any(inp in tagged for inp in node.all_input_nodes):
        tagged.add(node)

    for node in tagged:
      node.meta["recompute"] = partitioners.CheckpointPolicy.MUST_RECOMPUTE
      node.meta.setdefault("ac_graph_id", 0)
    logger.info(
        "scan in-body FSDP: tagged %d node(s) MUST_RECOMPUTE out of %d graph"
        " nodes (recompute_body=%s).",
        len(tagged),
        len(joint_module.graph.nodes),
        recompute_body,
    )
    fwd, bwd = _ORIGINAL_MIN_CUT(joint_module, joint_inputs, **kwargs)
    if os.environ.get("SCAN_FSDP_AUDIT_SAVED_SET") == "1":
      log_saved_set(fwd, bwd)
    return fwd, bwd

  partitioners.min_cut_rematerialization_partition = tag_then_partition
  logger.info(
      "scan in-body FSDP: forced recompute of in-body all-gathers"
      " (min_cut_rematerialization_partition wrapped)."
  )


def uninstall_scan_collective_recompute() -> None:
  """Restores the stock partitioner. Idempotent; safe to call when not installed.

  The install wraps a module-level function in `torch._functorch.partitioners`,
  which is process-global: without this, a test that installs it leaks the
  wrapper into every later partition in the same process, and a caller that
  wants a different `recompute_body` cannot get one.
  """
  global _ORIGINAL_MIN_CUT, _INSTALLED_RECOMPUTE_BODY
  if _ORIGINAL_MIN_CUT is None:
    return

  import torch._functorch.partitioners as partitioners  # pylint: disable=g-import-not-at-top

  partitioners.min_cut_rematerialization_partition = _ORIGINAL_MIN_CUT
  _ORIGINAL_MIN_CUT = None
  _INSTALLED_RECOMPUTE_BODY = None
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

  `install_scan_collective_recompute` is not optional either: without it the
  min-cut partitioner saves the *gathered* per-layer weight as a scan `ys`
  residual and the 136.9 GB comes straight back, just materialized
  layer-by-layer.
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
  install_scan_collective_recompute(recompute_body=recompute_body)
