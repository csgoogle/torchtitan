# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In-body FSDP all-gather for `ScannedLayerStack`.

SimpleFSDP installs its all-gather as a *property* on the parameter name, so
reading a stacked parameter gathers the whole `[n_layers, ...]` tensor at once,
necessarily before the `stablehlo.while` that consumes it. Measured on
`debugmodel`: 10 all-gathers in `ENTRY`, zero in the loop body. At LLaMA-3 70B
that is 136.9 GB of gathered bf16 weights resident at once against a 96 GB
budget -- OOM before the first activation. Gathering one layer at a time inside
the body needs 1.71 GB. That 80x is the whole reason to scan at 70B.

So the stacked parameter is sharded on dim 1 (dim 0 is the scan axis; slicing a
sharded axis inside the loop would need a dynamic collective per iteration),
`ScannedLayerStack.forward` unwraps it with `.to_local()`, and the body gathers
the single `[shard, ...]` slice it needs. A
`torch.distributed._functional_collectives` call does trace into a `scan` body
and does lower into the `stablehlo.while` region on torch_tpu, because its group
is a plain string that bakes into the graph. A DTensor cannot go there at all:
the mesh it closes over is lifted as a `FakeScriptObject` operand, and the scan
HOP accepts only `Tensor | int | SymInt`.

SimpleFSDP still does the *sharding* -- `distribute_tensor` plus
`OffsetBasedRNGTracker` is what keeps initialization bit-identical to a
single-device run. Only its gather is bypassed.

One half of the fix is missing from this file: the gathered weight must also be
recomputed in backward, or the min-cut partitioner saves it as a scan `ys`
residual and the 136.9 GB comes back layer by layer. See `recompute.py`.
"""

import dataclasses
from typing import Any

import torch
from torch import nn
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
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
      extent did not divide the mesh) -- cast it, do not gather it.
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
  backward_dtype=reduce_dtype)` is deliberately asymmetric -- gather in bf16 to
  halve comm volume, reduce in fp32 to keep the gradient accurate -- and no
  composition of casts around a symmetric collective reproduces that.
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


def plan_gathers(
    stack: nn.Module, mp_policy: MixedPrecisionPolicy
) -> dict[str, ScanFSDPSpec]:
  """Derives the in-body gather plan from the placements SimpleFSDP installed.

  The plan is read back off the DTensors rather than predicted from the mode,
  because `data_parallel` silently downgrades `Shard(1)` to `Replicate` for a
  parameter whose dim-1 extent does not divide the mesh. Gathering a replicated
  parameter would concatenate it `mesh.size()` times along a real weight axis --
  wrong, and quiet.

  Returns:
    A mangled-name -> spec mapping. A parameter absent from it is used as-is:
    no cast, no gather.

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
      plan[name] = ScanFSDPSpec(
          "", mp_policy.param_dtype, mp_policy.reduce_dtype, sharded=False
      )
      continue

    mesh = param.device_mesh
    if mesh.ndim != 1:
      raise ValueError(
          "scan in-body FSDP supports a 1-D dp mesh only (got"
          f" {mesh.ndim}-D for {name}); hybrid_shard would need one collective"
          " per mesh axis inside the loop body."
      )
    placement = param.placements[0]
    if isinstance(placement, Shard):
      if placement.dim != 1:
        raise ValueError(
            f"{name} is sharded on dim {placement.dim}; the scan axis is dim 0"
            " and the in-body gather assumes dim 1 (= dim 0 of the per-layer"
            " weight)."
        )
      sharded = True
      n_sharded += 1
    else:
      sharded = False
    plan[name] = ScanFSDPSpec(
        group_name=funcol._resolve_group_name(mesh),  # pylint: disable=protected-access
        param_dtype=mp_policy.param_dtype,
        reduce_dtype=mp_policy.reduce_dtype,
        sharded=sharded,
    )

  logger.info(
      "scan in-body FSDP: %d/%d stacked parameters gathered per layer inside"
      " the scan body (the rest are replicated).",
      n_sharded,
      len(plan),
  )
  return plan


def shard_stacked_layers(
    stack: nn.Module,
    *,
    dp_mesh: DeviceMesh,
    dp_mode: str,
    mp_policy: MixedPrecisionPolicy,
) -> dict[str, ScanFSDPSpec]:
  """Shards the stack on dim 1 and installs the matching in-body gather plan.

  Runs after SimpleFSDP has wrapped the rest of the model with the stacked
  parameters ignored -- `resolve_dp_plan` hands both wraps the same plan.

  `data_parallel` is used here for its *sharding* only. It turns each stacked
  parameter into a dim-1-sharded DTensor, which is what keeps initialization
  bit-identical to a single-device run (`distribute_tensor` plus
  `OffsetBasedRNGTracker`); the all-gather it installs alongside is the one this
  module exists to bypass.
  """
  data_parallel(stack, dp_mesh, mode=dp_mode, mp_policy=mp_policy, shard_dim=1)
  plan = plan_gathers(stack, mp_policy)
  stack.set_fsdp_plan(plan)
  return plan


__all__ = [
    "ScanFSDPSpec",
    "gather_slice",
    "plan_gathers",
    "shard_stacked_layers",
]
