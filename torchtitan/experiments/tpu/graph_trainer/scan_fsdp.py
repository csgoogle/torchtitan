# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""All-gather one decoder layer's weights, inside the scan body.

SimpleFSDP installs its all-gather as a property on the parameter name, so
reading a stacked parameter gathers the whole ``[n_layers, ...]`` tensor at
once -- necessarily before the loop that consumes it. That is correct, and it
is also every layer's weights resident at the same time, which past a few tens
of billions of parameters does not fit. Gathering one layer at a time is
``n_layers`` times smaller, and is the reason scan-over-layers is worth doing
at that size at all.

So the body reads this rank's *shard* of one layer and issues the collective
itself. A `torch.distributed._functional_collectives` call traces into a scan
body and lowers into the loop body region, so the gather ends up where the
compute is.

Moving the gather in is only half of it. Left alone, the min-cut partitioner
prices a collective as expensive to redo and saves the *gathered* per-layer
weight across the forward/backward cut: the gather is inside the loop but its
result is carried out of it, and the memory saving evaporates. ``layer_stack``
closes that with ``activation_memory_budget=0.0`` around the ``scan`` call; see
the comment there.

Upstream's *Note [Recomputing collectives in the partitioner]* bans recomputing
collectives because a pass that reorders or duplicates one inconsistently
across ranks deadlocks. Here every rank compiles the same SPMD graph, so the
duplication is identical on all of them.
"""

import dataclasses
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.distributed.tensor import DTensor, Shard

from torchtitan.tools.logging import logger


__all__ = ["ScanFSDPSpec", "gather_slice", "plan_stacked_gathers"]


@dataclasses.dataclass(frozen=True)
class ScanFSDPSpec:
    """How the scan body should gather one stacked parameter.

    Attributes:
        group_name: resolved process-group name, as funcol collectives take it.
        param_dtype: compute dtype for the gathered weight; ``None`` leaves it.
        reduce_dtype: dtype the gradient is reduced in; ``None`` means reduce in
            whatever the forward gathered in.
        sharded: ``False`` for a parameter SimpleFSDP left replicated -- cast
            it, do not gather it.
    """

    group_name: str
    param_dtype: torch.dtype | None
    reduce_dtype: torch.dtype | None
    sharded: bool


class _AllGatherReduceScatter(torch.autograd.Function):
    """``all_gather`` forward, ``reduce_scatter`` backward, with FSDP's dtypes.

    ``funcol.all_gather_tensor`` already differentiates to a reduce-scatter, but
    it reduces in whatever dtype the forward gathered in. SimpleFSDP's
    ``redistribute(forward_dtype=param_dtype, backward_dtype=reduce_dtype)`` is
    deliberately asymmetric -- gather in bf16 to halve comm volume, reduce in
    fp32 to keep the gradient accurate -- and no arrangement of casts around a
    symmetric collective reproduces that.
    """

    @staticmethod
    def forward(
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
    def backward(ctx: Any, grad: torch.Tensor):
        if ctx.reduce_dtype is not None:
            grad = grad.to(ctx.reduce_dtype)
        out = funcol.reduce_scatter_tensor(
            grad, reduceOp="sum", scatter_dim=0, group=ctx.group_name
        )
        return out.to(ctx.shard_dtype), None, None, None


def gather_slice(shard: torch.Tensor, spec: ScanFSDPSpec | None) -> torch.Tensor:
    """Turns one layer's local shard into that layer's whole weight."""
    if spec is None:
        return shard
    if not spec.sharded:
        return shard if spec.param_dtype is None else shard.to(spec.param_dtype)
    return _AllGatherReduceScatter.apply(
        shard, spec.group_name, spec.param_dtype, spec.reduce_dtype
    )


def plan_stacked_gathers(
    holder: nn.Module,
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
) -> dict[str, ScanFSDPSpec]:
    """Reads back the placements SimpleFSDP installed and derives the plan.

    Called once, after ``data_parallel`` has run over the stacks, so that the
    body reproduces whatever sharding actually happened rather than assuming it.

    Raises:
        ValueError: on a mesh or placement the in-body path cannot express.
    """
    plan: dict[str, ScanFSDPSpec] = {}
    n_sharded = 0
    for name, param in holder._parameters.items():
        if param is None:
            continue
        if not isinstance(param, DTensor):
            # Not under FSDP at all: nothing to gather, but still honour the
            # compute dtype so the scanned and unrolled paths agree.
            plan[name] = ScanFSDPSpec("", param_dtype, reduce_dtype, sharded=False)
            continue

        mesh = param.device_mesh
        if mesh.ndim != 1:
            raise ValueError(
                f"the scan body's gather needs a 1-D dp mesh, got {mesh.ndim}-D "
                f"for {name}; hybrid_shard would need one collective per mesh "
                "dim inside the loop"
            )
        placement = param.placements[0]
        if isinstance(placement, Shard):
            if placement.dim != 1:
                raise ValueError(
                    f"{name} is sharded on dim {placement.dim}, but dim 0 is the "
                    "scan axis and the body gathers dim 1 (= dim 0 of the "
                    "per-layer weight)"
                )
            sharded = True
            n_sharded += 1
        else:
            # `data_parallel` downgrades Shard -> Replicate when the extent does
            # not divide the mesh; mirror that rather than gather a replica.
            sharded = False
        # The mesh is 1-D, checked above, so it names exactly one group.
        plan[name] = ScanFSDPSpec(
            group_name=mesh.get_group().group_name,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            sharded=sharded,
        )

    logger.info(
        "Gathering %d/%d stacked weights per layer inside the scan body",
        n_sharded,
        len(plan),
    )
    return plan
