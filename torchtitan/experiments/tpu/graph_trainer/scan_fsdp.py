"""All-gather one decoder layer's weights, inside the scan body.

SimpleFSDP gathers on attribute access. Reading a stacked parameter therefore
pulls the whole ``[n_layers, ...]`` tensor, before the loop that consumes it, so
every layer's weights are resident at once. That stops fitting past a few tens
of billions of parameters.

Here the body takes this rank's shard of one layer and gathers that instead --
``n_layers`` times smaller. A functional collective traces into a scan body, so
the gather lowers into the loop and lands next to the compute that needs it.

``plan_stacked_gathers`` reads back what SimpleFSDP installed on the stacks;
``gather_slice`` is what the body calls on one row.

Recomputing this gather in the backward is safe, despite upstream's *Note
[Recomputing collectives in the partitioner]*: what that bans is a duplication
some ranks make and others don't, and every rank here compiles the same SPMD
graph.
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
        group_name: resolved process-group name, in the form funcol collectives
            take.
        param_dtype: compute dtype for the gathered weight; ``None`` leaves it
            alone.
        reduce_dtype: dtype to reduce the gradient in; ``None`` means reduce in
            whatever dtype the forward gathered in.
        sharded: ``False`` for a parameter SimpleFSDP left replicated. Nothing
            to gather in the forward; the gradient still has to be summed.
    """

    group_name: str
    param_dtype: torch.dtype | None
    reduce_dtype: torch.dtype | None
    sharded: bool


class _AllGatherReduceScatter(torch.autograd.Function):
    """Gathers one sharded layer's weight, and reduces its gradient back.

    This is the sharded case of ``gather_slice``: all-gather in the forward,
    reduce-scatter in the backward, the same pair SimpleFSDP runs unrolled.

    Written out by hand so the two halves can use different dtypes -- bf16 to
    gather, to halve the comm volume, and fp32 to reduce, to keep the gradient
    accurate. A plain collective reduces in whatever dtype the forward used,
    and casting after the reduce is already too late.
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


class _ReduceGradOnly(torch.autograd.Function):
    """Sums the gradient of a weight SimpleFSDP left replicated.

    This is the unsharded case of ``gather_slice``: the weight is already whole
    on every rank, so the forward only casts. The gradient is not whole -- each
    rank computed its own from its own microbatch -- so the backward all-reduces.

    Unrolled, that sum comes from ``to_local(grad_placements=Partial)``. The
    body reads the local tensor straight off the ``DTensor``, which promises no
    such thing, so it has to ask for the sum itself.
    """

    @staticmethod
    def forward(
        ctx: Any,
        replica: torch.Tensor,
        group_names: tuple[str, ...],
        param_dtype: torch.dtype | None,
        reduce_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        ctx.group_names = group_names
        ctx.reduce_dtype = reduce_dtype
        ctx.replica_dtype = replica.dtype
        return replica if param_dtype is None else replica.to(param_dtype)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        if ctx.reduce_dtype is not None:
            grad = grad.to(ctx.reduce_dtype)
        for group_name in ctx.group_names:
            grad = funcol.all_reduce(grad, "sum", group=group_name)
        return grad.to(ctx.replica_dtype), None, None, None


def gather_slice(shard: torch.Tensor, spec: ScanFSDPSpec | None) -> torch.Tensor:
    """Turns one layer's local shard into that layer's whole weight."""
    if spec is None:
        return shard
    if not spec.sharded:
        if not spec.group_name:
            # No mesh at all, so no other rank holds a gradient to add.
            return shard if spec.param_dtype is None else shard.to(spec.param_dtype)
        return _ReduceGradOnly.apply(
            shard, (spec.group_name,), spec.param_dtype, spec.reduce_dtype
        )
    return _AllGatherReduceScatter.apply(
        shard, spec.group_name, spec.param_dtype, spec.reduce_dtype
    )


def plan_stacked_gathers(holder: nn.Module) -> dict[str, ScanFSDPSpec]:
    """Save what the body needs to know about each stack, before ``to_local``.

    The scan takes plain tensors, so the placements and dtypes that told
    SimpleFSDP how to gather are gone by the time the body runs. This reads
    them off the holder while they are still there and keeps them in a
    ``ScanFSDPSpec``, which is what ``gather_slice`` then dispatches on.

    The caller runs this on the first forward, not at construction: the model
    stacks itself in ``__init__``, which is before ``data_parallel`` and
    ``to_empty``, so there is nothing to read back yet.

    Raises:
        ValueError: on a mesh, placement or shard size the body cannot gather.
    """
    # Registered by `_register_parametrization` (see `simple_fsdp.py`). It is
    # missing only when the holder never went through `data_parallel`, and in
    # that case there is no mixed precision to mirror either.
    computation = getattr(holder, "_simple_fsdp_parametrization_list", [None])[0]
    param_dtype = computation.param_dtype if computation else None
    reduce_dtype = computation.reduce_dtype if computation else None

    plan: dict[str, ScanFSDPSpec] = {}
    n_sharded = 0
    for name, param in holder._parameters.items():
        if param is None:
            continue
        if not isinstance(param, DTensor):
            # Not under FSDP at all, so there is nothing to gather. Still
            # honour the compute dtype, so the scanned and unrolled paths agree.
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
            if param.shape[1] % mesh.size():
                # Uneven shards. `redistribute` pads around its all-gather, so
                # the unrolled path handles them; a bare collective cannot.
                raise ValueError(
                    f"{name} has dim 1 = {param.shape[1]}, which {mesh.size()} "
                    "ranks cannot shard evenly, and the scan body's all-gather "
                    "needs equal shards"
                )
            sharded = True
            n_sharded += 1
        else:
            # `data_parallel` downgrades Shard to Replicate when dim 1 is
            # smaller than the mesh. Mirror that: no gather, but keep the
            # group, because the backward still has to sum the gradient.
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
