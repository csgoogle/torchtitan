# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU-only GraphTrainer memory policies."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node
from torchtitan.experiments.graph_trainer.cpu_offload import (
    tag_all_offloadable_activations,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _configured_fsdp_force_save_nodes,
    _full_memory_policy_pass,
    _make_full_memory_policy,
    tag_sac_policy,
)
from torchtitan.experiments.graph_trainer.registry import register_memory_policy

# Semantic hints that model code attaches with ``torch.fx.traceback.annotate_fn``
# so a policy can find values that op names alone cannot tell apart. There is
# one key, and the values are named after upstream ``RegionAC`` regions so that
# eager and graph AC can later share one vocabulary.
REMAT_REGION = "remat_region"
MOE_ROUTE_PERMUTATION = "moe.route_permutation"


@dataclass(frozen=True)
class SaveCandidateClass:
    """One kind of forward value that a TPU selective policy may keep.

    Attributes:
        name: The name used in ``graph_trainer_save_regions``.
        select: Returns this class's candidate forward nodes, in forward order.
        required: Raise if the graph has no candidates. A model or tracing
            change that hides the values then fails loudly instead of quietly
            recomputing them.
        budgeted: ``False`` means always save. Those classes are charged
            against the budget first.
    """

    name: str
    select: Callable[[torch.fx.GraphModule], list[torch.fx.Node]]
    required: bool
    budgeted: bool


def _forward_call_nodes(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    return [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and not _is_backward_node(node)
    ]


def _in_remat_region(node: torch.fx.Node, region: str) -> bool:
    return node.meta.get("custom", {}).get(REMAT_REGION) == region


def _is_integer_vector(value: object) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.ndim == 1
        and value.dtype in (torch.int32, torch.int64)
    )


def select_route_permutations(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    """Return the expert-sort permutations that ``local_reorder`` marks.

    ``token_dispatcher.py`` has three argsorts, so an op or module FQN alone
    cannot pick this one; the dispatcher marks it with a region hint instead.
    Every node traced inside the hinted function carries the hint (the sort
    and its getitems), so keep the one-dimensional integer value that leaves
    the region: the permutation itself.
    """
    return [
        node
        for node in _forward_call_nodes(gm)
        if _in_remat_region(node, MOE_ROUTE_PERMUTATION)
        and _is_integer_vector(node.meta.get("val"))
        and any(
            not _in_remat_region(user, MOE_ROUTE_PERMUTATION) for user in node.users
        )
    ]


# Saving the permutation avoids re-running a stable sort per layer in backward.
# It is small, so it is always saved rather than competing for the budget.
ROUTE_PERMUTATION_CLASS = SaveCandidateClass(
    name=MOE_ROUTE_PERMUTATION,
    select=select_route_permutations,
    required=True,
    budgeted=False,
)

SAVE_REGIONS: dict[str, SaveCandidateClass] = {
    candidate_class.name: candidate_class
    for candidate_class in (ROUTE_PERMUTATION_CLASS,)
}


@register_memory_policy("full_and_offload")
def _full_and_offload_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: "GraphTrainer.Config",
) -> torch.fx.GraphModule:
    """Full recompute plus CPU offload of saved layer boundaries."""
    _full_memory_policy_pass(gm, config=config)
    tag_all_offloadable_activations(
        gm,
        cpu_budget_gb=config.compile.cpu_offload_budget_gb,
    )
    return gm


def full_remat_with_saves(
    gm: torch.fx.GraphModule,
    config: "GraphTrainer.Config",
    save_nodes: Iterable[torch.fx.Node],
) -> torch.fx.GraphModule:
    """Recompute everything except ``save_nodes`` and layer boundaries.

    Every TPU selective policy tags the graph through this one function, so
    none of them can drop the FSDP contract: with
    ``reshard_after_forward=False`` (ZeRO-2) the all-gather outputs stay
    ``MUST_SAVE`` whatever else the policy chose to keep.
    """
    force_save_nodes = set(save_nodes)
    force_save_nodes |= _configured_fsdp_force_save_nodes(gm, config) or set()
    return tag_sac_policy(
        gm,
        policy_fn=_make_full_memory_policy(),
        force_save_nodes=force_save_nodes,
    )
