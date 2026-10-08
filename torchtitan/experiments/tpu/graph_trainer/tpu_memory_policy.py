# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU-only GraphTrainer memory policies."""

import operator
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _get_module_fqn,
    _is_backward_node,
    _NOT_IN_LAYERS,
    matches_module_fqn_pattern,
)
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
from torchtitan.experiments.tpu.remat_regions import (
    ATTENTION_INNER_ATTENTION,
    MOE_ROUTE_PERMUTATION,
    MOE_ROUTING_DECISION,
    REMAT_REGION,
)
from torchtitan.tools.logging import logger


@dataclass(frozen=True)
class SaveCandidateClass:
    """One kind of forward value that a TPU selective policy may keep.

    Attributes:
        name: The name used in ``graph_trainer_save_regions``.
        select: Returns this class's candidate forward nodes, in forward order.
    """

    name: str
    select: Callable[[torch.fx.GraphModule], list[torch.fx.Node]]


def _forward_call_nodes(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    return [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and not _is_backward_node(node)
    ]


def _in_remat_region(node: torch.fx.Node, region: str) -> bool:
    return node.meta.get("custom", {}).get(REMAT_REGION) == region


def _is_integer_tensor(value: object, ndim: int) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.ndim == ndim
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
        and _is_integer_tensor(node.meta.get("val"), ndim=1)
        and any(
            not _in_remat_region(user, MOE_ROUTE_PERMUTATION) for user in node.users
        )
    ]


# Saving the permutation avoids re-running a stable sort per layer in backward.
# It is small (one integer per routed token), so recipes save it in every layer.
ROUTE_PERMUTATION_CLASS = SaveCandidateClass(
    name=MOE_ROUTE_PERMUTATION,
    select=select_route_permutations,
)

# The module that runs ``TokenChoiceTopKRouter._select_experts``.
_ROUTER_FQN_PATTERN = "layers.*.moe.router"


def select_routing_decisions(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    """Return each MoE router's top-k expert ids.

    ``TokenChoiceTopKRouter._select_experts`` ends in ``torch.topk``, whose
    indices are the routing decision. Group-limited routing runs group
    top-ks before it, so keep the indices of the last top-k in each router.
    Raises if they are not one 2-D integer value, so a router change fails
    loudly instead of silently saving the wrong thing.
    """
    last_topk: dict[str, torch.fx.Node] = {}
    for node in _forward_call_nodes(gm):
        fqn = _get_module_fqn(node)
        if node.target is torch.ops.aten.topk.default and (
            matches_module_fqn_pattern(_ROUTER_FQN_PATTERN, fqn)
        ):
            last_topk[fqn] = node
    decisions = set()
    for topk in last_topk.values():
        indices = [
            user
            for user in topk.users
            if user.target is operator.getitem and user.args[1] == 1
        ]
        if len(indices) != 1 or not _is_integer_tensor(
            indices[0].meta.get("val"), ndim=2
        ):
            raise ValueError(
                f"Router top-k {topk.name} does not have one 2-D integer indices "
                "value; update select_routing_decisions for this router."
            )
        decisions.add(indices[0])
    return [node for node in _forward_call_nodes(gm) if node in decisions]


# Saving the top-k expert ids keeps backward on the forward's routing: a
# recomputed top-k can pick different experts when scores (nearly) tie.
# Upstream eager AC always keeps this region (``recompute=False``) for the
# same reason. It also skips the top-k recompute, and it is small (top_k
# integers per token), so recipes save it in every layer.
ROUTING_DECISION_CLASS = SaveCandidateClass(
    name=MOE_ROUTING_DECISION,
    select=select_routing_decisions,
)

# The forward ops of both TPU Splash kernels. Stock Splash
# (``kernels/splash_attention.py``) registers one op per kernel configuration,
# ``pallas::splash_attn_fwd_<16 hex digits>``; Tokamax Splash
# (``kernels/tokamax_splash_attention.py``) registers
# ``pallas::tokamax_splash_attn_fwd``.
_SPLASH_FORWARD_OP_NAME = re.compile(
    r"pallas::(?:splash_attn_fwd_[0-9a-f]{16}|tokamax_splash_attn_fwd)"
)


def is_splash_forward_op_name(name: str) -> bool:
    """Whether ``name`` (``namespace::op``) is a Splash forward op."""
    return _SPLASH_FORWARD_OP_NAME.fullmatch(name) is not None


def _is_splash_forward_op(target: object) -> bool:
    return isinstance(target, torch._ops.OpOverload) and is_splash_forward_op_name(
        target.name()
    )


def _is_splash_output(value: object) -> bool:
    """Whether ``value`` is a (4-D BF16 output, 3-D FP32 LSE) pair.

    The LSE is (batch, heads, seq) for both kernels. Stock Splash returns the
    output in kernel layout (batch, heads, seq, head_dim); Tokamax returns it
    in model layout (batch, seq, heads, head_dim).
    """
    if not (isinstance(value, (tuple, list)) and len(value) == 2):
        return False
    output, lse = value
    if not (
        isinstance(output, torch.Tensor)
        and isinstance(lse, torch.Tensor)
        and output.ndim == 4
        and lse.ndim == 3
        and output.dtype == torch.bfloat16
        and lse.dtype == torch.float32
    ):
        return False
    kernel_layout = tuple(output.shape[:3])
    model_layout = (output.shape[0], output.shape[2], output.shape[1])
    return tuple(lse.shape) in (kernel_layout, model_layout)


def select_splash_attention(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    """Return the Splash forward nodes, whose output and LSE backward reads.

    Matched by op rather than by module FQN: the Splash patches install their
    module or forward after ``annotate_module_fqns`` has run, so the op carries
    its parent's ``layers.N.attention`` FQN, not
    ``layers.N.attention.inner_attention``. Raises if a Splash op does not
    return the expected pair, so a kernel change fails loudly instead of
    silently saving the wrong thing.
    """
    nodes = []
    for node in _forward_call_nodes(gm):
        if not _is_splash_forward_op(node.target):
            continue
        if not _is_splash_output(node.meta.get("val")):
            raise ValueError(
                f"Splash forward {node.name} ({node.target}) does not return a "
                "(4-D BF16 output, 3-D FP32 LSE) pair; update "
                "select_splash_attention for this kernel."
            )
        nodes.append(node)
    return nodes


# Saving the output and LSE lets backward skip re-running the Splash forward.
# It is large (and the LSE is padded on device), so recipes may save it only in
# the last few layers with ``graph_trainer_save_last_n_layers``.
#
# This is not the same as an eager ``RegionAC`` save region of the same name.
# Here only what the Splash backward reads is saved (output and LSE); its
# q/k/v inputs are still recomputed from the layer boundary. An eager save
# region also keeps the region's inputs. The difference is on purpose, so
# don't "align" the two.
SPLASH_ATTENTION_CLASS = SaveCandidateClass(
    name=ATTENTION_INNER_ATTENTION,
    select=select_splash_attention,
)

SAVE_REGIONS: dict[str, SaveCandidateClass] = {
    candidate_class.name: candidate_class
    for candidate_class in (
        ROUTE_PERMUTATION_CLASS,
        ROUTING_DECISION_CLASS,
        SPLASH_ATTENTION_CLASS,
    )
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


def _saved_bytes(node: torch.fx.Node) -> int:
    """Bytes of a node's outputs, from the fake values FX recorded.

    These are logical sizes (numel times element size); on-device padding or
    tiling is not counted.
    """
    value = node.meta["val"]
    values = value if isinstance(value, (tuple, list)) else (value,)
    return sum(
        int(v.numel()) * v.element_size()
        for v in values
        if isinstance(v, torch.Tensor)
    )


def _log_region(name: str, selected: int, candidates: int, nbytes: int) -> None:
    logger.info(
        f"tpu_selective: {name}: saving {selected}/{candidates} candidates, "
        f"{nbytes / 1024**3:.3f} GiB (logical)"
    )


def _in_last_n_layers(
    nodes: list[torch.fx.Node], name: str, n_layers: int
) -> list[torch.fx.Node]:
    """Keep the ``nodes`` in the ``n_layers`` highest-numbered layers."""
    layer_ids = {_get_layer_id(node) for node in nodes}
    if _NOT_IN_LAYERS in layer_ids:
        raise ValueError(
            f"graph_trainer_save_last_n_layers limits {name!r}, but some of its "
            "values are outside the transformer layers."
        )
    kept_layer_ids = set(sorted(layer_ids)[-n_layers:])
    return [node for node in nodes if _get_layer_id(node) in kept_layer_ids]


def select_saves(
    gm: torch.fx.GraphModule,
    regions: Sequence[SaveCandidateClass],
    last_n_layers: Mapping[str, int],
) -> set[torch.fx.Node]:
    """Choose which candidate nodes of ``regions`` to save.

    A region named in ``last_n_layers`` is saved only in its last N layers:
    backward reaches later layers first, so their values are held for the
    shortest time. Other regions are saved in every layer.

    Raises if a region matches no forward nodes, so a model or tracing change
    that hides its values fails loudly instead of quietly recomputing them.
    """
    selected: set[torch.fx.Node] = set()
    for region in regions:
        candidates = region.select(gm)
        if not candidates:
            raise ValueError(
                f"Save region {region.name!r} matched no forward nodes; the "
                "model or its tracing no longer exposes it."
            )
        nodes = candidates
        if region.name in last_n_layers:
            nodes = _in_last_n_layers(nodes, region.name, last_n_layers[region.name])
        selected.update(nodes)
        region_bytes = sum(_saved_bytes(node) for node in nodes)
        _log_region(region.name, len(nodes), len(candidates), region_bytes)
    total_bytes = sum(_saved_bytes(node) for node in selected)
    logger.info(
        f"tpu_selective: saving {total_bytes / 1024**3:.3f} GiB in total (logical)"
    )
    return selected


def resolve_save_regions(names: Sequence[str]) -> list[SaveCandidateClass]:
    """Map ``graph_trainer_save_regions`` to classes, rejecting bad names."""
    known = ", ".join(sorted(SAVE_REGIONS))
    if not names:
        raise ValueError(
            "graph_trainer_memory_policy='tpu_selective' needs "
            f"graph_trainer_save_regions; known regions: {known}."
        )
    unknown = [name for name in names if name not in SAVE_REGIONS]
    if unknown:
        raise ValueError(f"Unknown save regions {unknown}; known regions: {known}.")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate entries in graph_trainer_save_regions: {names}.")
    return [SAVE_REGIONS[name] for name in names]


def check_save_last_n_layers(
    last_n_layers: Mapping[str, int], save_regions: Sequence[str]
) -> None:
    """Reject ``graph_trainer_save_last_n_layers`` entries that cannot apply."""
    for name, n_layers in last_n_layers.items():
        if name not in save_regions:
            raise ValueError(
                f"graph_trainer_save_last_n_layers names {name!r}, which is not "
                f"in graph_trainer_save_regions {list(save_regions)}."
            )
        if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers < 1:
            raise ValueError(
                f"graph_trainer_save_last_n_layers[{name!r}] must be a positive "
                f"integer, got {n_layers!r}."
            )


@register_memory_policy("tpu_selective")
def _tpu_selective_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: "GraphTrainer.Config",
) -> torch.fx.GraphModule:
    """Full recompute, except the configured save regions."""
    tpu_config = config.tpu_config
    regions = resolve_save_regions(tpu_config.graph_trainer_save_regions)
    last_n_layers = tpu_config.graph_trainer_save_last_n_layers
    check_save_last_n_layers(last_n_layers, tpu_config.graph_trainer_save_regions)
    full_remat_with_saves(gm, config, select_saves(gm, regions, last_n_layers))
    return gm
