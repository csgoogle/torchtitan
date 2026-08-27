# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Outline transformer-layer compute into ``invoke_subgraph`` regions."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.fx as fx
from torch.fx.passes.utils.fuser_utils import (
    fuse_by_partitions,
    validate_partition,
)

from torchtitan.experiments.graph_trainer.common_utils import (
    _is_backward_node,
    _MODULE_FQN,
)
from torchtitan.tools.logging import logger


_PREFIX = "__tpu_invoke_subgraph_"
_INFRA_NAMESPACES = {
    "_c10d_functional",
    "bucketing",
    "c10d",
    "c10d_functional",
}


def _layer_id(node: fx.Node) -> int | None:
    fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")
    parts = fqn.split(".", 2)
    if len(parts) < 2 or parts[0] != "layers" or not parts[1].isdigit():
        return None
    return int(parts[1])


def _is_infra_node(node: fx.Node) -> bool:
    if node.target is torch.ops.higher_order.invoke_subgraph:
        return True
    namespace = getattr(node.target, "namespace", None)
    if namespace in _INFRA_NAMESPACES:
        return True
    target = str(node.target)
    return (
        "wait_tensor" in target
        or "all_gather" in target
        or "reduce_scatter" in target
        or "optimization_barrier" in target
    )


def _is_tensor_value(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (tuple, list)):
        return all(_is_tensor_value(item) for item in value)
    return False


def _has_tensor_boundary(nodes: list[fx.Node]) -> bool:
    node_set = set(nodes)
    for node in nodes:
        for input_node in node.all_input_nodes:
            if input_node in node_set:
                continue
            value = input_node.meta.get("val")
            if value is not None and not _is_tensor_value(value):
                return False
        if any(user not in node_set for user in node.users):
            value = node.meta.get("val")
            if value is not None and not _is_tensor_value(value):
                return False
    return True


def _normalize_arg(arg: Any, node_indices: dict[fx.Node, int]) -> Any:
    if isinstance(arg, fx.Node):
        return ("node", node_indices[arg])
    if isinstance(arg, tuple):
        return tuple(_normalize_arg(item, node_indices) for item in arg)
    if isinstance(arg, list):
        return [_normalize_arg(item, node_indices) for item in arg]
    if isinstance(arg, dict):
        return tuple(
            sorted(
                (key, _normalize_arg(value, node_indices)) for key, value in arg.items()
            )
        )
    return repr(arg)


def _value_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), str(value.dtype), tuple(value.stride()))
    if isinstance(value, (tuple, list)):
        return tuple(_value_signature(item) for item in value)
    return repr(value)


def _subgraph_signature(subgraph: fx.GraphModule) -> str:
    nodes = list(subgraph.graph.nodes)
    node_indices = {node: index for index, node in enumerate(nodes)}
    records = []
    for node in nodes:
        records.append(
            (
                node.op,
                None if node.op == "placeholder" else str(node.target),
                _normalize_arg(node.args, node_indices),
                _normalize_arg(node.kwargs, node_indices),
                _value_signature(node.meta.get("val")),
            )
        )
    return hashlib.sha256(repr(records).encode()).hexdigest()[:16]


def _region_direction(subgraph: fx.GraphModule) -> str:
    directions = {
        "backward" if _is_backward_node(node) else "forward"
        for node in subgraph.graph.nodes
        if node.op not in {"placeholder", "output"}
    }
    if len(directions) != 1:
        raise ValueError(
            "invoke_subgraph region must contain only forward or only backward nodes"
        )
    return next(iter(directions))


def _replace_call_modules(gm: fx.GraphModule) -> tuple[int, set[str]]:
    graph = gm.graph
    replaced = 0
    identifiers = set()
    for node in list(graph.nodes):
        if node.op != "call_module" or not str(node.target).startswith(_PREFIX):
            continue
        subgraph = gm.get_submodule(node.target)
        if not isinstance(subgraph, fx.GraphModule):
            raise TypeError(f"Expected GraphModule at {node.target!r}")
        direction = _region_direction(subgraph)
        identifier = f"qwen_layer_{direction}_{_subgraph_signature(subgraph)}"
        identifiers.add(identifier)
        with graph.inserting_before(node):
            subgraph_ref = graph.get_attr(node.target)
            invoke = graph.call_function(
                torch.ops.higher_order.invoke_subgraph,
                args=(subgraph_ref, identifier, *node.args),
                kwargs=node.kwargs,
            )
        invoke.meta = dict(node.meta)
        node.replace_all_uses_with(invoke)
        graph.erase_node(node)
        replaced += 1
    return replaced, identifiers


def outline_transformer_layers_pass(
    gm: fx.GraphModule,
    example_inputs=None,
    *,
    min_region_nodes: int = 2,
) -> fx.GraphModule:
    """Outline each layer's forward and backward compute after SAC/FSDP passes."""
    del example_inputs
    input_node_count = len(list(gm.graph.nodes))
    grouped: dict[tuple[int, bool], list[fx.Node]] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_infra_node(node):
            continue
        layer_id = _layer_id(node)
        if layer_id is None:
            continue
        grouped.setdefault((layer_id, _is_backward_node(node)), []).append(node)

    partitions = []
    skipped = []
    for key, nodes in grouped.items():
        if len(nodes) < min_region_nodes:
            skipped.append((key, "too small"))
            continue
        if not _has_tensor_boundary(nodes):
            skipped.append((key, "non-tensor boundary"))
            continue
        if not validate_partition(nodes):
            skipped.append((key, "dependency cycle"))
            continue
        partitions.append(dict.fromkeys(nodes))

    if not partitions:
        logger.warning(
            "invoke_subgraph outlining found no eligible transformer regions; "
            "skipped=%s",
            skipped,
        )
        return gm

    gm = fuse_by_partitions(
        gm,
        partitions,
        prefix=_PREFIX,
        always_return_tuple=True,
    )
    replaced, identifiers = _replace_call_modules(gm)
    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Outlined %d transformer regions as %d unique invoke_subgraph bodies "
        "(%d -> %d outer graph nodes); skipped=%s",
        replaced,
        len(identifiers),
        input_node_count,
        len(list(gm.graph.nodes)),
        skipped,
    )
    return gm


__all__ = ["outline_transformer_layers_pass"]
