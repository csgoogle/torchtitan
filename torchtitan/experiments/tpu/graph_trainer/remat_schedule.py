# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU scheduling constraints for GraphTrainer activation rematerialization."""

from __future__ import annotations

import operator
from collections import defaultdict

import torch

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
    _NOT_IN_LAYERS,
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    find_fsdp_reduce_grad_input,
)
from torchtitan.tools.logging import logger


def _optimization_barrier(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if not values:
        return ()
    device_types = {value.device.type for value in values}
    if device_types == {"cpu"}:
        return values
    if device_types != {"tpu"}:
        raise ValueError(
            "TPU optimization barriers require all-CPU test inputs or all-TPU "
            f"runtime inputs, got device types {sorted(device_types)}"
        )
    return tuple(torch.ops.tpu.optimization_barrier.default(list(values)))


def _is_tensor_node(node: torch.fx.Node) -> bool:
    return isinstance(node.meta.get("val"), torch.Tensor)


def _find_local_grad_before_reduce_dtype_cast(
    reduce_grad_input: torch.fx.Node,
) -> torch.fx.Node | None:
    """Match the expert BF16 dWeight before its FP32 FSDP reduction cast."""
    node = reduce_grad_input
    while hasattr(node.target, "is_view") and node.target.is_view:
        if len(node.all_input_nodes) != 1:
            return None
        node = node.all_input_nodes[0]

    if (
        node.target is not torch.ops.aten._to_copy.default
        or len(node.all_input_nodes) != 1
    ):
        return None

    source = node.all_input_nodes[0]
    source_val = source.meta.get("val")
    cast_val = node.meta.get("val")
    if not isinstance(source_val, torch.Tensor) or not isinstance(
        cast_val, torch.Tensor
    ):
        return None
    if (
        source_val.dtype != torch.bfloat16
        or cast_val.dtype != torch.float32
        or source_val.shape != cast_val.shape
        or source_val.device != cast_val.device
    ):
        return None
    return source


def _parameter_layer_id(parameter_fqns: tuple[str, ...]) -> int:
    layer_ids = set()
    for fqn in parameter_fqns:
        parts = fqn.split(".")
        if len(parts) >= 2 and parts[0] == "layers":
            try:
                layer_ids.add(int(parts[1]))
            except ValueError:
                pass
    if len(layer_ids) > 1:
        raise RuntimeError(
            "TPU remat scheduling does not support parameters shared across "
            f"remat regions: {parameter_fqns}"
        )
    return next(iter(layer_ids), _NOT_IN_LAYERS)


def _ordering_gradient(
    gradient: torch.fx.Node,
    parameter_fqns: tuple[str, ...],
) -> torch.fx.Node:
    reduce_grad_input = find_fsdp_reduce_grad_input(gradient) or gradient
    if not any(".moe.experts." in f".{fqn}." for fqn in parameter_fqns):
        return reduce_grad_input

    # Depending on the BF16 local dWeight lets its FP32 cast and reduce-scatter
    # overlap the preceding layer's backward computation. This was measured
    # only for the large expert gradients; if the same ordering point is
    # neutral or beneficial for other parameters, generalize it and remove the
    # parameter-name special case.
    return (
        _find_local_grad_before_reduce_dtype_cast(reduce_grad_input)
        or reduce_grad_input
    )


def apply_tpu_remat_schedule(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    num_static_inputs: int,
) -> torch.fx.GraphModule:
    """Keep each layer's parameter gradients adjacent to its recomputation.

    Selective rematerialization labels the recompute and backward nodes for
    each transformer layer. For adjacent layers 0 and 1, the relevant graph is
    conceptually::

        saved_x1 -> recompute_1 -> backward_1 -> (dw1, dx1)
        saved_x0 ------------------------------> recompute_0(dx1) -> backward_0

    XLA may otherwise schedule ``recompute_1`` early, follow ``dx1`` into layer
    0, and defer ``dw1``. That keeps layer 1's recomputed workspace live while
    layer 0 is running. This pass rewrites only the inputs to layer 0's
    recompute/backward entry::

        saved_x0_barrier, dx1_barrier, _ = optimization_barrier(
            saved_x0, dx1, dw1
        )
        recompute_0(saved_x0_barrier)
        backward_0(dx1_barrier)

    The tuple barrier does not reshape or numerically change any tensor. The
    inserted ``getitem`` nodes preserve each input's original shape and dtype;
    they only make ``dw1`` an XLA scheduling dependency of layer 0's entry.
    Thus layer 1 must finish its local parameter gradients before layer 0 can
    start, while the later FSDP cast and reduce-scatter may still overlap with
    layer 0.

    Every saved primal is barriered. The earliest external backward value
    entering the region anchors it at backward entry; later cotangents retain
    their natural dependencies because they may be produced after local
    backward has begun. Each tagged parameter gradient must belong to exactly
    one region. Parameters shared across regions are not supported.

    The barrier uses the next layer's local parameter gradients before FSDP
    synchronization. The currently validated expert path anchors before the
    BF16-to-FP32 reduce cast, so both the cast and reduce-scatter can overlap
    the preceding layer's backward computation. Other parameters retain the
    generic FSDP split at the collective input pending equivalent performance
    validation. The collective data path remains unchanged.
    """
    del example_inputs

    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    static_placeholders = set(placeholders[:num_static_inputs])
    nonstatic_cache: dict[torch.fx.Node, bool] = {}

    # Nonstatic values depend on runtime batch inputs rather than only model
    # state. Only those activation boundaries should anchor rematerialization.
    def depends_on_nonstatic_input(node: torch.fx.Node) -> bool:
        cached = nonstatic_cache.get(node)
        if cached is not None:
            return cached
        if node.op == "placeholder":
            result = node not in static_placeholders
        elif node.op == "get_attr":
            result = False
        else:
            result = any(
                depends_on_nonstatic_input(inp) for inp in node.all_input_nodes
            )
        nonstatic_cache[node] = result
        return result

    recompute_by_region: dict[int, set[torch.fx.Node]] = defaultdict(set)
    backward_by_region: dict[int, set[torch.fx.Node]] = defaultdict(set)
    for node in gm.graph.nodes:
        region_id = _get_layer_id(node)
        if region_id == _NOT_IN_LAYERS:
            continue
        if node.name.endswith("_recomputed"):
            recompute_by_region[region_id].add(node)
        elif _is_backward_node(node):
            backward_by_region[region_id].add(node)

    if not recompute_by_region:
        return gm
    if recompute_by_region.keys() != backward_by_region.keys():
        raise RuntimeError(
            "TPU remat scheduling requires matching recompute and backward regions"
        )

    tagged_gradients_by_region: dict[
        int, list[tuple[torch.fx.Node, tuple[str, ...]]]
    ] = defaultdict(list)
    gradient_regions: dict[torch.fx.Node, set[int]] = defaultdict(set)
    gradient_fqns: dict[torch.fx.Node, list[str]] = defaultdict(list)
    for gradient in gm.graph.nodes:
        parameter_fqns = gradient.meta.get("custom", {}).get(
            PARAMETER_GRADIENT_FQNS_META
        )
        if parameter_fqns is None:
            continue
        if not isinstance(parameter_fqns, tuple) or not all(
            isinstance(parameter_fqn, str) for parameter_fqn in parameter_fqns
        ):
            raise RuntimeError(
                "Parameter-gradient metadata must be a tuple of FQNs, got "
                f"{parameter_fqns!r} on {gradient.name}"
            )
        region_id = _parameter_layer_id(parameter_fqns)
        if region_id in backward_by_region:
            tagged_gradients_by_region[region_id].append((gradient, parameter_fqns))
            gradient_regions[gradient].add(region_id)
            gradient_fqns[gradient].extend(parameter_fqns)

    shared_across_regions = {
        gradient.name: tuple(gradient_fqns[gradient])
        for gradient, regions in gradient_regions.items()
        if len(regions) > 1
    }
    if shared_across_regions:
        raise RuntimeError(
            "TPU remat scheduling does not support parameter-gradient values "
            f"shared across remat regions: {shared_across_regions}"
        )

    dweights_by_region = {
        region_id: list(
            dict.fromkeys(
                _ordering_gradient(gradient, parameter_fqns)
                for gradient, parameter_fqns in sorted(
                    gradients, key=lambda item: order[item[0]]
                )
            )
        )
        for region_id, gradients in tagged_gradients_by_region.items()
    }

    region_ids = sorted(recompute_by_region)
    barrier_count = 0
    gradient_count = 0
    for region_index, region_id in enumerate(region_ids):
        recompute_nodes = recompute_by_region[region_id]
        backward_nodes = backward_by_region[region_id]

        # Region metadata labels internal nodes, not their live-ins. Derive the
        # saved activations and incoming cotangent from edges crossing into the
        # recompute/backward region.
        primal_boundaries = sorted(
            {
                inp
                for node in recompute_nodes
                for inp in node.all_input_nodes
                if inp not in recompute_nodes
                and _is_tensor_node(inp)
                and depends_on_nonstatic_input(inp)
            },
            key=order.__getitem__,
        )
        cotangent_candidates = sorted(
            {
                inp
                for node in backward_nodes
                for inp in node.all_input_nodes
                if inp not in recompute_nodes
                and inp not in backward_nodes
                and _is_backward_node(inp)
                and _is_tensor_node(inp)
            },
            key=order.__getitem__,
        )
        if not primal_boundaries:
            raise RuntimeError(f"remat region {region_id} has no saved activation")
        if not cotangent_candidates:
            raise RuntimeError(
                f"remat region {region_id} has no incoming backward value"
            )
        cotangent = cotangent_candidates[0]
        next_region_id = (
            region_ids[region_index + 1] if region_index + 1 < len(region_ids) else None
        )
        ordering_gradients = (
            dweights_by_region.get(next_region_id, [])
            if next_region_id is not None
            else []
        )
        if next_region_id is not None and not ordering_gradients:
            raise RuntimeError(
                f"remat region {next_region_id} has no parameter-gradient markers"
            )

        # The next layer's local gradients are dependency-only barrier inputs;
        # only the saved primals and cotangent have users redirected through
        # their corresponding tuple outputs.
        boundaries = list(
            dict.fromkeys(primal_boundaries + [cotangent] + ordering_gradients)
        )
        redirect_users: dict[torch.fx.Node, set[torch.fx.Node]] = {
            boundary: {user for user in boundary.users if user in recompute_nodes}
            for boundary in primal_boundaries
        }
        redirect_users[cotangent] = {
            user for user in cotangent.users if _is_backward_node(user)
        }
        redirect_users = {
            boundary: users for boundary, users in redirect_users.items() if users
        }
        first_user = min(
            (user for users in redirect_users.values() for user in users),
            key=order.__getitem__,
        )
        late_boundaries = [
            boundary.name
            for boundary in boundaries
            if order[boundary] >= order[first_user]
        ]
        if late_boundaries:
            raise RuntimeError(
                f"remat region {region_id} has boundary values unavailable before "
                f"{first_user.name}: {late_boundaries}"
            )

        with gm.graph.inserting_before(first_user):
            barrier = gm.graph.call_function(
                _optimization_barrier,
                args=tuple(boundaries),
            )
            barrier.meta["val"] = tuple(boundary.meta["val"] for boundary in boundaries)
            barrier.meta["autograd_backward"] = True

            barrier_outputs: dict[torch.fx.Node, torch.fx.Node] = {}
            for index, boundary in enumerate(boundaries):
                output = gm.graph.call_function(operator.getitem, args=(barrier, index))
                output.meta = boundary.meta.copy()
                output.meta["autograd_backward"] = True
                barrier_outputs[boundary] = output

        for boundary, output in barrier_outputs.items():
            for user in redirect_users.get(boundary, ()):
                user.replace_input_with(boundary, output)

        barrier_count += 1
        gradient_count += len(ordering_gradients)

    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Inserted %d TPU remat barriers ordering %d parameter gradients",
        barrier_count,
        gradient_count,
    )
    return gm


__all__ = ["apply_tpu_remat_schedule"]
