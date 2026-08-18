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
    _is_backward_node,
)
from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    find_fsdp_local_grad_before_reduce_dtype_cast,
    find_fsdp_reduce_grad_input,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    REMAT_REGION_ID_META,
    REMAT_REGION_ROLE_META,
    REMAT_ROLE_BACKWARD,
    REMAT_ROLE_RECOMPUTE,
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


def _ordering_gradient(gradient: torch.fx.Node) -> torch.fx.Node:
    reduce_grad_input = find_fsdp_reduce_grad_input(gradient) or gradient
    module_fqn = gradient.meta.get("custom", {}).get("module_fqn", "")
    if module_fqn != "moe.experts" and not module_fqn.endswith(".moe.experts"):
        return reduce_grad_input
    return (
        find_fsdp_local_grad_before_reduce_dtype_cast(reduce_grad_input)
        or reduce_grad_input
    )


def apply_tpu_remat_schedule(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    num_static_inputs: int,
) -> torch.fx.GraphModule:
    """Keep each layer's parameter gradients adjacent to its recomputation.

    The entry barrier jointly protects the saved activations and incoming
    cotangents from forward/recompute CSE. Adding the next region's materialized
    local parameter gradients completes every consumer of that region's
    recomputed state before XLA can enter the preceding region. Gradient
    synchronization remains free to overlap with the preceding region.

    Every saved primal is barriered. The earliest external backward value
    entering the region anchors it at backward entry; later cotangents retain
    their natural dependencies because they may be produced after local
    backward has begun. Each parameter-gradient output must belong to exactly
    one region. Parameters shared across regions are not supported.

    The barrier uses the next layer's local parameter gradients before FSDP
    synchronization. Expert gradients anchor before the BF16-to-FP32 reduce
    cast, so both the cast and reduce-scatter can overlap the preceding layer's
    backward computation. The collective data path remains unchanged.
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
        region_id = node.meta.get(REMAT_REGION_ID_META)
        role = node.meta.get(REMAT_REGION_ROLE_META)
        if not isinstance(region_id, int):
            continue
        if role == REMAT_ROLE_RECOMPUTE:
            recompute_by_region[region_id].add(node)
        elif role == REMAT_ROLE_BACKWARD:
            backward_by_region[region_id].add(node)

    if not recompute_by_region:
        return gm
    if recompute_by_region.keys() != backward_by_region.keys():
        raise RuntimeError(
            "TPU remat scheduling requires matching recompute and backward regions"
        )

    # GraphTrainer returns [loss, *parameter_grads, *routing_metrics]. Of those
    # outputs, only parameter gradients are backward-region nodes; dInputs are
    # internal because fwd_bwd_step differentiates only model parameters.
    final_dweights_by_region: dict[int, list[torch.fx.Node]] = {}
    for region_id, backward_nodes in backward_by_region.items():
        final_dweights_by_region[region_id] = sorted(
            (
                node
                for node in backward_nodes
                if _is_tensor_node(node)
                and any(user.op == "output" for user in node.users)
            ),
            key=order.__getitem__,
        )

    dweights_by_region = {
        region_id: list(
            dict.fromkeys(
                _ordering_gradient(gradient) for gradient in gradients
            )
        )
        for region_id, gradients in final_dweights_by_region.items()
    }

    region_ids = sorted(recompute_by_region)
    barrier_count = 0
    gradient_count = 0
    for region_index, region_id in enumerate(region_ids):
        recompute_nodes = recompute_by_region[region_id]
        backward_nodes = backward_by_region[region_id]

        # Region metadata labels internal nodes, not their live-ins. Derive the
        # direct activation and cotangent inputs from the FX edges.
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
            region_ids[region_index + 1]
            if region_index + 1 < len(region_ids)
            else None
        )
        ordering_gradients = (
            dweights_by_region.get(next_region_id, [])
            if next_region_id is not None
            else []
        )
        if next_region_id is not None and not ordering_gradients:
            raise RuntimeError(
                f"remat region {next_region_id} has no parameter-gradient outputs"
            )

        boundaries = list(
            dict.fromkeys(primal_boundaries + [cotangent] + ordering_gradients)
        )
        redirect_users: dict[torch.fx.Node, set[torch.fx.Node]] = {
            boundary: {
                user for user in boundary.users if user in recompute_nodes
            }
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
            barrier.meta["val"] = tuple(
                boundary.meta["val"] for boundary in boundaries
            )
            barrier.meta["autograd_backward"] = True
            barrier.meta[REMAT_REGION_ID_META] = region_id

            barrier_outputs: dict[torch.fx.Node, torch.fx.Node] = {}
            for index, boundary in enumerate(boundaries):
                output = gm.graph.call_function(operator.getitem, args=(barrier, index))
                output.meta = boundary.meta.copy()
                output.meta["autograd_backward"] = True
                output.meta[REMAT_REGION_ID_META] = region_id
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
