# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest

import torch
from torch import nn

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
    annotate_module_fqns,
    compute_parameter_gradients,
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _make_default_memory_policy,
    _make_full_memory_policy,
    tag_sac_policy,
)
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    remove_parameter_gradient_markers_pass,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import _FSDP_BUCKET_META
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    _TPU_FSDP_RS_DEADLINE_META,
    _TPU_REMAT_BARRIER_KIND_META,
    _TPU_REMAT_LAYER_ENTRY_ANCHOR_META,
    _TPU_REMAT_REGION_ID_META,
    _optimization_barrier,
    apply_tpu_fsdp_rs_deadline_schedule,
    apply_tpu_remat_fsdp_all_gather_schedule,
    apply_tpu_remat_schedule,
)


def _set_tensor_meta(
    node: torch.fx.Node,
    *,
    layer_id: int | None = None,
    backward: bool = False,
    recomputed: bool = False,
) -> torch.fx.Node:
    node.meta["val"] = torch.empty(4, 4, dtype=torch.bfloat16)
    if layer_id is not None:
        node.meta["custom"] = {"module_fqn": f"layers.{layer_id}.linear"}
    if backward:
        node.meta["autograd_backward"] = True
    if recomputed:
        node.name = f"{node.name}_recomputed"
    return node


def _make_fsdp_remat_graph() -> tuple[
    torch.fx.GraphModule,
    list[torch.fx.Node],
    list[torch.fx.Node],
    torch.fx.Node,
    dict[int, torch.fx.Node],
    dict[int, torch.fx.Node],
]:
    graph = torch.fx.Graph()
    weights = [_set_tensor_meta(graph.placeholder(f"weight_{i}")) for i in range(3)]
    saved = [_set_tensor_meta(graph.placeholder(f"saved_{i}")) for i in range(3)]
    top_cotangent = _set_tensor_meta(
        graph.placeholder("top_cotangent"), backward=True
    )

    forward_casts = []
    forward = saved[0]
    for layer_id in range(3):
        cast = _set_tensor_meta(
            graph.call_function(
                torch.ops.aten._to_copy.default,
                args=(weights[layer_id],),
                kwargs={"dtype": torch.bfloat16},
            ),
            layer_id=layer_id,
        )
        forward_casts.append(cast)
        all_gather = _set_tensor_meta(
            graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(cast, 1, "fsdp"),
            ),
            layer_id=layer_id,
        )
        wait = _set_tensor_meta(
            graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(all_gather,),
            ),
            layer_id=layer_id,
        )
        forward = _set_tensor_meta(
            graph.call_function(torch.ops.aten.mm.default, args=(forward, wait)),
            layer_id=layer_id,
        )
    loss = graph.call_function(torch.ops.aten.sum.default, args=(forward,))
    loss.meta["val"] = torch.empty((), dtype=torch.float32)

    recompute_casts = {}
    layer_entry_anchors = {}
    incoming_cotangent = top_cotangent
    for layer_id in reversed(range(3)):
        entry_barrier = graph.call_function(
            _optimization_barrier,
            args=(saved[layer_id], incoming_cotangent),
        )
        entry_barrier.meta["val"] = (
            saved[layer_id].meta["val"],
            incoming_cotangent.meta["val"],
        )
        saved_entry = _set_tensor_meta(
            graph.call_function(operator.getitem, args=(entry_barrier, 0)),
            backward=True,
        )
        cotangent_entry = _set_tensor_meta(
            graph.call_function(operator.getitem, args=(entry_barrier, 1)),
            backward=True,
        )
        cotangent_entry.meta[_TPU_REMAT_LAYER_ENTRY_ANCHOR_META] = layer_id
        layer_entry_anchors[layer_id] = cotangent_entry
        cast = _set_tensor_meta(
            graph.call_function(
                torch.ops.aten._to_copy.default,
                args=(weights[layer_id],),
                kwargs={"dtype": torch.bfloat16},
            ),
            layer_id=layer_id,
            backward=True,
            recomputed=True,
        )
        recompute_casts[layer_id] = cast
        all_gather = _set_tensor_meta(
            graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(cast, 1, "fsdp"),
            ),
            layer_id=layer_id,
            backward=True,
            recomputed=True,
        )
        wait = _set_tensor_meta(
            graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(all_gather,),
            ),
            layer_id=layer_id,
            backward=True,
            recomputed=True,
        )
        recompute = _set_tensor_meta(
            graph.call_function(torch.ops.aten.mm.default, args=(saved_entry, wait)),
            layer_id=layer_id,
            backward=True,
            recomputed=True,
        )
        incoming_cotangent = _set_tensor_meta(
            graph.call_function(
                torch.ops.aten.mm.default,
                args=(recompute, cotangent_entry),
            ),
            layer_id=layer_id,
            backward=True,
        )

    graph.output((loss, incoming_cotangent))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    gm.graph.lint()
    gm.recompile()
    return gm, weights, forward_casts, loss, recompute_casts, layer_entry_anchors


class TestTpuFsdpRematSchedule(unittest.TestCase):
    def test_all_gather_barrier_uses_configured_layer_distance(self) -> None:
        for distance, expected_anchor_ids in (
            (1, (1, 2, None)),
            (2, (2, None, None)),
        ):
            with self.subTest(distance=distance):
                (
                    gm,
                    weights,
                    forward_casts,
                    loss,
                    recompute_casts,
                    layer_entry_anchors,
                ) = _make_fsdp_remat_graph()

                apply_tpu_remat_fsdp_all_gather_schedule(
                    gm,
                    (),
                    num_static_inputs=3,
                    prefetch_distance=distance,
                )

                barriers = [
                    node
                    for node in gm.graph.nodes
                    if node.meta.get(_TPU_REMAT_BARRIER_KIND_META)
                    == "fsdp_all_gather"
                    and node.target is _optimization_barrier
                ]
                self.assertEqual(len(barriers), 3)
                barrier_by_weight = {barrier.args[0]: barrier for barrier in barriers}
                for layer_id, weight in enumerate(weights):
                    expected_anchor_id = expected_anchor_ids[layer_id]
                    expected_anchor = (
                        loss
                        if expected_anchor_id is None
                        else layer_entry_anchors[expected_anchor_id]
                    )
                    barrier = barrier_by_weight[weight]
                    self.assertIs(barrier.args[1], expected_anchor)
                    barrier_output = next(
                        user
                        for user in barrier.users
                        if user.target is operator.getitem and user.args[1] == 0
                    )
                    self.assertIs(recompute_casts[layer_id].args[0], barrier_output)
                    self.assertIs(forward_casts[layer_id].args[0], weight)

                graph_after_first_application = str(gm.graph)
                apply_tpu_remat_fsdp_all_gather_schedule(
                    gm,
                    (),
                    num_static_inputs=3,
                    prefetch_distance=distance,
                )
                self.assertEqual(str(gm.graph), graph_after_first_application)

    def test_all_gather_barrier_requires_positive_prefetch_distance(self) -> None:
        gm, *_ = _make_fsdp_remat_graph()

        with self.assertRaisesRegex(ValueError, "must be positive"):
            apply_tpu_remat_fsdp_all_gather_schedule(
                gm,
                (),
                num_static_inputs=3,
                prefetch_distance=0,
            )

    def test_all_gather_barrier_ignores_activation_collectives(self) -> None:
        graph = torch.fx.Graph()
        _set_tensor_meta(graph.placeholder("weight"))
        activation = _set_tensor_meta(graph.placeholder("activation"))
        all_gather = _set_tensor_meta(
            graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(activation, 1, "tp"),
            ),
            layer_id=0,
            backward=True,
            recomputed=True,
        )
        graph.output(all_gather)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        apply_tpu_remat_fsdp_all_gather_schedule(gm, (), num_static_inputs=1)

        self.assertFalse(
            any(
                node.meta.get(_TPU_REMAT_BARRIER_KIND_META) == "fsdp_all_gather"
                for node in gm.graph.nodes
            )
        )
        self.assertIs(all_gather.args[0], activation)

    def test_rs_deadline_reuses_all_gather_barrier(self) -> None:
        graph = torch.fx.Graph()
        shard_2 = _set_tensor_meta(graph.placeholder("shard_2"))
        shard_1 = _set_tensor_meta(graph.placeholder("shard_1"))
        activation = _set_tensor_meta(graph.placeholder("activation"))

        barrier_2 = graph.call_function(
            _optimization_barrier,
            args=(shard_2, activation),
        )
        barrier_2.meta["val"] = (shard_2.meta["val"], activation.meta["val"])
        barrier_2.meta[_TPU_REMAT_BARRIER_KIND_META] = "fsdp_all_gather"
        barrier_2.meta[_TPU_REMAT_REGION_ID_META] = 2
        shard_2_ready = _set_tensor_meta(
            graph.call_function(operator.getitem, args=(barrier_2, 0)),
            backward=True,
        )
        _set_tensor_meta(
            graph.call_function(operator.getitem, args=(barrier_2, 1)),
            backward=True,
        )
        ag_2 = _set_tensor_meta(
            graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(shard_2_ready, 1, "fsdp"),
            ),
            layer_id=2,
            backward=True,
        )
        ag_2.meta[_FSDP_BUCKET_META] = {
            "plan_fqns": ("layers.2",),
            "direction": "bwd",
        }
        ag_2_wait = _set_tensor_meta(
            graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(ag_2,),
            ),
            backward=True,
        )
        dense_2 = _set_tensor_meta(
            graph.call_function(torch.ops.aten.relu.default, args=(ag_2_wait,)),
            layer_id=2,
            backward=True,
        )
        early_rs_2 = _set_tensor_meta(
            graph.call_function(
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                args=(dense_2, "sum", 0, "fsdp"),
            ),
            layer_id=2,
            backward=True,
        )
        early_rs_2.meta[_FSDP_BUCKET_META] = {
            "plan_fqns": ("layers.2",),
            "direction": "bwd",
        }
        early_rs_2_wait = _set_tensor_meta(
            graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(early_rs_2,),
            ),
            backward=True,
        )
        early_rs_2_user = _set_tensor_meta(
            graph.call_function(
                torch.ops.aten.alias.default,
                args=(early_rs_2_wait,),
            ),
            backward=True,
        )

        barrier_1 = graph.call_function(
            _optimization_barrier,
            args=(shard_1, dense_2),
        )
        barrier_1.meta["val"] = (shard_1.meta["val"], dense_2.meta["val"])
        barrier_1.meta[_TPU_REMAT_BARRIER_KIND_META] = "fsdp_all_gather"
        barrier_1.meta[_TPU_REMAT_REGION_ID_META] = 1
        shard_1_ready = _set_tensor_meta(
            graph.call_function(operator.getitem, args=(barrier_1, 0)),
            backward=True,
        )
        _set_tensor_meta(
            graph.call_function(operator.getitem, args=(barrier_1, 1)),
            backward=True,
        )

        rs_2 = _set_tensor_meta(
            graph.call_function(
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                args=(dense_2, "sum", 0, "fsdp"),
            ),
            layer_id=2,
            backward=True,
        )
        rs_2.meta[_FSDP_BUCKET_META] = {
            "plan_fqns": ("layers.2",),
            "direction": "bwd",
        }
        ag_1 = _set_tensor_meta(
            graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(shard_1_ready, 1, "fsdp"),
            ),
            layer_id=1,
            backward=True,
        )
        ag_1.meta[_FSDP_BUCKET_META] = {
            "plan_fqns": ("layers.1",),
            "direction": "bwd",
        }
        ag_1_wait = _set_tensor_meta(
            graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(ag_1,),
            ),
            backward=True,
        )
        dense_1 = _set_tensor_meta(
            graph.call_function(torch.ops.aten.relu.default, args=(ag_1_wait,)),
            layer_id=1,
            backward=True,
        )
        rs_2_wait = _set_tensor_meta(
            graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(rs_2,),
            ),
            backward=True,
        )
        graph.output((dense_1, early_rs_2_user, rs_2_wait))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        barrier_count = sum(
            node.target is _optimization_barrier for node in gm.graph.nodes
        )
        apply_tpu_fsdp_rs_deadline_schedule(gm, (), n_layers=3, strict=True)

        self.assertEqual(
            sum(node.target is _optimization_barrier for node in gm.graph.nodes),
            barrier_count,
        )
        self.assertIn(early_rs_2_wait, barrier_1.args)
        self.assertIn(rs_2_wait, barrier_1.args)
        self.assertEqual(
            barrier_1.meta[_TPU_FSDP_RS_DEADLINE_META],
            (early_rs_2_wait.name, rs_2_wait.name),
        )
        self.assertNotIn(_TPU_FSDP_RS_DEADLINE_META, barrier_2.meta)
        order = {node: index for index, node in enumerate(gm.graph.nodes)}
        self.assertLess(order[early_rs_2_wait], order[early_rs_2_user])
        self.assertLess(order[early_rs_2_user], order[barrier_1])
        self.assertLess(order[rs_2], order[rs_2_wait])
        self.assertLess(order[rs_2_wait], order[barrier_1])
        self.assertLess(order[barrier_1], order[ag_1])


class _Projection(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.to(torch.bfloat16)


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.moe = nn.Module()
        self.moe.experts = _Projection(dim)
        self.attention = _Projection(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.moe.experts(x)) + self.attention(x)


class _Epilogue(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.square()


class _ToyModel(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(dim), _Block(dim)])
        self.norm = _Epilogue()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def _nearest_parameter_fqns(node: torch.fx.Node) -> tuple[str, ...]:
    frontier = {node}
    seen: set[torch.fx.Node] = set()
    while frontier:
        fqns = {
            fqn
            for candidate in frontier
            for fqn in candidate.meta.get("custom", {}).get(
                PARAMETER_GRADIENT_FQNS_META, ()
            )
        }
        if fqns:
            return tuple(sorted(fqns))
        seen.update(frontier)
        frontier = {
            user
            for candidate in frontier
            for user in candidate.users
            if user not in seen
        }
    return ()


def _describe_barriers(gm: torch.fx.GraphModule) -> dict[int, dict[str, tuple]]:
    description = {}
    for barrier in gm.graph.nodes:
        if barrier.op != "call_function" or barrier.target is not _optimization_barrier:
            continue

        redirected_edges = []
        ordering_edges = []
        redirected_regions = set()
        outputs = {
            user.args[1]: user
            for user in barrier.users
            if user.op == "call_function" and user.target is operator.getitem
        }
        for index, value in enumerate(barrier.args):
            destinations = {
                (
                    _get_layer_id(user),
                    (
                        "recompute"
                        if user.name.endswith("_recomputed")
                        else "backward"
                        if _is_backward_node(user)
                        else "forward"
                    ),
                )
                for user in outputs[index].users
            }
            dtype = str(value.meta["val"].dtype)
            if destinations:
                redirected_regions.update(region_id for region_id, _ in destinations)
                redirected_edges.append((tuple(sorted(destinations)), dtype))
            else:
                ordering_edges.append((_nearest_parameter_fqns(value), dtype))

        assert len(redirected_regions) == 1
        description[redirected_regions.pop()] = {
            "redirected": tuple(sorted(redirected_edges)),
            "ordering": tuple(sorted(ordering_edges)),
        }
    return description


class TestTpuRematSchedule(unittest.TestCase):
    def test_traced_two_layer_schedule(self) -> None:
        for policy_name, policy_fn in (
            ("full", _make_full_memory_policy()),
            ("sac", _make_default_memory_policy(set())),
        ):
            with self.subTest(policy=policy_name):
                torch.manual_seed(0)
                model = _ToyModel()
                annotate_module_fqns(model)

                def step(x: torch.Tensor) -> list[torch.Tensor]:
                    loss = model(x).float().sum()
                    named_params = [
                        (name, parameter)
                        for name, parameter in model.named_parameters(
                            remove_duplicate=False
                        )
                        if parameter.requires_grad
                    ]
                    grads = compute_parameter_gradients(loss, named_params)
                    updated_params = [
                        parameter - 0.01 * grad
                        for (_, parameter), grad in zip(
                            named_params, grads, strict=True
                        )
                    ]
                    return [loss, *updated_params]

                x = torch.randn(2, 4, dtype=torch.bfloat16)
                traced = minimal_fx_tracer(step, module=model)(x)
                expected_outputs = [
                    output.detach().clone()
                    for output in run_traced(traced, module=model)(x)
                ]
                remove_parameter_gradient_markers_pass(
                    traced.gm, traced.example_inputs
                )
                tag_sac_policy(traced.gm, policy_fn=policy_fn)
                selective_activation_remat_pass(traced.gm)
                apply_tpu_remat_schedule(
                    traced.gm,
                    traced.example_inputs,
                    num_static_inputs=traced.num_static_inputs,
                )

                # This single topology comparison proves that every layer entry
                # redirects its saved activation and incoming cotangent through
                # one barrier; layer 0 additionally waits for both layer 1 local
                # dWeights even though an optimizer-like update consumes the
                # tagged gradients instead of returning them. The non-expert
                # path preserves the generic FSDP ordering point at the FP32
                # collective input. The validated expert specialization instead
                # orders on BF16 local dWeight, leaving its cast and collective
                # outside the barrier and available to overlap. Generalizing
                # that ordering point to all eligible gradients is a follow-up.
                self.assertEqual(
                    _describe_barriers(traced.gm),
                    {
                        0: {
                            "redirected": (
                                (((0, "backward"),), "torch.bfloat16"),
                                (((0, "recompute"),), "torch.bfloat16"),
                            ),
                            "ordering": (
                                (("layers.1.attention.weight",), "torch.float32"),
                                (("layers.1.moe.experts.weight",), "torch.bfloat16"),
                            ),
                        },
                        1: {
                            "redirected": (
                                (((1, "backward"),), "torch.bfloat16"),
                                (((1, "recompute"),), "torch.bfloat16"),
                            ),
                            "ordering": (),
                        },
                    },
                )

                actual_outputs = run_traced(traced, module=model)(x)
                for actual, expected in zip(
                    actual_outputs, expected_outputs, strict=True
                ):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
