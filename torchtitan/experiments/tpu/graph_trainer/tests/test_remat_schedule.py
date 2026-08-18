# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    find_fsdp_local_grad_before_reduce_dtype_cast,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    REMAT_REGION_ID_META,
    selective_activation_remat_pass,
)
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    _optimization_barrier,
    apply_tpu_remat_schedule,
)


class TestTpuRematSchedule(unittest.TestCase):
    def _make_graph(
        self,
        *,
        layer_ids: tuple[int, int] = (0, 1),
        derived_cotangent: bool = False,
        multiple_cotangents: bool = False,
        fsdp_sync: bool = False,
    ) -> torch.fx.GraphModule:
        graph = torch.fx.Graph()
        expert_w0 = graph.placeholder("expert_w0")
        attention_w0 = graph.placeholder("attention_w0")
        expert_w1 = graph.placeholder("expert_w1")
        attention_w1 = graph.placeholder("attention_w1")
        x0 = graph.placeholder("x0")
        x1 = graph.placeholder("x1")
        dy1 = graph.placeholder("dy1")

        remat0 = graph.call_function(torch.ops.aten.mm.default, args=(x0, expert_w0))
        remat1 = graph.call_function(torch.ops.aten.mm.default, args=(x1, expert_w1))
        incoming1 = graph.call_function(torch.ops.aten.clone.default, args=(dy1,))
        if derived_cotangent:
            incoming1b = graph.call_function(
                torch.ops.aten.add.Tensor, args=(incoming1, incoming1)
            )
        elif multiple_cotangents:
            incoming1b = graph.call_function(
                torch.ops.aten.clone.default, args=(dy1,)
            )
        else:
            incoming1b = incoming1
        expert_dw1 = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(remat1, incoming1)
        )
        attention_dw1 = graph.call_function(
            torch.ops.aten.mul.Tensor,
            args=(
                (
                    remat1
                    if multiple_cotangents or derived_cotangent
                    else incoming1b
                ),
                (
                    incoming1b
                    if multiple_cotangents or derived_cotangent
                    else attention_w1
                ),
            ),
        )

        def add_sync(gradient: torch.fx.Node) -> tuple[torch.fx.Node, ...]:
            if not fsdp_sync:
                return (
                    graph.call_function(
                        torch.ops.aten.clone.default, args=(gradient,)
                    ),
                )
            cast = graph.call_function(
                torch.ops.aten._to_copy.default,
                args=(gradient,),
                kwargs={"dtype": torch.float32},
            )
            collective = graph.call_function(
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                args=(cast, "sum", 0, "0"),
            )
            wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(collective,),
            )
            return cast, collective, wait

        expert_sync1_nodes = add_sync(expert_dw1)
        attention_sync1_nodes = add_sync(attention_dw1)
        expert_sync1 = expert_sync1_nodes[-1]
        attention_sync1 = attention_sync1_nodes[-1]
        dinput1 = graph.call_function(torch.ops.aten.clone.default, args=(incoming1,))
        expert_dw0 = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(remat0, dinput1)
        )
        attention_dw0 = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(dinput1, attention_w0)
        )
        expert_sync0_nodes = add_sync(expert_dw0)
        attention_sync0_nodes = add_sync(attention_dw0)
        expert_sync0 = expert_sync0_nodes[-1]
        attention_sync0 = attention_sync0_nodes[-1]
        dinput0 = graph.call_function(torch.ops.aten.clone.default, args=(dinput1,))
        graph.output(
            (expert_sync0, attention_sync0, expert_sync1, attention_sync1, dinput0)
        )

        tensor_nodes = [
            expert_w0,
            attention_w0,
            expert_w1,
            attention_w1,
            x0,
            x1,
            dy1,
            remat0,
            remat1,
            incoming1,
            expert_dw1,
            attention_dw1,
            dinput1,
            expert_dw0,
            attention_dw0,
            dinput0,
            expert_sync0,
            attention_sync0,
            expert_sync1,
            attention_sync1,
        ]
        if multiple_cotangents or derived_cotangent:
            tensor_nodes.append(incoming1b)
        if fsdp_sync:
            tensor_nodes.extend(
                expert_sync0_nodes[:-1]
                + attention_sync0_nodes[:-1]
                + expert_sync1_nodes[:-1]
                + attention_sync1_nodes[:-1]
            )
        for node in tensor_nodes:
            node.meta["val"] = torch.randn(4, 4)
        if fsdp_sync:
            expert_dw0.meta["val"] = torch.randn(4, 4, dtype=torch.bfloat16)
            expert_dw1.meta["val"] = torch.randn(4, 4, dtype=torch.bfloat16)
            expert_sync0_nodes[0].meta["val"] = torch.randn(4, 4)
            expert_sync1_nodes[0].meta["val"] = torch.randn(4, 4)

        for layer_id, node in zip(layer_ids, (remat0, remat1)):
            node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
            node.meta["custom"] = {
                "module_fqn": f"layers.{layer_id}.moe.experts"
            }

        incoming1.meta["autograd_backward"] = True
        incoming1.meta["custom"] = {"module_fqn": "loss"}
        if multiple_cotangents or derived_cotangent:
            incoming1b.meta["autograd_backward"] = True
            incoming1b.meta["custom"] = {"module_fqn": "loss"}
        for layer_id, nodes in (
            (layer_ids[1], (expert_dw1, attention_dw1, dinput1)),
            (layer_ids[0], (expert_dw0, attention_dw0)),
        ):
            components = ("moe.experts", "attention", "moe")
            for node, component in zip(nodes, components):
                node.meta["autograd_backward"] = True
                node.meta["custom"] = {
                    "module_fqn": f"layers.{layer_id}.{component}"
                }
        dinput0.meta["autograd_backward"] = True
        dinput0.meta["custom"] = {"module_fqn": ""}

        for layer_id, sync_groups in (
            (layer_ids[0], (expert_sync0_nodes, attention_sync0_nodes)),
            (layer_ids[1], (expert_sync1_nodes, attention_sync1_nodes)),
        ):
            for nodes, component in zip(sync_groups, ("moe.experts", "attention")):
                for node in nodes:
                    node.meta["autograd_backward"] = True
                    node.meta["custom"] = {
                        "module_fqn": f"layers.{layer_id}.{component}"
                    }

        return torch.fx.GraphModule(torch.nn.Module(), graph)

    def test_orders_all_next_layer_parameter_gradients(self):
        torch.manual_seed(0)
        inputs = tuple(torch.randn(4, 4) for _ in range(7))
        reference = self._make_graph()(*inputs)
        gm = self._make_graph()

        selective_activation_remat_pass(gm)
        apply_tpu_remat_schedule(
            gm,
            (),
            num_static_inputs=4,
        )

        actual = gm(*inputs)
        for result, expected in zip(actual, reference):
            torch.testing.assert_close(result, expected)

        barriers = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and "_optimization_barrier" in str(node.target)
        ]
        self.assertEqual(len(barriers), 2)
        barriers_by_region = {
            node.meta[REMAT_REGION_ID_META]: node for node in barriers
        }
        self.assertEqual(len(barriers_by_region[0].args), 4)
        self.assertEqual(len(barriers_by_region[1].args), 2)

        region0_fqns = {
            node.meta.get("custom", {}).get("module_fqn")
            for node in barriers_by_region[0].args
        }
        self.assertIn("layers.1.moe.experts", region0_fqns)
        self.assertIn("layers.1.attention", region0_fqns)
        gm.graph.lint()

    def test_orders_local_parameter_gradients(self):
        gm = self._make_graph(fsdp_sync=True)
        selective_activation_remat_pass(gm)
        apply_tpu_remat_schedule(
            gm,
            (),
            num_static_inputs=4,
        )

        barrier = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and "_optimization_barrier" in str(node.target)
            and node.meta[REMAT_REGION_ID_META] == 0
        )
        casted_local_gradients = {
            node
            for node in barrier.args
            if node.op == "call_function"
            and node.target == torch.ops.aten._to_copy.default
        }
        raw_expert_gradients = {
            node
            for node in barrier.args
            if node.meta.get("custom", {}).get("module_fqn", "").endswith(
                ".moe.experts"
            )
        }
        collectives = {
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target
            == torch.ops._c10d_functional.reduce_scatter_tensor.default
            and node.meta.get("custom", {}).get("module_fqn", "").startswith(
                "layers.1."
            )
        }
        expert_collective = next(
            node
            for node in collectives
            if node.meta.get("custom", {}).get("module_fqn")
            == "layers.1.moe.experts"
        )
        attention_collective = next(
            node
            for node in collectives
            if node.meta.get("custom", {}).get("module_fqn")
            == "layers.1.attention"
        )
        raw_expert_gradient = next(iter(raw_expert_gradients))
        expert_cast = next(
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.aten._to_copy.default
            and node.meta.get("custom", {}).get("module_fqn")
            == "layers.1.moe.experts"
        )
        attention_cast = next(iter(casted_local_gradients))
        self.assertEqual(len(casted_local_gradients), 1)
        self.assertEqual(len(raw_expert_gradients), 1)
        self.assertEqual(len(collectives), 2)
        self.assertIn(raw_expert_gradient, barrier.args)
        self.assertNotIn(expert_cast, barrier.args)
        self.assertIs(expert_collective.args[0], expert_cast)
        self.assertIs(attention_collective.args[0], attention_cast)
        self.assertEqual(expert_collective.args[0].meta["val"].dtype, torch.float32)
        gm.graph.lint()

    def test_finds_local_gradient_through_post_cast_view(self):
        graph = torch.fx.Graph()
        local_gradient = graph.placeholder("local_gradient")
        cast = graph.call_function(
            torch.ops.aten._to_copy.default,
            args=(local_gradient,),
            kwargs={"dtype": torch.float32},
        )
        view = graph.call_function(torch.ops.aten.view.default, args=(cast, (4, 4)))
        local_gradient.meta["val"] = torch.randn(4, 4, dtype=torch.bfloat16)
        cast.meta["val"] = torch.randn(4, 4)
        view.meta["val"] = torch.randn(4, 4)

        self.assertIs(
            find_fsdp_local_grad_before_reduce_dtype_cast(view), local_gradient
        )

    def test_supports_noncontiguous_remat_regions(self):
        gm = self._make_graph(layer_ids=(0, 2))
        selective_activation_remat_pass(gm)
        apply_tpu_remat_schedule(
            gm,
            (),
            num_static_inputs=4,
        )

        barrier_regions = {
            node.meta[REMAT_REGION_ID_META]
            for node in gm.graph.nodes
            if node.op == "call_function"
            and "_optimization_barrier" in str(node.target)
        }
        self.assertEqual(barrier_regions, {0, 2})

    def test_anchors_at_earliest_root_cotangent(self):
        gm = self._make_graph(multiple_cotangents=True)
        selective_activation_remat_pass(gm)
        apply_tpu_remat_schedule(
            gm,
            (),
            num_static_inputs=4,
        )

        barrier = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and "_optimization_barrier" in str(node.target)
            and node.meta[REMAT_REGION_ID_META] == 1
        )
        cotangents = [
            node
            for node in barrier.args
            if node.meta.get("custom", {}).get("module_fqn") == "loss"
        ]
        self.assertEqual(len(cotangents), 1)
        self.assertEqual(cotangents[0].name, "clone_default")

    def test_accepts_derived_cotangents(self):
        gm = self._make_graph(derived_cotangent=True)
        selective_activation_remat_pass(gm)
        apply_tpu_remat_schedule(
            gm,
            (),
            num_static_inputs=4,
        )

        barrier = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and "_optimization_barrier" in str(node.target)
            and node.meta[REMAT_REGION_ID_META] == 1
        )
        self.assertEqual(len(barrier.args), 2)

    def test_optimization_barrier_rejects_mixed_devices(self):
        with self.assertRaisesRegex(ValueError, "all-CPU test inputs or all-TPU"):
            _optimization_barrier(torch.ones(1), torch.ones(1, device="meta"))


if __name__ == "__main__":
    unittest.main()
