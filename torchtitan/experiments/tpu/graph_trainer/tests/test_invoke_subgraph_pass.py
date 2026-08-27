# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.fsdp_passes import (
    schedule_fsdp_comms_to_dense_regions_pass,
)
from torchtitan.experiments.tpu.graph_trainer.passes.invoke_subgraph import (
    outline_transformer_layers_pass,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
    construct_tpu_pass_pipeline,
)
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig


def _set_tensor_meta(node: fx.Node) -> fx.Node:
    node.meta["val"] = torch.empty(4, 4)
    return node


def _set_layer_meta(node: fx.Node, layer: int, *, backward: bool) -> fx.Node:
    _set_tensor_meta(node)
    node.meta["custom"] = {"module_fqn": f"layers.{layer}.linear"}
    if backward:
        node.meta["autograd_backward"] = True
    return node


def _make_overlap_graph() -> fx.GraphModule:
    graph = fx.Graph()
    x = _set_tensor_meta(graph.placeholder("x"))
    weight0 = _set_tensor_meta(graph.placeholder("weight0"))
    weight1 = _set_tensor_meta(graph.placeholder("weight1"))
    grad = _set_tensor_meta(graph.placeholder("grad"))

    ag0 = _set_tensor_meta(
        graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            args=(weight0, 1, "0"),
        )
    )
    wait0 = _set_tensor_meta(
        graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(ag0,),
        )
    )
    forward0 = _set_layer_meta(
        graph.call_function(torch.ops.aten.mm.default, args=(x, wait0)),
        0,
        backward=False,
    )
    ag1 = _set_tensor_meta(
        graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            args=(weight1, 1, "0"),
        )
    )
    wait1 = _set_tensor_meta(
        graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(ag1,),
        )
    )
    forward1 = _set_layer_meta(
        graph.call_function(torch.ops.aten.mm.default, args=(forward0, wait1)),
        1,
        backward=False,
    )
    bwd_ag1 = _set_tensor_meta(
        graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            args=(weight1, 1, "0"),
        )
    )
    bwd_wait1 = _set_tensor_meta(
        graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(bwd_ag1,),
        )
    )
    backward1 = _set_layer_meta(
        graph.call_function(torch.ops.aten.mm.default, args=(grad, bwd_wait1)),
        1,
        backward=True,
    )
    rs1 = _set_tensor_meta(
        graph.call_function(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            args=(backward1, "sum", 1, "0"),
        )
    )
    wait_rs1 = _set_tensor_meta(
        graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(rs1,),
        )
    )
    bwd_ag0 = _set_tensor_meta(
        graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            args=(weight0, 1, "0"),
        )
    )
    bwd_wait0 = _set_tensor_meta(
        graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(bwd_ag0,),
        )
    )
    backward0 = _set_layer_meta(
        graph.call_function(torch.ops.aten.mm.default, args=(backward1, bwd_wait0)),
        0,
        backward=True,
    )
    rs0 = _set_tensor_meta(
        graph.call_function(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            args=(backward0, "sum", 1, "0"),
        )
    )
    wait_rs0 = _set_tensor_meta(
        graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(rs0,),
        )
    )
    graph.output((forward1, wait_rs1, wait_rs0))
    return fx.GraphModule(torch.nn.Module(), graph)


class InvokeSubgraphPassTest(unittest.TestCase):

    def test_prototype_flags_default_off(self) -> None:
        config = TPUConfig()
        self.assertTrue(config.graph_trainer_enable_sac)
        self.assertFalse(config.graph_trainer_fsdp_overlap)
        self.assertFalse(config.graph_trainer_invoke_subgraph)

    def test_pipeline_orders_sac_and_fsdp_before_outlining(self) -> None:
        config = types.SimpleNamespace(
            compile=types.SimpleNamespace(memory_policy="full"),
            tpu_config=types.SimpleNamespace(
                graph_trainer_enable_sac=True,
                graph_trainer_fsdp_overlap=True,
                graph_trainer_invoke_subgraph=True,
            ),
            model_spec=types.SimpleNamespace(
                model=types.SimpleNamespace(
                    layers=[types.SimpleNamespace(moe=None) for _ in range(2)]
                )
            ),
        )
        passes = construct_tpu_pass_pipeline(
            types.SimpleNamespace(state_fqns=[]), config
        )
        names = [getattr(pass_fn, "func", pass_fn).__name__ for pass_fn in passes]

        self.assertLess(
            names.index("selective_activation_remat_pass"),
            names.index("schedule_fsdp_comms_to_dense_regions_pass"),
        )
        self.assertLess(
            names.index("schedule_fsdp_comms_to_dense_regions_pass"),
            names.index("outline_transformer_layers_pass"),
        )
        self.assertLess(
            names.index("outline_transformer_layers_pass"),
            names.index("tpu_backend_compile_pass"),
        )

    def test_pipeline_can_isolate_fsdp_overlap_without_sac(self) -> None:
        config = types.SimpleNamespace(
            compile=types.SimpleNamespace(memory_policy="default"),
            tpu_config=types.SimpleNamespace(
                graph_trainer_enable_sac=False,
                graph_trainer_fsdp_overlap=True,
                graph_trainer_invoke_subgraph=True,
            ),
            model_spec=types.SimpleNamespace(
                model=types.SimpleNamespace(
                    layers=[types.SimpleNamespace(moe=None) for _ in range(2)]
                )
            ),
        )
        passes = construct_tpu_pass_pipeline(
            types.SimpleNamespace(state_fqns=[]), config
        )
        names = [getattr(pass_fn, "func", pass_fn).__name__ for pass_fn in passes]

        self.assertNotIn("selective_activation_remat_pass", names)
        self.assertLess(
            names.index("schedule_fsdp_comms_to_dense_regions_pass"),
            names.index("outline_transformer_layers_pass"),
        )

    def test_collectives_stay_outside_repeated_layer_calls(self) -> None:
        gm = schedule_fsdp_comms_to_dense_regions_pass(
            _make_overlap_graph(),
            moe_layer_ids=frozenset(),
            n_layers=2,
            strict=False,
        )
        gm = outline_transformer_layers_pass(gm, min_region_nodes=1)
        nodes = list(gm.graph.nodes)
        invoke_nodes = [
            node
            for node in nodes
            if node.op == "call_function"
            and node.target is torch.ops.higher_order.invoke_subgraph
        ]
        self.assertEqual(len(invoke_nodes), 4)
        self.assertEqual(len({node.args[1] for node in invoke_nodes}), 2)

        outer_targets = {node.target for node in nodes if node.op == "call_function"}
        self.assertNotIn(torch.ops.aten.mm.default, outer_targets)
        self.assertIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            outer_targets,
        )
        self.assertIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            outer_targets,
        )

        ag1 = next(
            node
            for node in nodes
            if node.target is torch.ops._c10d_functional.all_gather_into_tensor.default
            and node.args[0].target == "weight1"
        )
        wait1 = next(
            node
            for node in nodes
            if node.target is torch.ops._c10d_functional.wait_tensor.default
            and node.args[0] is ag1
        )
        self.assertLess(nodes.index(ag1), nodes.index(invoke_nodes[0]))
        self.assertLess(nodes.index(invoke_nodes[0]), nodes.index(wait1))

        for invoke in invoke_nodes:
            subgraph = gm.get_submodule(invoke.args[0].target)
            subgraph_targets = {
                node.target
                for node in subgraph.graph.nodes
                if node.op == "call_function"
            }
            self.assertEqual(subgraph_targets, {torch.ops.aten.mm.default})


if __name__ == "__main__":
    unittest.main()
