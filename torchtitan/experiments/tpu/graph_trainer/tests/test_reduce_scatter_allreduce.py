# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from unittest import mock

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.experiments.graph_trainer.common_utils import (
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.tpu.graph_trainer.passes.reduce_scatter_allreduce import (
    apply_selected_reduce_scatter_allreduce_pass,
)


_QKV_PATTERN = "layers.*.attention.qkv_linear.wqkv_packed.weight"
_O_PATTERN = "layers.*.attention.wo.weight"
_MAX_INPUT_BYTES = 40 * 1024**2


class TestReduceScatterAllReduce(unittest.TestCase):
    def _make_graph(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype | None = None,
        group_size: int = 128,
        parameter_fqn: str = "layers.0.attention.qkv_linear.wqkv_packed.weight",
    ) -> torch.fx.GraphModule:
        graph = torch.fx.Graph()
        inp = graph.placeholder("inp")
        rs = graph.call_function(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            args=(inp, "sum", group_size, "world"),
        )
        wait = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(rs,),
        )
        graph.output(wait)
        gm = torch.fx.GraphModule({}, graph)
        output_shape = (shape[0] // group_size, *shape[1:])
        output_dtype = output_dtype or dtype
        with FakeTensorMode():
            inp.meta["val"] = torch.empty(shape, dtype=dtype)
            rs.meta["val"] = torch.empty(output_shape, dtype=output_dtype)
            wait.meta["val"] = rs.meta["val"]
        wait.meta["custom"] = {
            PARAMETER_GRADIENT_FQNS_META: (parameter_fqn,)
        }
        return gm

    @mock.patch("torch.distributed.get_rank", return_value=3)
    @mock.patch("torch.distributed.get_world_size", return_value=128)
    @mock.patch(
        "torch.distributed.distributed_c10d._resolve_process_group",
        return_value=mock.sentinel.process_group,
    )
    def test_rewrites_qkv_reduce_scatter(
        self,
        resolve_process_group,
        get_world_size,
        get_rank,
    ):
        gm = self._make_graph((5120, 2048))

        apply_selected_reduce_scatter_allreduce_pass(
            gm,
            (),
            parameter_patterns=(_QKV_PATTERN,),
            max_input_bytes=_MAX_INPUT_BYTES,
        )

        targets = [
            node.target for node in gm.graph.nodes if node.op == "call_function"
        ]
        self.assertNotIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default, targets
        )
        self.assertEqual(
            targets.count(torch.ops._c10d_functional.all_reduce.default), 1
        )
        output = next(node for node in gm.graph.nodes if node.op == "output")
        local_shard = output.args[0]
        self.assertEqual(local_shard.target, torch.ops.aten.slice.Tensor)
        self.assertEqual(local_shard.args[2:4], (120, 160))
        self.assertEqual(tuple(local_shard.meta["val"].shape), (40, 2048))
        resolve_process_group.assert_called_once_with("world")
        get_world_size.assert_called_once_with()
        get_rank.assert_called_once_with(mock.sentinel.process_group)

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torch.distributed.get_world_size", return_value=128)
    @mock.patch(
        "torch.distributed.distributed_c10d._resolve_process_group",
        return_value=mock.sentinel.process_group,
    )
    def test_rewrites_output_projection_reduce_scatter(self, *_):
        gm = self._make_graph(
            (2048, 4096), parameter_fqn="layers.0.attention.wo.weight"
        )

        apply_selected_reduce_scatter_allreduce_pass(
            gm,
            (),
            parameter_patterns=(_O_PATTERN,),
            max_input_bytes=_MAX_INPUT_BYTES,
        )

        targets = [node.target for node in gm.graph.nodes if node.op == "call_function"]
        self.assertEqual(
            targets.count(torch.ops._c10d_functional.all_reduce.default), 1
        )

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torch.distributed.get_world_size", return_value=128)
    def test_missing_pattern_errors_without_mutating_graph(self, *_):
        gm = self._make_graph((5120, 2048))
        before = str(gm.graph)

        with self.assertRaisesRegex(ValueError, "matched no parameter gradient"):
            apply_selected_reduce_scatter_allreduce_pass(
                gm,
                (),
                parameter_patterns=(_O_PATTERN,),
                max_input_bytes=_MAX_INPUT_BYTES,
            )

        self.assertEqual(str(gm.graph), before)

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torch.distributed.get_world_size", return_value=128)
    def test_selected_oversized_gradient_errors_without_mutating_graph(self, *_):
        gm = self._make_graph((5120, 2048))
        before = str(gm.graph)

        with self.assertRaisesRegex(ValueError, "exceed the configured maximum"):
            apply_selected_reduce_scatter_allreduce_pass(
                gm,
                (),
                parameter_patterns=(_QKV_PATTERN,),
                max_input_bytes=32 * 1024**2,
            )

        self.assertEqual(str(gm.graph), before)

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torch.distributed.get_world_size", return_value=128)
    @mock.patch(
        "torch.distributed.distributed_c10d._resolve_process_group",
        return_value=mock.sentinel.process_group,
    )
    def test_rewrites_selected_bf16_gradient(self, *_):
        gm = self._make_graph((5120, 2048), dtype=torch.bfloat16)

        apply_selected_reduce_scatter_allreduce_pass(
            gm,
            (),
            parameter_patterns=(_QKV_PATTERN,),
            max_input_bytes=_MAX_INPUT_BYTES,
        )

        targets = [node.target for node in gm.graph.nodes if node.op == "call_function"]
        self.assertNotIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default, targets
        )
        self.assertEqual(
            targets.count(torch.ops._c10d_functional.all_reduce.default), 1
        )

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torch.distributed.get_world_size", return_value=128)
    def test_selected_mixed_dtype_gradient_errors_without_mutating_graph(self, *_):
        gm = self._make_graph(
            (5120, 2048), output_dtype=torch.bfloat16
        )
        before = str(gm.graph)

        with self.assertRaisesRegex(ValueError, "matching FP32 or BF16"):
            apply_selected_reduce_scatter_allreduce_pass(
                gm,
                (),
                parameter_patterns=(_QKV_PATTERN,),
                max_input_bytes=_MAX_INPUT_BYTES,
            )

        self.assertEqual(str(gm.graph), before)

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torch.distributed.get_world_size", return_value=128)
    def test_selected_subgroup_gradient_errors_without_mutating_graph(self, *_):
        gm = self._make_graph((5120, 2048), group_size=64)
        before = str(gm.graph)

        with self.assertRaisesRegex(ValueError, "is not world size"):
            apply_selected_reduce_scatter_allreduce_pass(
                gm,
                (),
                parameter_patterns=(_QKV_PATTERN,),
                max_input_bytes=_MAX_INPUT_BYTES,
            )

        self.assertEqual(str(gm.graph), before)


if __name__ == "__main__":
    unittest.main()
