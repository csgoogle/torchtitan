# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the TPU selective memory policies on traced toy graphs."""

from types import SimpleNamespace
import unittest

from absl.testing import absltest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    compute_parameter_gradients,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.graph_trainer.memory_policy import (
    _find_fsdp_unshard_save_nodes,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_memory_policy import (
    full_remat_with_saves,
)


def _parallelism(reshard_after_forward: str) -> SimpleNamespace:
    return SimpleNamespace(
        fsdp_reshard_after_forward=reshard_after_forward,
        pipeline_parallel_degree=1,
    )


class _DenseBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.weight.to(x.dtype))


class _DenseToy(nn.Module):
    def __init__(self, dim: int = 8, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_DenseBlock(dim) for _ in range(n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _trace_simple_fsdp_graph() -> torch.fx.GraphModule:
    """Trace a two-rank SimpleFSDP toy on a fake process group."""
    assert not dist.is_initialized()
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
    try:
        mesh = DeviceMesh("cpu", [0, 1])
        with torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True):
            model = _DenseToy()
            annotate_module_fqns(model)
            model = data_parallel(
                model,
                device_mesh=mesh,
                mode="fully_shard",
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
            )
            inputs = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)

            def step(inputs: torch.Tensor) -> list[torch.Tensor]:
                loss = model(inputs).float().sum()
                named_params = list(model.named_parameters())
                return [loss, *compute_parameter_gradients(loss, named_params)]

            traced = minimal_fx_tracer(step, module=model)(inputs)
    finally:
        dist.destroy_process_group()
    return traced.gm


class TestFullRematWithSaves(unittest.TestCase):
    def test_zero2_keeps_all_gather_outputs_saved(self) -> None:
        for reshard_after_forward, expected in (
            ("never", CheckpointPolicy.MUST_SAVE),
            ("always", CheckpointPolicy.MUST_RECOMPUTE),
        ):
            with self.subTest(reshard_after_forward=reshard_after_forward):
                gm = _trace_simple_fsdp_graph()
                unshards = _find_fsdp_unshard_save_nodes(gm)
                self.assertTrue(unshards)

                # No policy saves of its own: only the FSDP contract applies.
                full_remat_with_saves(
                    gm,
                    SimpleNamespace(parallelism=_parallelism(reshard_after_forward)),
                    save_nodes=(),
                )

                self.assertEqual(
                    {node.meta["recompute"] for node in unshards}, {expected}
                )


if __name__ == "__main__":
    absltest.main()
