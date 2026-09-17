# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import cast
from unittest import mock

from absl.testing import absltest
import torch
from torch import nn

from torchtitan.config import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.tpu.graph_trainer import qwen3 as graph_trainer_qwen3
from torchtitan.experiments.tpu.qwen3.infra.parallelize import (
    configure_qwen_simple_fsdp_expert_shard_dims,
)


def _model() -> nn.Module:
  model = nn.Module()
  layer = nn.Module()
  layer.moe = nn.Module()
  layer.moe.experts = nn.Module()
  layer.moe.experts.weight = nn.Parameter(torch.empty(2, 8, device="meta"))
  model.layers = nn.ModuleDict({"0": layer})
  return model


def _wrapped_model() -> nn.Module:
  model = _model()
  wrapper = nn.Module()
  wrapper.inner = model.layers["0"]
  model.layers["0"] = wrapper
  return model


def _parallel_dims(fsdp_degree: int) -> ParallelDims:
  return ParallelDims(
      dp_replicate=1,
      dp_shard=fsdp_degree,
      cp=1,
      tp=1,
      pp=1,
      ep=1,
      world_size=fsdp_degree,
  )


class SimpleFSDPShardingTest(absltest.TestCase):

  def test_uses_hidden_axis_when_leading_axis_is_uneven(self) -> None:
    model = _model()

    configure_qwen_simple_fsdp_expert_shard_dims(model, _parallel_dims(4))

    self.assertEqual(model.layers["0"].moe.experts._fsdp_shard_dim, 1)

  def test_keeps_default_axis_when_leading_axis_is_even(self) -> None:
    model = _model()

    configure_qwen_simple_fsdp_expert_shard_dims(model, _parallel_dims(2))

    self.assertFalse(hasattr(model.layers["0"].moe.experts, "_fsdp_shard_dim"))

  def test_finds_experts_beneath_layer_wrappers(self) -> None:
    model = _wrapped_model()

    configure_qwen_simple_fsdp_expert_shard_dims(model, _parallel_dims(4))

    self.assertEqual(
        model.layers["0"].inner.moe.experts._fsdp_shard_dim,
        1,
    )

  def test_rejects_when_neither_axis_is_divisible(self) -> None:
    model = _model()
    model.layers["0"].moe.experts.weight = nn.Parameter(
        torch.empty(2, 6, device="meta")
    )

    with self.assertRaisesRegex(ValueError, "moe.experts.*dimension 0 or 1"):
      configure_qwen_simple_fsdp_expert_shard_dims(model, _parallel_dims(4))

  def test_graph_trainer_configures_experts_before_simple_fsdp(self) -> None:
    model = _model()
    parallel_dims = _parallel_dims(4)
    training = cast(
        TrainingConfig,
        SimpleNamespace(mixed_precision_reduce="bfloat16"),
    )

    with (
        mock.patch.object(
            graph_trainer_qwen3,
            "configure_expert_dweight_output_dtype",
        ),
        mock.patch.object(
            graph_trainer_qwen3,
            "graph_trainer_parallelize_qwen3",
            return_value=model,
        ) as upstream_parallelize,
    ):
      result = graph_trainer_qwen3.parallelize_qwen3(
          cast(GraphTrainerQwen3Model, model),
          training=training,
          parallel_dims=parallel_dims,
      )

    self.assertIs(result, model)
    self.assertEqual(model.layers["0"].moe.experts._fsdp_shard_dim, 1)
    upstream_parallelize.assert_called_once_with(
        model,
        training=training,
        parallel_dims=parallel_dims,
    )


if __name__ == "__main__":
  absltest.main()
