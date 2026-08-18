# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for TPU GraphTrainer config conversion."""

import typing
from types import SimpleNamespace

from absl.testing import absltest
from absl.testing import parameterized
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.graph_trainer import qwen3 as qwen3_gt
from torchtitan.experiments.tpu.graph_trainer.config import (
    to_tpu_graph_trainer_config,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
    construct_tpu_pass_pipeline,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.qwen3.config_registry import qwen3_8b
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig


class TpuGraphTrainerConfigTest(parameterized.TestCase):

  @parameterized.parameters("default", "full", "eager")
  def test_cli_memory_policy_is_propagated(self, policy: str) -> None:
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            f"--tpu_config.graph_trainer_memory_policy={policy}",
        ]),
    )

    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertEqual(config.tpu_config.graph_trainer_memory_policy, policy)
    self.assertEqual(graph_config.compile.memory_policy, policy)

  def test_unsupported_memory_policy_is_rejected_during_conversion(self) -> None:
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            "--tpu_config.graph_trainer_memory_policy=sac_and_offload",
        ]),
    )

    self.assertEqual(
        config.tpu_config.graph_trainer_memory_policy,
        "sac_and_offload",
    )
    with self.assertRaisesRegex(ValueError, "does not support"):
      to_tpu_graph_trainer_config(
          config,
          qwen3_gt.model_registry,
          target_cls=TpuGraphTrainer.Config,
      )

  def test_cli_remat_schedule_is_propagated(self) -> None:
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            "--tpu_config.enable_graph_trainer_remat_schedule",
        ]),
    )

    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertTrue(
        graph_config.tpu_config.enable_graph_trainer_remat_schedule
    )

  def test_qwen3_8b_explicitly_uses_default_graph_policy(self) -> None:
    config = qwen3_8b()

    self.assertEqual(
        config.tpu_config.graph_trainer_memory_policy,
        "default",
    )

    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )
    self.assertEqual(graph_config.compile.memory_policy, "default")

  def test_remat_schedule_rejects_optimizer_in_graph(self) -> None:
    config = qwen3_8b()
    config.tpu_config.enable_graph_trainer_remat_schedule = True
    config.tpu_config.optimizer_in_graph = True
    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    with self.assertRaisesRegex(ValueError, "optimizer_in_graph"):
      construct_tpu_pass_pipeline(
          typing.cast(typing.Any, SimpleNamespace(num_static_inputs=0)),
          graph_config,
      )


if __name__ == "__main__":
  absltest.main()
