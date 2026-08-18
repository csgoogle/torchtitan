# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for TPU GraphTrainer config conversion."""

import functools
import typing
from types import SimpleNamespace

from absl.testing import absltest
from absl.testing import parameterized
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.graph_trainer import qwen3 as qwen3_gt
from torchtitan.experiments.tpu.graph_trainer.config import (
    to_tpu_graph_trainer_config,
)
from torchtitan.experiments.tpu.graph_trainer.passes.elide_paired_dweight_casts import (
    apply_elide_paired_dweight_casts_pass,
)
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    apply_tpu_remat_schedule,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
    construct_tpu_pass_pipeline,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.qwen3.config_registry import (
    qwen3_8b,
    qwen3_moe_30b_fsdp_only_full_ac_profile,
)
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

    pipeline = construct_tpu_pass_pipeline(
        typing.cast(typing.Any, SimpleNamespace(num_static_inputs=0)),
        graph_config,
    )
    pass_targets = [
        item.func if isinstance(item, functools.partial) else item
        for item in pipeline
    ]
    self.assertNotIn(apply_elide_paired_dweight_casts_pass, pass_targets)

  def test_qwen3_moe_fsdp_enables_paired_dweight_elision(self) -> None:
    config = qwen3_moe_30b_fsdp_only_full_ac_profile()
    assert config.model_spec is not None

    self.assertTrue(
        config.tpu_config.elide_bf16_rounded_paired_dweight_casts
    )
    grouped_experts = list(
        config.model_spec.model.traverse(TpuGroupedExperts.Config)
    )
    self.assertNotEmpty(grouped_experts)
    self.assertTrue(
        all(
            experts.emit_bf16_rounded_fp32_paired_dweight
            and experts.use_pallas_paired_grouped_mm
            for _, experts, _, _ in grouped_experts
        )
    )

    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )
    pipeline = construct_tpu_pass_pipeline(
        typing.cast(typing.Any, SimpleNamespace(num_static_inputs=0)),
        graph_config,
    )
    pass_targets = [
        item.func if isinstance(item, functools.partial) else item
        for item in pipeline
    ]
    self.assertLess(
        pass_targets.index(apply_elide_paired_dweight_casts_pass),
        pass_targets.index(apply_tpu_remat_schedule),
    )

  def test_paired_dweight_elision_rejects_unsupported_parallelism(self) -> None:
    config = qwen3_moe_30b_fsdp_only_full_ac_profile()
    config.parallelism.expert_parallel_degree = 2

    with self.assertRaisesRegex(ValueError, "expert_parallel_degree=1"):
      to_tpu_graph_trainer_config(
          config,
          qwen3_gt.model_registry,
          target_cls=TpuGraphTrainer.Config,
      )

  def test_paired_dweight_elision_requires_matching_kernel_mode(self) -> None:
    config = qwen3_moe_30b_fsdp_only_full_ac_profile()
    assert config.model_spec is not None
    for _, experts, _, _ in config.model_spec.model.traverse(
        TpuGroupedExperts.Config
    ):
      experts.emit_bf16_rounded_fp32_paired_dweight = False

    with self.assertRaisesRegex(ValueError, "every TPU grouped expert"):
      to_tpu_graph_trainer_config(
          config,
          qwen3_gt.model_registry,
          target_cls=TpuGraphTrainer.Config,
      )

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
