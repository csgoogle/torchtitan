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
import torch
from torch import nn
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.config import apply_overrides
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.graph_trainer import qwen3 as qwen3_gt
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.tpu.graph_trainer.config import (
    to_tpu_graph_trainer_config,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
    construct_tpu_pass_pipeline,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.graph_trainer.qwen3 import (
    model_registry as tpu_qwen3_graph_registry,
)
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.overrides.fused_loss import (
    FusedLossLinearWrapper,
    fused_loss_lm_head,
)
from torchtitan.experiments.tpu.qwen3.config_registry import (
    qwen3_8b,
    qwen3_moe_30b,
    qwen3_moe_30b_fsdp_only_attention_parity_benchmark,
    qwen3_moe_30b_fsdp_only_full_ac_benchmark,
    qwen3_moe_30b_fsdp_only_full_ac_profile,
)
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.models.common import Linear


class TpuGraphTrainerConfigTest(parameterized.TestCase):

  def test_qwen3_graph_config_preserves_maxtext_flops(self) -> None:
    config = qwen3_moe_30b()
    graph_config = to_tpu_graph_trainer_config(
        config,
        tpu_qwen3_graph_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    with torch.device("meta"):
      model = graph_config.model_spec.model.build()

    self.assertIsInstance(model, GraphTrainerQwen3Model)
    self.assertEqual(
        graph_config.model_spec.model.get_nparams_and_flops(model, seq_len=2048)[1],
        20_665_860_096,
    )

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

    self.assertIsInstance(config.loss, PallasCrossEntropyLoss.Config)
    self.assertEqual(
        config.tpu_config.graph_trainer_memory_policy,
        "default",
    )
    self.assertEqual(config.parallelism.fsdp_reshard_after_forward, "never")

    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )
    self.assertIsInstance(graph_config.loss, PallasCrossEntropyLoss.Config)
    self.assertEqual(graph_config.compile.memory_policy, "default")

  def test_fsdp_only_recipe_uses_fused_adamw(self) -> None:
    config = qwen3_moe_30b_fsdp_only_full_ac_benchmark()

    self.assertEqual(config.optimizer.implementation, "fused")
    self.assertTrue(config.tpu_config.enable_fused_optimizer)
    self.assertEqual(config.splash_attention_kernel.sa_q_layout, "SEQ_MINOR")
    self.assertEqual(config.splash_attention_kernel.sa_k_layout, "SEQ_MINOR")
    self.assertEqual(
        config.splash_attention_kernel.sa_v_layout,
        "HEAD_DIM_MINOR",
    )

  def test_attention_parity_recipe_uses_batch_native_maxtext_tiles(self) -> None:
    config = qwen3_moe_30b_fsdp_only_attention_parity_benchmark()
    splash = config.splash_attention_kernel

    self.assertTrue(splash.use_vmap_bwd)
    self.assertFalse(splash.sa_use_document_aware_block_skipping)
    self.assertEqual(splash.sa_block_q, 1024)
    self.assertEqual(splash.sa_block_kv, 1024)
    self.assertEqual(splash.sa_block_kv_compute, 512)
    self.assertEqual(splash.sa_block_q_dkv, 2048)
    self.assertEqual(splash.sa_block_kv_dkv, 2048)
    self.assertEqual(splash.sa_block_kv_dkv_compute, 1024)
  def test_dense_recipe_does_not_enable_fused_adamw_on_tpu(self) -> None:
    config = qwen3_8b()

    self.assertFalse(config.tpu_config.enable_fused_optimizer)

  def test_qwen3_30b_fsdp_profile_explicitly_uses_zero2(self) -> None:
    config = qwen3_moe_30b_fsdp_only_full_ac_profile()

    self.assertEqual(config.tpu_config.graph_trainer_memory_policy, "full")
    self.assertEqual(config.parallelism.fsdp_reshard_after_forward, "never")

  def test_qwen3_moe_fsdp_uses_pallas_cross_entropy(self) -> None:
    config = qwen3_moe_30b_fsdp_only_full_ac_profile()

    self.assertIsInstance(config.loss, PallasCrossEntropyLoss.Config)

  @parameterized.parameters(qwen3_8b, qwen3_moe_30b_fsdp_only_full_ac_profile)
  def test_pallas_loss_recipe_overrides_lm_head(self, recipe) -> None:
    config = recipe()
    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertIn(
        "torchtitan.experiments.tpu.overrides.fused_loss",
        graph_config.override.imports,
    )
    apply_overrides(graph_config.override, graph_config)
    assert graph_config.model_spec is not None
    lm_head_config = graph_config.model_spec.model.lm_head
    self.assertIsInstance(lm_head_config, FusedLossLinearWrapper.Config)

    with torch.device("meta"):
      model = graph_config.model_spec.model.build()
    self.assertIsInstance(model.lm_head, FusedLossLinearWrapper)
    self.assertEqual(
        [name for name in model.state_dict() if name.endswith("lm_head.weight")],
        ["lm_head.weight"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer_params = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    self.assertTrue(
        any(parameter is model.lm_head.weight for parameter in optimizer_params)
    )

  def test_fused_loss_linear_wrapper_contract(self) -> None:
    param_init = {"weight": lambda weight: nn.init.constant_(weight, 0.25)}
    sharding_config = object()
    linear_config = Linear.Config(
        in_features=4,
        out_features=8,
        param_init=param_init,
        sharding_config=sharding_config,
    )
    config = fused_loss_lm_head(linear_config)

    self.assertIs(config.param_init, linear_config.param_init)
    self.assertIs(config.sharding_config, linear_config.sharding_config)
    lm_head = config.build()
    lm_head.init_states()
    hidden = torch.randn(2, 4)
    output, output_weight = lm_head(hidden)

    self.assertIs(output, hidden)
    self.assertEqual(list(lm_head.state_dict()), ["weight"])
    torch.testing.assert_close(
        lm_head.weight,
        torch.full_like(lm_head.weight, 0.25),
    )
    torch.testing.assert_close(output_weight, lm_head.weight.t())

  def test_qwen3_moe_ep_keeps_plain_cross_entropy(self) -> None:
    config = qwen3_moe_30b()

    self.assertIsInstance(config.loss, CrossEntropyLoss.Config)

  def test_chunked_loss_is_rejected(self) -> None:
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
        ]),
    )
    config.loss = ChunkedLossWrapper.Config()
    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    with self.assertRaisesRegex(
        ValueError, "TPU GraphTrainer does not support ChunkedLossWrapper"
    ):
      TpuGraphTrainer(graph_config)

  def test_remat_schedule_supports_optimizer_in_graph(self) -> None:
    config = qwen3_8b()
    config.tpu_config.enable_graph_trainer_remat_schedule = True
    config.tpu_config.optimizer_in_graph = True
    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    passes = construct_tpu_pass_pipeline(
        typing.cast(typing.Any, SimpleNamespace(num_static_inputs=0)),
        graph_config,
    )
    pass_names = [
        getattr(graph_pass, "func", graph_pass).__name__ for graph_pass in passes
    ]

    self.assertIn("apply_tpu_remat_schedule", pass_names)
    self.assertEqual(pass_names[0], "remove_parameter_gradient_markers_pass")


if __name__ == "__main__":
  absltest.main()
