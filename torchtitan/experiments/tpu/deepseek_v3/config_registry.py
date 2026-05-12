# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.deepseek_v3.config_registry import model_registry


def deepseek_v3_debugmodel() -> TPUTrainerConfig:
  """DeepSeek-v3 debug model configuration for TPU."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry("debugmodel"),
      dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
      optimizer=OptimizersContainer.Config(lr=8e-4),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          local_batch_size=8,
          seq_len=2048,
          steps=10,
      ),
      parallelism=ParallelismConfig(
          expert_parallel_degree=1,
      ),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=ActivationCheckpointConfig(
          mode="selective",
      ),
  )
