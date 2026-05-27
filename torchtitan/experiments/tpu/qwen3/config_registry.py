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
from torchtitan.components.validate import Validator
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.experiments.tpu.qwen3.infra.parallelize import parallelize_qwen3 as tpu_parallelize_qwen3
from torchtitan.models.qwen3.config_registry import model_registry
from torchtitan.experiments.tpu.qwen3 import qwen3_configs
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter


def qwen3_debugmodel() -> TPUTrainerConfig:
  """Qwen3 debug model configuration for TPU."""
  model_spec = model_registry("debugmodel")
  model_spec.parallelize_fn = tpu_parallelize_qwen3
  model_spec.model.enable_weight_tying = False
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
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
      dataloader=HuggingFaceTextDataLoader.Config(
          dataset="c4_test",
      ),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          pipeline_parallel_schedule="Interleaved1F1B"
      ),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=ActivationCheckpointConfig(
          mode="selective",
      ),
      validator=Validator.Config(
          freq=5,
          steps=10,
      ),
  )


def qwen3_moe_testmodel() -> TPUTrainerConfig:
  """Qwen3 MoE test model configuration for TPU."""
  cfg = qwen3_debugmodel()
  config = qwen3_configs["testmodel_moe"]
  cfg.model_spec = ModelSpec(
      name="qwen3_tpu",
      flavor="testmodel_moe",
      model=config,
      parallelize_fn=tpu_parallelize_qwen3,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=Qwen3StateDictAdapter,
  )
  cfg.model_spec.model.enable_weight_tying = False

  # Override with previous debug_model train_config values
  cfg.training.seq_len = 128
  cfg.training.local_batch_size = 4

  return cfg


def grpo_qwen3_0_6b() -> TPUTrainerConfig:
  """Qwen3 0.6B model SFT configuration for TPU."""
  cfg = qwen3_debugmodel()
  config = qwen3_configs["0.6B"]
  cfg.model_spec = ModelSpec(
      name="qwen3_tpu",
      flavor="0.6B",
      model=config,
      parallelize_fn=tpu_parallelize_qwen3,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=Qwen3StateDictAdapter,
  )
  cfg.model_spec.model.enable_weight_tying = False

  # Override with previous debug_model train_config values
  cfg.training.seq_len = 512
  cfg.training.local_batch_size = 2
  
  cfg.parallelism.data_parallel_shard_degree = -1
  cfg.parallelism.data_parallel_replicate_degree = 1
  cfg.parallelism.pipeline_parallel_schedule = ""
  cfg.parallelism.tensor_parallel_degree = 1

  return cfg
