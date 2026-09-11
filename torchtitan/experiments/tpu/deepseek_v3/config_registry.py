# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, default_adamw
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.config.configs import (
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    SplashAttentionKernelConfig,
    TPUTrainerConfig,
)
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import DEFAULT_DEBUG_MODEL_SEQ_LEN
from torchtitan.models.deepseek_v3.config_registry import model_registry


def deepseek_v3_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> TPUTrainerConfig:
  """DeepSeek-v3 debug model configuration for TPU."""
  # TODO(tpu): swap in the splash converter (``converters=[...]``) once the
  # first-class TPU attention backend lands; "sdpa" no longer exists upstream.
  model_spec = model_registry("debugmodel", seq_len=seq_len, attn_backend="flex")
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=MetricsProcessor.Config(log_freq=1),
      model_spec=model_spec,
      dataloader=GrainDataLoader.Config(
          dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
      ),
      optimizer=default_adamw(lr=8e-4),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * model_spec.max_context_length,
          max_context_length=model_spec.max_context_length,
          steps=10,
      ),
      parallelism=ParallelismConfig(
          expert_parallel_degree=1,
      ),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(),
  )
