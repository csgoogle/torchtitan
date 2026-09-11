# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Glimmer training configs for the TorchTPU lane.

Text-only. The output soft-cap lives in the ``SoftCappedLinear`` lm_head, so
the plain (non-fused) cross entropy is used: the TPU fused Pallas CE kernel
projects the LM head inside the kernel and has no logit-transform hook yet.
"""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.tpu.muse_glimmer import model_registry
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig, TPUTrainerConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size


def _loader(dataset: str, *, shuffle: bool) -> GrainDataLoader.Config:
  return GrainDataLoader.Config(
      dataset=ConcatThenSplitPackingConfig(dataset=DATASETS[dataset]),
      shuffle=shuffle,
  )


def muse_glimmer_debugmodel(seq_len: int = 2048) -> TPUTrainerConfig:
  """Debug flavor (dim 256, 8 layers, vocab 2048, window pattern
  [128, 128, 128, global]); 8 rows of ``seq_len`` per rank, c4_test."""
  model_spec = model_registry("debugmodel", seq_len=seq_len)
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(
          global_vocab_size=decoder_vocab_size(model_spec),
      ),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=8e-4),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * seq_len,
          max_context_length=seq_len,
          steps=10,
      ),
      dataloader=_loader("c4_test", shuffle=False),
      metrics=MetricsProcessor.Config(log_freq=1),
      # Muse Glimmer's sharding configs are written for the spmd_types backend
      # (upstream's parallelizer refuses anything else).
      parallelism=ParallelismConfig(spmd_backend="spmd_types"),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(),
      tpu_config=TPUConfig(),
  )


def muse_glimmer_30b(seq_len: int = 8192) -> TPUTrainerConfig:
  """30B text flavor (dim 6656, 52 layers, 32/2 heads, vocab 202048,
  windows [2048, 2048, 2048, global]); 1 row of ``seq_len`` per rank.

  Tokenizer/config assets: ``assets/hf/Muse-Glimmer-30B`` (from HF
  ``meta-models/Muse-Glimmer-30B``; weights are not needed for pretraining
  from scratch).
  """
  model_spec = model_registry("30B", seq_len=seq_len)
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(
          global_vocab_size=decoder_vocab_size(model_spec),
      ),
      hf_assets_path="./assets/hf/Muse-Glimmer-30B",
      model_spec=model_spec,
      optimizer=default_adamw(lr=3e-4),
      lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=1 * seq_len,
          max_context_length=seq_len,
          steps=1000,
      ),
      dataloader=_loader("c4", shuffle=True),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          spmd_backend="spmd_types",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          context_parallel_degree=1,
          pipeline_parallel_degree=1,
      ),
      checkpoint=CheckpointManager.Config(
          interval=500,
          last_save_model_only=False,
          enable=False,
      ),
      activation_checkpoint=FullAC.Config(),
      tpu_config=TPUConfig(),
  )
