# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""TPU job configurations for Mamba-2 models."""

# pylint: disable=invalid-name,unused-argument

from torchtitan.components import checkpointer as checkpoint
from torchtitan.components import data as data_components
from torchtitan.components import loss
from torchtitan.components import optimizer as lr_scheduler
from torchtitan.components import metrics
from torchtitan.components import optimizer
from torchtitan.config import configs
from torchtitan.distributed import activation_checkpoint
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.hf_datasets import text_datasets
from torchtitan.protocols import model_spec
from . import model
from . import parallelize


def _debugmodel(ssm_backend: str = "pallas", **kwargs) -> model.Mamba2Config:
  return model.Mamba2Config(
      vocab_size=2048,
      d_model=128,
      n_layers=2,
      n_heads=4,
      d_state=16,
      d_conv=4,
      expand=2,
      ssm_backend=ssm_backend,
  )


def _130m(ssm_backend: str = "pallas", **kwargs) -> model.Mamba2Config:
  return model.Mamba2Config(
      vocab_size=128256,
      d_model=768,
      n_layers=24,
      n_heads=24,
      d_state=16,
      d_conv=4,
      expand=2,
      ssm_backend=ssm_backend,
  )


def _370m(ssm_backend: str = "pallas", **kwargs) -> model.Mamba2Config:
  return model.Mamba2Config(
      vocab_size=128256,
      d_model=1024,
      n_layers=48,
      n_heads=32,
      d_state=16,
      d_conv=4,
      expand=2,
      ssm_backend=ssm_backend,
  )


def _780m(ssm_backend: str = "pallas", **kwargs) -> model.Mamba2Config:
  return model.Mamba2Config(
      vocab_size=128256,
      d_model=1536,
      n_layers=48,
      n_heads=48,
      d_state=16,
      d_conv=4,
      expand=2,
      ssm_backend=ssm_backend,
  )


def _1_3b(ssm_backend: str = "pallas", **kwargs) -> model.Mamba2Config:
  return model.Mamba2Config(
      vocab_size=128256,
      d_model=2048,
      n_layers=48,
      n_heads=64,
      d_state=16,
      d_conv=4,
      expand=2,
      ssm_backend=ssm_backend,
  )


def _2_7b(ssm_backend: str = "pallas", **kwargs) -> model.Mamba2Config:
  return model.Mamba2Config(
      vocab_size=128256,
      d_model=2560,
      n_layers=64,
      n_heads=80,
      d_state=16,
      d_conv=4,
      expand=2,
      ssm_backend=ssm_backend,
  )


mamba2_configs = {
    "debugmodel": _debugmodel,
    "130M": _130m,
    "130m_pallas": _130m,
    "370M": _370m,
    "780M": _780m,
    "1.3B": _1_3b,
    "2.7B": _2_7b,
}


DEFAULT_MAX_CONTEXT_LENGTH = 2048


def model_registry(
    flavor: str,
    ssm_backend: str = "pallas",
    *,
    seq_len: int | None = None,
    **kwargs,
) -> model_spec.ModelSpec:
  # Mamba-2 has no positional cache, so ``seq_len`` only reports the training
  # context length on the ModelSpec (mirrors upstream ``model_registry``).
  config = mamba2_configs[flavor](ssm_backend=ssm_backend, **kwargs)
  return model_spec.ModelSpec(
      name="mamba2",
      flavor=flavor,
      model=config,
      max_context_length=seq_len or DEFAULT_MAX_CONTEXT_LENGTH,
      parallelize_fn=parallelize.parallelize_mamba2,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def mamba2_debug() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry("debugmodel", seq_len=512),
      dataloader=data_components.GrainDataLoader.Config(
          dataset=data_components.ConcatThenSplitPackingConfig(
              dataset=text_datasets.DATASETS["c4_test"]
          ),
      ),
      optimizer=optimizer.default_adamw(lr=8e-4),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * 512,
          max_context_length=512,
          steps=10,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=10, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.SelectiveAC.Config(),
  )


def mamba2_130m() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=10),
      model_spec=model_registry("130M", seq_len=2048),
      dataloader=data_components.GrainDataLoader.Config(
          dataset=data_components.ConcatThenSplitPackingConfig(
              dataset=text_datasets.DATASETS["c4_test"]
          ),
      ),
      optimizer=optimizer.default_adamw(lr=8e-4),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=16 * 2048,
          max_context_length=2048,
          steps=10000,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=1000, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.SelectiveAC.Config(),
  )


def mamba2_370m() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=10),
      model_spec=model_registry("370M", seq_len=2048),
      dataloader=data_components.GrainDataLoader.Config(
          dataset=data_components.ConcatThenSplitPackingConfig(
              dataset=text_datasets.DATASETS["c4_test"]
          ),
      ),
      optimizer=optimizer.default_adamw(lr=6e-4),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * 2048,
          max_context_length=2048,
          steps=10000,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=1000, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.SelectiveAC.Config(),
  )


def mamba2_780m() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=10),
      model_spec=model_registry("780M", seq_len=2048),
      dataloader=data_components.GrainDataLoader.Config(
          dataset=data_components.ConcatThenSplitPackingConfig(
              dataset=text_datasets.DATASETS["c4_test"]
          ),
      ),
      optimizer=optimizer.default_adamw(lr=5e-4),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=4 * 2048,
          max_context_length=2048,
          steps=10000,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=1000, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.SelectiveAC.Config(),
  )


def mamba2_1_3b() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=10),
      model_spec=model_registry("1.3B", seq_len=2048),
      dataloader=data_components.GrainDataLoader.Config(
          dataset=data_components.ConcatThenSplitPackingConfig(
              dataset=text_datasets.DATASETS["c4_test"]
          ),
      ),
      optimizer=optimizer.default_adamw(lr=4e-4),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=2 * 2048,
          max_context_length=2048,
          steps=10000,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=1000, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.SelectiveAC.Config(),
  )


def mamba2_2_7b() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=10),
      model_spec=model_registry("2.7B", seq_len=2048),
      dataloader=data_components.GrainDataLoader.Config(
          dataset=data_components.ConcatThenSplitPackingConfig(
              dataset=text_datasets.DATASETS["c4_test"]
          ),
      ),
      optimizer=optimizer.default_adamw(lr=3e-4),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=2 * 2048,
          max_context_length=2048,
          steps=10000,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=1000, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.SelectiveAC.Config(),
  )


def mamba2_2_7b_256k() -> tpu_job_config.TPUTrainerConfig:
  return tpu_job_config.TPUTrainerConfig(
      loss=loss.CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=metrics.MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry("2.7B", seq_len=262144),
      dataloader=data_components.GrainDataLoader.Config(
          dataset=data_components.ConcatThenSplitPackingConfig(
              dataset=text_datasets.DATASETS["c4_test"]
          ),
      ),
      optimizer=optimizer.default_adamw(lr=3e-4),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=100,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * 262144,
          max_context_length=262144,
          steps=1000,
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=100, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.FullAC.Config(),
  )
