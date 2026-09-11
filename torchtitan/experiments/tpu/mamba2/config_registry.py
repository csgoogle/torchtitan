# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""TPU job configurations for Mamba-2 models."""

# pylint: disable=invalid-name,unused-argument

from torchtitan.components import checkpoint
from torchtitan.components import loss
from torchtitan.components import lr_scheduler
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


def model_registry(
    flavor: str,
    ssm_backend: str = "pallas",
    **kwargs,
) -> model_spec.ModelSpec:
  config = mamba2_configs[flavor](ssm_backend=ssm_backend, **kwargs)
  return model_spec.ModelSpec(
      name="mamba2",
      flavor=flavor,
      model=config,
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
      model_spec=model_registry("debugmodel"),
      dataloader=text_datasets.HuggingFaceTextDataLoader.Config(
          dataset="c4_test"
      ),
      optimizer=optimizer.default_adamw(lr=8e-4),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          local_batch_size=8, seq_len=512, steps=10
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
      model_spec=model_registry("130M"),
      dataloader=text_datasets.HuggingFaceTextDataLoader.Config(
          dataset="c4_test"
      ),
      optimizer=optimizer.default_adamw(lr=8e-4),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          local_batch_size=16, seq_len=2048, steps=10000
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
      model_spec=model_registry("370M"),
      dataloader=text_datasets.HuggingFaceTextDataLoader.Config(
          dataset="c4_test"
      ),
      optimizer=optimizer.default_adamw(lr=6e-4),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          local_batch_size=8, seq_len=2048, steps=10000
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
      model_spec=model_registry("780M"),
      dataloader=text_datasets.HuggingFaceTextDataLoader.Config(
          dataset="c4_test"
      ),
      optimizer=optimizer.default_adamw(lr=5e-4),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          local_batch_size=4, seq_len=2048, steps=10000
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
      model_spec=model_registry("1.3B"),
      dataloader=text_datasets.HuggingFaceTextDataLoader.Config(
          dataset="c4_test"
      ),
      optimizer=optimizer.default_adamw(lr=4e-4),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          local_batch_size=2, seq_len=2048, steps=10000
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
      model_spec=model_registry("2.7B"),
      dataloader=text_datasets.HuggingFaceTextDataLoader.Config(
          dataset="c4_test"
      ),
      optimizer=optimizer.default_adamw(lr=3e-4),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=1000,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          local_batch_size=2, seq_len=2048, steps=10000
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
      model_spec=model_registry("2.7B"),
      dataloader=text_datasets.HuggingFaceTextDataLoader.Config(
          dataset="c4_test"
      ),
      optimizer=optimizer.default_adamw(lr=3e-4),
      lr_scheduler=lr_scheduler.LRSchedulersContainer.Config(
          warmup_steps=100,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=configs.TrainingConfig(
          local_batch_size=8, seq_len=262144, steps=1000
      ),
      parallelism=configs.ParallelismConfig(expert_parallel_degree=1),
      checkpoint=checkpoint.CheckpointManager.Config(
          interval=100, last_save_model_only=False
      ),
      activation_checkpoint=activation_checkpoint.FullAC.Config(),
  )


# ---------------------------------------------------------------------------
# Recipe configs
#
# One config per recipe variant under torchtitan/experiments/tpu/recipes,
# carrying everything the recipe used to pass on the launcher command line.
# The Mamba-2 long-context recipe needs no libtpu init flags.
# ---------------------------------------------------------------------------


def _mamba2_2_7b_256k_tpu7x_2x2x1(
    local_batch_size: int,
) -> tpu_job_config.TPUTrainerConfig:
  """Shared base for the 256K-context Mamba-2 2.7B recipes on tpu7x-2x2x1.

  Mamba-2's linear O(B * S) attention substitute is what makes a 262,144-token
  sequence fit; the two recipe variants only trade batch size against memory.
  """
  cfg = mamba2_2_7b_256k()
  cfg.training.local_batch_size = local_batch_size
  cfg.training.seq_len = 262144
  cfg.training.steps = 100
  cfg.training.mixed_precision_param = "bfloat16"
  return cfg


def mamba2_2_7b_256k_tpu7x_2x2x1_lbs8() -> tpu_job_config.TPUTrainerConfig:
  """Mamba-2 2.7B at 256K on tpu7x-2x2x1, local batch 8 (global 64).

  See recipes/ironwood/mamba2-2.7b-256k/2x2x1: high throughput and high memory
  utilization, 16.8M tokens per step.
  """
  return _mamba2_2_7b_256k_tpu7x_2x2x1(local_batch_size=8)


def mamba2_2_7b_256k_tpu7x_2x2x1_lbs1() -> tpu_job_config.TPUTrainerConfig:
  """Mamba-2 2.7B at 256K on tpu7x-2x2x1, local batch 1 (global 8).

  See recipes/ironwood/mamba2-2.7b-256k/2x2x1: the low-memory variant, 2.1M
  tokens per step.
  """
  return _mamba2_2_7b_256k_tpu7x_2x2x1(local_batch_size=1)
