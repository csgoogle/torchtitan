# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, default_adamw
from torchtitan.components.validate import Validator
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.config.configs import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
    LossKernelConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.hf_datasets.text_datasets import (
    ChatDataLoader,
    HuggingFaceTextDataLoader,
)
from torchtitan.experiments.tpu.llama3.infra.parallelize import parallelize_llama as tpu_parallelize_llama
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.models.llama3.config_registry import model_registry


def llama3_debugmodel() -> TPUTrainerConfig:
  """Llama3 debug model configuration for TPU."""
  model_spec = model_registry("debugmodel", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
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
      activation_checkpoint=SelectiveAC.Config(),
      validator=Validator.Config(
          freq=5,
          steps=10,
      ),
  )


def llama3_1b() -> TPUTrainerConfig:
  """Llama3 1B model configuration for TPU."""
  model_spec = model_registry("1B", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
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
          local_batch_size=4,
          seq_len=128,
          steps=3,
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
      activation_checkpoint=SelectiveAC.Config(),
      validator=Validator.Config(
          freq=5,
          steps=10,
      ),
  )


def llama3_8b() -> TPUTrainerConfig:
  """Llama3 8B model configuration for TPU."""
  model_spec = model_registry("8B", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  config = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./assets/hf/Llama-3.1-8B",
      model_spec=model_spec,
      optimizer=default_adamw(lr=3e-4, eps=1e-8),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          local_batch_size=1,
          seq_len=8192,
          steps=1000,
      ),
      dataloader=HuggingFaceTextDataLoader.Config(
          dataset="c4_test",
      ),
      metrics=MetricsProcessor.Config(log_freq=5),
      parallelism=ParallelismConfig(use_simple_fsdp=True),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=True,
      ),
      activation_checkpoint=FullAC.Config(),
      validator=Validator.Config(
          enable=False,
          freq=500,
          steps=1200,
          dataloader=HuggingFaceTextDataLoader.Config(
              dataset="c4_test",
          ),
      ),
      tpu_config=TPUConfig(
          eager_mode="DEFER_AND_FUSE",
          use_graph_split=False,
      ),
      compile=CompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_q_layout="SEQ_MINOR",
          sa_k_layout="SEQ_MINOR",
          sa_block_q=2048,
          sa_block_kv=2048,
          sa_block_dkv=2048,
          sa_block_kv_compute=2048,
          sa_block_q_dkv=2048,
          sa_block_kv_dkv=2048,
          sa_block_kv_dkv_compute=2048,
          sa_block_q_dq=2048,
          sa_block_kv_dq=2048,
      ),
      loss_kernel=LossKernelConfig(
          use_loss_kernel=True,
      ),
  )
  config.optimizer.implementation = "foreach"
  return config


def llama3_70b() -> TPUTrainerConfig:
  """Llama3 70B model configuration for TPU."""
  model_spec = model_registry("70B", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=3e-4),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          local_batch_size=1,
          seq_len=8192,
          steps=1000,
      ),
      dataloader=HuggingFaceTextDataLoader.Config(
          dataset="c4_test",
      ),
      metrics=MetricsProcessor.Config(log_freq=5),
      parallelism=ParallelismConfig(use_simple_fsdp=True),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=True,
      ),
      activation_checkpoint=FullAC.Config(),
      validator=Validator.Config(
          enable=False,
          freq=500,
          steps=1200,
          dataloader=HuggingFaceTextDataLoader.Config(
              dataset="c4_test",
          ),
      ),
      tpu_config=TPUConfig(
          eager_mode="DEFER_AND_FUSE",
          use_graph_split=False,
      ),
      compile=CompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_q_layout="SEQ_MINOR",
          sa_k_layout="SEQ_MINOR",
          sa_block_q=2048,
          sa_block_kv=2048,
          sa_block_dkv=2048,
          sa_block_kv_compute=2048,
          sa_block_q_dkv=2048,
          sa_block_kv_dkv=2048,
          sa_block_kv_dkv_compute=2048,
          sa_block_q_dq=2048,
          sa_block_kv_dq=2048,
      ),
      loss_kernel=LossKernelConfig(
          use_loss_kernel=True,
      ),
  )


def llama3_405b() -> TPUTrainerConfig:
  """Llama3 405B model configuration for TPU."""
  model_spec = model_registry("405B", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=3e-4),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          local_batch_size=1,
          seq_len=8192,
          steps=1000,
      ),
      dataloader=HuggingFaceTextDataLoader.Config(
          dataset="c4_test",
      ),
      metrics=MetricsProcessor.Config(log_freq=5),
      parallelism=ParallelismConfig(use_simple_fsdp=True),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=True,
      ),
      activation_checkpoint=FullAC.Config(),
      validator=Validator.Config(
          enable=False,
          freq=500,
          steps=1200,
          dataloader=HuggingFaceTextDataLoader.Config(
              dataset="c4_test",
          ),
      ),
      tpu_config=TPUConfig(
          eager_mode="DEFER_AND_FUSE",
          use_graph_split=False,
      ),
      compile=CompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_q_layout="SEQ_MINOR",
          sa_k_layout="SEQ_MINOR",
          sa_block_q=2048,
          sa_block_kv=2048,
          sa_block_dkv=2048,
          sa_block_kv_compute=2048,
          sa_block_q_dkv=2048,
          sa_block_kv_dkv=2048,
          sa_block_kv_dkv_compute=2048,
          sa_block_q_dq=2048,
          sa_block_kv_dq=2048,
      ),
      loss_kernel=LossKernelConfig(
          use_loss_kernel=True,
      ),
  )


_FUSED_LOSS_OVERRIDE = "torchtitan.experiments.tpu.overrides.fused_loss"


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Select the fused loss and its pre-construction LM-head override together.

  Same mechanism as the qwen3 registry. The legacy ``loss_kernel.use_loss_kernel``
  path swaps the LM head in ``workarounds.apply_patches`` but leaves
  ``config.loss`` as plain cross entropy, which the graph trainer then feeds the
  wrapper's ``(hidden, weight.T)`` tuple -- so it is turned off here.
  """
  config.loss = PallasCrossEntropyLoss.Config()
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)
  config.loss_kernel.use_loss_kernel = False


def _enable_scan_layers(config: TPUTrainerConfig) -> TPUTrainerConfig:
  """Run the decoder as one scan under the graph trainer, offloading the stacked
  layer input inside the loop bodies (PRs 353-361, 373, 374)."""
  config.parallelism.use_simple_fsdp = True
  config.tpu_config.use_graph_trainer = True
  config.tpu_config.compile_mode = "aot_fx_trace"
  config.tpu_config.use_scan_layers = True
  config.tpu_config.graph_trainer_memory_policy = "full_and_scan_offload"
  _enable_pallas_cross_entropy(config)
  return config


def llama3_405b_scan() -> TPUTrainerConfig:
  """Llama3 405B as ``llama3_405b`` plus scan-over-layers + in-loop offload."""
  return _enable_scan_layers(llama3_405b())


def llama3_405b_scan_lr8e5() -> TPUTrainerConfig:
  """``llama3_405b_scan`` with upstream's 405B learning rate (8e-5 instead of the
  70B-derived 3e-4); the lr lives in a param-group dict, so it is not CLI-addressable."""
  config = llama3_405b_scan()
  config.optimizer = default_adamw(lr=8e-5)
  return config


def llama3_debugmodel_scan() -> TPUTrainerConfig:
  """Debug-scale twin of ``llama3_405b_scan`` for single-host validation."""
  return _enable_scan_layers(llama3_debugmodel())


def sft_llama3_8b_metamath() -> TPUTrainerConfig:
  """Llama3 8B SFT on MetaMathQA for TPU."""

  def process_sample(sample):
    # Required by ChatDataset: maps this dataset's columns onto chat messages.
    # `response` holds the full chain-of-thought and the final answer, so it
    # passes through whole -- splitting reasoning into its own key risks a chat
    # template that does not reference that key silently dropping it.
    return [
        {"role": "user", "content": sample["query"]},
        {"role": "assistant", "content": sample["response"]},
    ]

  model_spec = model_registry("8B", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  config = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./assets/hf/Llama-3.1-8B",
      model_spec=model_spec,
      optimizer=default_adamw(lr=2e-5),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=25,
          decay_ratio=0.9,
          decay_type="cosine",
          min_lr_factor=0.1,
      ),
      training=TrainingConfig(
          local_batch_size=2,
          seq_len=8192,
          steps=500,
      ),
      dataloader=ChatDataLoader.Config(
          dataset_path="meta-math/MetaMathQA",
          load_dataset_kwargs={"split": "train"},
          sample_processor=process_sample,
      ),
      metrics=MetricsProcessor.Config(
          log_freq=5,
          enable_tensorboard=True,
          save_tb_folder="tb_logs/sft_llama3_8b_metamath",
      ),
      parallelism=ParallelismConfig(use_simple_fsdp=True),
      checkpoint=CheckpointManager.Config(
          # `enable` is on only so the initial HF weight load happens;
          # `load_only` then suppresses every save, including the
          # unconditional last-step one.
          enable=True,
          load_only=True,
          initial_load_in_hf=True,
          initial_load_model_only=True,
      ),
      activation_checkpoint=FullAC.Config(),
      tpu_config=TPUConfig(
          eager_mode="DEFER_AND_FUSE",
          use_graph_split=False,
      ),
      compile=CompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_q_layout="SEQ_MINOR",
          sa_k_layout="SEQ_MINOR",
          sa_block_q=2048,
          sa_block_kv=2048,
          sa_block_dkv=2048,
          sa_block_kv_compute=2048,
          sa_block_q_dkv=2048,
          sa_block_kv_dkv=2048,
          sa_block_kv_dkv_compute=2048,
          sa_block_q_dq=2048,
          sa_block_kv_dq=2048,
      ),
      loss_kernel=LossKernelConfig(
          use_loss_kernel=True,
      ),
  )
  config.optimizer.implementation = "foreach"
  return config
