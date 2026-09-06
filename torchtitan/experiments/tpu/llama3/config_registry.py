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
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.hf_datasets.text_datasets import (
    ChatDataLoader,
    HuggingFaceTextDataLoader,
)
from torchtitan.experiments.tpu.llama3.infra.parallelize import parallelize_llama as tpu_parallelize_llama
from torchtitan.models.llama3.config_registry import model_registry


# Llama3's `ComplexRoPE` calls `view_as_complex`, which cannot be traced inside
# a higher-order op such as `scan`; see `overrides/real_rope.py`. The rewrite is
# bit-identical and perf-neutral, so every recipe carries it.
_REAL_ROPE_OVERRIDE = "torchtitan.experiments.tpu.overrides.real_rope"

_SPLASH_BLOCKS = (
    "q",
    "kv",
    "dkv",
    "kv_compute",
    "q_dkv",
    "kv_dkv",
    "kv_dkv_compute",
    "q_dq",
    "kv_dq",
)


def llama3_debugmodel() -> TPUTrainerConfig:
  """Llama3 debug model configuration for TPU."""
  model_spec = model_registry("debugmodel", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  config = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
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
  config.override.imports.append(_REAL_ROPE_OVERRIDE)
  return config


def llama3_debugmodel_kernels() -> TPUTrainerConfig:
  """`llama3_debugmodel` with the kernel set that 8B/70B/405B actually run.

  `llama3_debugmodel` is plain SDPA with `SelectiveAC` and no kernels; `8B`,
  `70B` and `405B` use `FullAC`, splash attention and the Pallas loss. Anything
  validated only on `llama3_debugmodel` is therefore never once run against the
  code path it ships into -- it passes, then fails at 8B on a kernel the debug
  config does not contain. This closes that gap at debug cost.

  Most of what follows *is* expressible as a command-line flag, and is written
  here only so the recipe is runnable as one name. The one thing that is not is
  the model itself: `model_spec` is excluded from the tyro surface (see the
  comment in `protocols/model_spec.py`), so `--model_spec.flavor=...` and
  `--model_spec.model.dim=...` are both rejected. `debugmodel` is `head_dim=16`
  and splash attention needs 128, so selecting the flavour in Python is the
  only way to reach a debug-sized model the kernels will accept.

  Sized for **all 8 chips of a v6e-8**, not one: FSDP is the thing under test,
  and at world size 1 there is nothing to gather.
  """
  config = llama3_debugmodel()
  config.model_spec = model_registry("debugmodel_kernels", attn_backend="sdpa")
  config.model_spec.parallelize_fn = tpu_parallelize_llama

  # The fused loss is selected the new way -- `PallasCrossEntropyLoss.Config`
  # plus the LM-head override below -- not by `loss_kernel.use_loss_kernel`.
  # The legacy flag is applied inside `build_tpu_loss`, which the graph trainer
  # never calls: `compute_annotated_loss` reads `job_config.loss` directly, so
  # the flag would patch the LM head into returning `(hidden, weight.T)` while
  # the loss stayed ordinary cross entropy, and the run dies in `pred.flatten`
  # on a tuple. The override swaps that LM head in, and has to be registered
  # before the model is constructed.
  config.loss = PallasCrossEntropyLoss.Config()
  config.override.imports.append(
      "torchtitan.experiments.tpu.overrides.fused_loss"
  )

  config.training.local_batch_size = 1
  config.parallelism = ParallelismConfig(use_simple_fsdp=True)
  config.activation_checkpoint = FullAC.Config()
  config.checkpoint = CheckpointManager.Config(
      enable=False, interval=500, last_save_model_only=True
  )
  config.validator = Validator.Config(enable=False, freq=500, steps=10)
  config.compile = CompileConfig(enable=True, backend="tpu", components=["model"])
  config.tpu_config = TPUConfig(
      eager_mode="DEFER_AND_FUSE",
      use_graph_split=False,
      use_graph_trainer=True,
      compile_mode="aot_fx_trace",
  )
  # Block sizes are the sequence length quartered, as in the 8B recipe --
  # scaled down with `seq_len` rather than copied from it, since a block larger
  # than the sequence is not a valid splash configuration.
  config.splash_attention_kernel = SplashAttentionKernelConfig(
      use_splash_attention_kernel=True,
      **{f"sa_block_{n}": 512 for n in _SPLASH_BLOCKS},
  )
  return config


def llama3_1b() -> TPUTrainerConfig:
  """Llama3 1B model configuration for TPU."""
  model_spec = model_registry("1B", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  config = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
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
  config.override.imports.append(_REAL_ROPE_OVERRIDE)
  return config


def llama3_8b() -> TPUTrainerConfig:
  """Llama3 8B model configuration for TPU."""
  model_spec = model_registry("8B", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  config = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./assets/hf/Llama-3.1-8B",
      model_spec=model_spec,
      optimizer=default_adamw(lr=3e-4, eps=1e-8, implementation="foreach"),
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
  config.override.imports.append(_REAL_ROPE_OVERRIDE)
  return config


def llama3_70b() -> TPUTrainerConfig:
  """Llama3 70B model configuration for TPU."""
  model_spec = model_registry("70B", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  config = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=3e-4, implementation="foreach"),
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
  config.override.imports.append(_REAL_ROPE_OVERRIDE)
  return config


def llama3_405b() -> TPUTrainerConfig:
  """Llama3 405B model configuration for TPU."""
  model_spec = model_registry("405B", attn_backend="sdpa")
  model_spec.parallelize_fn = tpu_parallelize_llama
  config = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=3e-4, implementation="foreach"),
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
  config.override.imports.append(_REAL_ROPE_OVERRIDE)
  return config


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
      optimizer=default_adamw(lr=2e-5, implementation="foreach"),
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
  config.override.imports.append(_REAL_ROPE_OVERRIDE)
  return config
