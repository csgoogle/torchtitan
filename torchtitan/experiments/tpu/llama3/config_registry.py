# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Llama3 recipes for the TorchTPU lane.

Batch geometry follows upstream's token-budget ``TrainingConfig``: a recipe
that used to say ``local_batch_size=B, seq_len=S`` now says
``num_tokens_per_microbatch_per_dp_rank=B * S, max_context_length=S`` (and
``num_tokens_per_train_step=G * S`` for a global batch of ``G`` rows).
"""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.validate import Validator
from torchtitan.config.configs import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.tpu.llama3 import model_registry
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.tpu_job_config import (
    LossKernelConfig,
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
)
from torchtitan.hf_datasets.text_datasets import ChatProcessor, DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size


def _c4_test_loader() -> GrainDataLoader.Config:
  return GrainDataLoader.Config(
      dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
      shuffle=False,
  )


# Block sizes shared by the large llama3 recipes: 2048 everywhere, with the
# sequence-minor Q/K layouts that keep the scan arm's all-gather overlapped
# (see tests/test_splash_layouts.py).
def _splash_2048() -> SplashAttentionKernelConfig:
  return SplashAttentionKernelConfig(
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
  )


def llama3_debugmodel(seq_len: int = 2048) -> TPUTrainerConfig:
  """Llama3 debug model configuration for TPU (8 rows of ``seq_len`` per rank)."""
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
      dataloader=_c4_test_loader(),
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
          dataloader=_c4_test_loader(),
      ),
  )


def llama3_1b(seq_len: int = 128) -> TPUTrainerConfig:
  """Llama3 1B model configuration for TPU."""
  model_spec = model_registry("1B", seq_len=seq_len)
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
          num_tokens_per_microbatch_per_dp_rank=4 * seq_len,
          max_context_length=seq_len,
          steps=3,
      ),
      dataloader=_c4_test_loader(),
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
          dataloader=_c4_test_loader(),
      ),
  )


def _large_recipe(
    flavor: str,
    *,
    hf_assets_path: str,
    lr: float,
    eps: float | None = None,
    seq_len: int = 8192,
) -> TPUTrainerConfig:
  """Shared shape of the 8B / 70B / 405B recipes: one row per rank, SimpleFSDP,
  full AC, per-layer TPU compile, splash 2048 and the Pallas loss kernel."""
  model_spec = model_registry(flavor, seq_len=seq_len)
  optimizer = default_adamw(lr=lr) if eps is None else default_adamw(lr=lr, eps=eps)
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(
          global_vocab_size=decoder_vocab_size(model_spec),
      ),
      hf_assets_path=hf_assets_path,
      model_spec=model_spec,
      optimizer=optimizer,
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=seq_len,
          max_context_length=seq_len,
          steps=1000,
      ),
      dataloader=_c4_test_loader(),
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
          dataloader=_c4_test_loader(),
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
      splash_attention_kernel=_splash_2048(),
      loss_kernel=LossKernelConfig(
          use_loss_kernel=True,
      ),
  )


def llama3_8b() -> TPUTrainerConfig:
  """Llama3 8B model configuration for TPU."""
  config = _large_recipe(
      "8B", hf_assets_path="./assets/hf/Llama-3.1-8B", lr=3e-4, eps=1e-8
  )
  config.optimizer.implementation = "foreach"
  return config


def llama3_70b() -> TPUTrainerConfig:
  """Llama3 70B model configuration for TPU."""
  return _large_recipe("70B", hf_assets_path="./tests/assets/tokenizer", lr=3e-4)


def llama3_405b() -> TPUTrainerConfig:
  """Llama3 405B model configuration for TPU."""
  return _large_recipe("405B", hf_assets_path="./tests/assets/tokenizer", lr=3e-4)


_FUSED_LOSS_OVERRIDE = "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"


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


def sft_llama3_8b_metamath(seq_len: int = 8192) -> TPUTrainerConfig:
  """Llama3 8B SFT on MetaMathQA for TPU."""

  def process_sample(sample):
    # Required by ChatProcessor: maps this dataset's columns onto chat messages.
    # `response` holds the full chain-of-thought and the final answer, so it
    # passes through whole -- splitting reasoning into its own key risks a chat
    # template that does not reference that key silently dropping it.
    return [
        {"role": "user", "content": sample["query"]},
        {"role": "assistant", "content": sample["response"]},
    ]

  model_spec = model_registry("8B", seq_len=seq_len)
  metamath = SingleDatasetConfig(
      source=HuggingFaceStreamingSource.Config(
          path="meta-math/MetaMathQA", split="train"
      ),
      processor=ChatProcessor.Config(messages_fn=process_sample),
      post_filters=(lambda sample: sample is not None,),
  )
  config = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(
          global_vocab_size=decoder_vocab_size(model_spec),
      ),
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
          num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
          max_context_length=seq_len,
          steps=500,
      ),
      dataloader=GrainDataLoader.Config(
          dataset=ConcatThenSplitPackingConfig(dataset=metamath),
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
      splash_attention_kernel=_splash_2048(),
      loss_kernel=LossKernelConfig(
          use_loss_kernel=True,
      ),
  )
  config.optimizer.implementation = "foreach"
  return config
