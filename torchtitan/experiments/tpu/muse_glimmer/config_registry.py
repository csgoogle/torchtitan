# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Glimmer training configs for the TorchTPU lane.

Text-only. The output soft-cap lives in the ``SoftCappedLinear`` lm_head. The
``*_fused_ce`` configs swap that head for the TPU fused Pallas cross entropy
(``_enable_pallas_cross_entropy``): the kernel projects the LM head per vocab
tile and re-applies the head's ``cap * tanh(z * mul / cap)`` transform to each
tile in-kernel, so the ``[T, 202048]`` fp32 logits are never materialized. The
two base configs keep the plain cross entropy.
"""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import CompileConfig
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.muse_glimmer import model_registry
from torchtitan.experiments.tpu.tpu_job_config import (
    LossKernelConfig,
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
)
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size


def _loader(dataset: str, *, shuffle: bool) -> GrainDataLoader.Config:
  return GrainDataLoader.Config(
      dataset=ConcatThenSplitPackingConfig(dataset=DATASETS[dataset]),
      shuffle=shuffle,
  )


_FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_soft_capped_lm_head"
)


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Select the fused loss and its pre-construction LM-head override together.

  Same idiom as the llama3 / qwen3 registries, with two Muse specifics: the
  override targets ``SoftCappedLinear.Config`` (Muse's lm_head type), and the
  head's soft-cap / output multiplier are copied into the loss config, because
  the wrapper only hands ``(hidden, weight.T)`` to the loss and the transform
  is applied inside the fused kernel. The legacy ``loss_kernel.use_loss_kernel``
  path is turned off for the same reason as in llama3.
  """
  lm_head = config.model_spec.model.lm_head
  config.loss = PallasCrossEntropyLoss.Config(
      soft_cap=lm_head.output_soft_cap_temp,
      output_multiplier=lm_head.output_multiplier,
  )
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)
  config.loss_kernel.use_loss_kernel = False


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


def _splash_2048() -> SplashAttentionKernelConfig:
  """Same splash tiling as the llama3 8B/70B/405B TPU recipes."""
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


def muse_glimmer_30b(seq_len: int = 8192) -> TPUTrainerConfig:
  """30B text flavor (dim 6656, 52 layers, 32/2 heads, vocab 202048,
  windows [2048, 2048, 2048, global]).

  Same shape as the llama3 70B/405B TPU recipes: one row of ``seq_len`` per
  rank, SimpleFSDP, full AC, per-layer TPU compile, splash 2048 tiling,
  DEFER_AND_FUSE eager mode. Plain cross entropy (the lm_head soft-cap has no
  fused-CE hook yet). Data: the bundled ``c4_test`` shard for bring-up.
  Tokenizer/config assets: ``assets/hf/Muse-Glimmer-30B`` (HF
  ``meta-models/Muse-Glimmer-30B``; weights are not needed).
  """
  model_spec = model_registry("30B", seq_len=seq_len)
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(
          global_vocab_size=decoder_vocab_size(model_spec),
      ),
      hf_assets_path="./assets/hf/Muse-Glimmer-30B",
      model_spec=model_spec,
      optimizer=default_adamw(lr=3e-4),
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
      dataloader=_loader("c4_test", shuffle=False),
      metrics=MetricsProcessor.Config(log_freq=5),
      parallelism=ParallelismConfig(
          spmd_backend="spmd_types",
          use_simple_fsdp=True,
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=True,
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
      loss_kernel=LossKernelConfig(use_loss_kernel=False),
  )


def muse_glimmer_debugmodel_fused_ce(seq_len: int = 2048) -> TPUTrainerConfig:
  """``muse_glimmer_debugmodel`` with the soft-capped LM head deferred to the
  fused Pallas cross entropy (cap 20, multiplier 1.0 for the debug flavor)."""
  config = muse_glimmer_debugmodel(seq_len=seq_len)
  _enable_pallas_cross_entropy(config)
  return config


def muse_glimmer_30b_fused_ce(seq_len: int = 8192) -> TPUTrainerConfig:
  """``muse_glimmer_30b`` with the soft-capped LM head deferred to the fused
  Pallas cross entropy (cap 20, multiplier 0.196), so the ``[T, 202048]`` fp32
  logits and their gradient are never materialized in HBM."""
  config = muse_glimmer_30b(seq_len=seq_len)
  _enable_pallas_cross_entropy(config)
  return config


def muse_glimmer_30b_gt(seq_len: int = 8192) -> TPUTrainerConfig:
  """``muse_glimmer_30b_fused_ce`` under the graph trainer: the whole step
  (forward, loss, backward) traced once (``aot_fx_trace``) and compiled into
  one XLA program, with the graph trainer's full-recompute memory policy in
  place of module-level full AC. Same SimpleFSDP data parallelism."""
  config = muse_glimmer_30b_fused_ce(seq_len=seq_len)
  config.tpu_config.use_graph_trainer = True
  config.tpu_config.compile_mode = "aot_fx_trace"
  config.tpu_config.graph_trainer_memory_policy = "full"
  return config
