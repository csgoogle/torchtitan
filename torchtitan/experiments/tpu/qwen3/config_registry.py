# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import partial
from typing import cast
from torch import nn
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, default_adamw
from torchtitan.components.validate import Validator
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.config.configs import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
import torchtitan.experiments.tpu.qwen3 as tpu_qwen3
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.qwen3.infra.parallelize import parallelize_qwen3 as tpu_parallelize_qwen3
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    LossKernelConfig,
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
)
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.experiments.tpu.qwen3 import model_registry
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3.config_registry import qwen3_30b_a3b_global_aux
from torchtitan.models.common.moe import MoELoadBalanceAuxLoss
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.profiler import Profiler


_FUSED_LOSS_OVERRIDE = "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"
_BF16_ROPE_OVERRIDE = "torchtitan.experiments.tpu.overrides.bf16_rope.tpu_activation_dtype_cossin_rope"


def _c4_test_loader() -> GrainDataLoader.Config:
  return GrainDataLoader.Config(
      dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
      shuffle=False,
  )


def _set_batch(
    training: TrainingConfig,
    *,
    local_batch_size: int,
    seq_len: int,
    global_batch_size: int = -1,
) -> None:
  """Express the old ``(local_batch_size, seq_len, global_batch_size)`` recipe
  geometry in upstream's token budget."""
  training.num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len
  training.max_context_length = seq_len
  training.num_tokens_per_train_step = (
      -1 if global_batch_size < 0 else global_batch_size * seq_len
  )


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Select the fused loss and its pre-construction LM-head override together."""
  config.loss = PallasCrossEntropyLoss.Config()
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)


def qwen3_debugmodel(seq_len: int = 2048) -> TPUTrainerConfig:
  """Qwen3 debug model configuration for TPU (8 rows of ``seq_len`` per rank)."""

  model_spec = model_registry("debugmodel", seq_len=seq_len)
  model_spec.parallelize_fn = tpu_parallelize_qwen3

  model_config = cast(Qwen3Model.Config, model_spec.model)
  # Disable weight tying to mimic the configuration of larger Qwen3 models
  # (e.g. 8B, 14B, 32B). Note that this is different from the non-TPU
  # "debugmodel" in the config registry.
  model_config.enable_weight_tying = False
  model_config.tok_embeddings.param_init = {
      "weight": partial(nn.init.normal_, std=1.0)
  }
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


def qwen3_moe_testmodel() -> TPUTrainerConfig:
  """Qwen3 MoE test model configuration for TPU."""
  cfg = qwen3_debugmodel()
  config = tpu_qwen3.qwen3_configs["testmodel_moe"]
  cfg.model_spec = ModelSpec(
      name="qwen3_tpu",
      flavor="testmodel_moe",
      model=config,
      max_context_length=128,
      parallelize_fn=tpu_parallelize_qwen3,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=Qwen3StateDictAdapter,
  )
  cfg.model_spec.model.enable_weight_tying = False
  cfg.loss = CrossEntropyLoss.Config(
      global_vocab_size=decoder_vocab_size(cfg.model_spec),
  )

  # Override with previous debug_model train_config values
  _set_batch(cfg.training, local_batch_size=4, seq_len=128)

  return cfg


def qwen3_1_7b() -> TPUTrainerConfig:
  """Qwen3 1.7B model configuration for TPU."""
  model_spec = model_registry("1.7B")
  model_config = cast(Qwen3Model.Config, model_spec.model)
  model_config.enable_weight_tying = False
  model_config.tok_embeddings.param_init = {
      "weight": partial(nn.init.normal_, std=1.0)
  }
  model_spec.parallelize_fn = tpu_parallelize_qwen3

  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(
          global_vocab_size=decoder_vocab_size(model_spec),
      ),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=8e-4),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=600,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          max_norm=1.0,
          num_tokens_per_microbatch_per_dp_rank=1 * 2048,
          max_context_length=2048,
          steps=15,
          mixed_precision_reduce="bfloat16",
      ),
      dataloader=_c4_test_loader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=False,
      ),
      activation_checkpoint=FullAC.Config(),
      tpu_config=TPUConfig(
          eager_mode="DEFER_AND_FUSE",
          use_graph_split=True,
          use_jax_profiler=False,
          compile_optimizer=False,
          compile_mode="layer",
      ),
      compile=CompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_block_q=512,
          sa_block_kv=512,
          sa_block_dkv=512,
          sa_block_kv_compute=512,
          sa_block_q_dkv=512,
          sa_block_kv_dkv=512,
          sa_block_kv_dkv_compute=512,
          sa_block_q_dq=512,
          sa_block_kv_dq=512,
      ),
      loss_kernel=LossKernelConfig(
          use_loss_kernel=False,
      ),
      validator=Validator.Config(
          enable=False,
          freq=5,
          steps=10,
      ),
  )


def reg_trainer_qwen3_8b() -> TPUTrainerConfig:
  """Qwen3 8B model configuration for TPU (regular trainer)."""
  model_spec = model_registry("8B")
  model_config = cast(Qwen3Model.Config, model_spec.model)
  model_config.enable_weight_tying = False
  model_config.tok_embeddings.param_init = {
      "weight": partial(nn.init.normal_, std=1.0)
  }
  model_spec.parallelize_fn = tpu_parallelize_qwen3

  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(
          global_vocab_size=decoder_vocab_size(model_spec),
      ),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=8e-4),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=600,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          max_norm=1.0,
          num_tokens_per_microbatch_per_dp_rank=6 * 2048,
          max_context_length=2048,
          steps=15,
          mixed_precision_reduce="float32",
      ),
      dataloader=_c4_test_loader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          enable_tpu_topo_mesh=True,
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=False,
      ),
      activation_checkpoint=FullAC.Config(),
      tpu_config=TPUConfig(
          eager_mode="DEFER_AND_FUSE",
          use_graph_split=True,
          use_jax_profiler=True,
          compile_mode="layer",
      ),
      compile=CompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_block_q=1024,
          sa_block_kv=1024,
          sa_block_dkv=1024,
          sa_block_kv_compute=1024,
          sa_block_q_dkv=1024,
          sa_block_kv_dkv=1024,
          sa_block_kv_dkv_compute=1024,
          sa_block_q_dq=1024,
          sa_block_kv_dq=1024,
      ),
      loss_kernel=LossKernelConfig(
          use_loss_kernel=True,
      ),
      validator=Validator.Config(
          enable=False,
          freq=5,
          steps=10,
      ),
  )


def qwen3_8b() -> TPUTrainerConfig:
  """Qwen3 8B model configuration for TPU."""
  model_spec = model_registry("8B")
  model_config = cast(Qwen3Model.Config, model_spec.model)
  model_config.enable_weight_tying = False
  model_config.tok_embeddings.param_init = {
      "weight": partial(nn.init.normal_, std=1.0)
  }
  model_spec.parallelize_fn = tpu_parallelize_qwen3

  # Optionally override any model config settings here
  # model_config = cast(Qwen3Model.Config, model_spec.model)

  config = TPUTrainerConfig(
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      # Keep the established dense optimizer path until native fused AdamW has
      # a dedicated long-convergence validation for this recipe.
      optimizer=default_adamw(lr=8e-4),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=600,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          max_norm=1.0,
          num_tokens_per_microbatch_per_dp_rank=6 * 2048,
          max_context_length=2048,
          steps=15,
          mixed_precision_reduce="float32",
      ),
      dataloader=_c4_test_loader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          # The current best-performing benchmark recipe uses Zero-2. Keep it
          # explicit while Zero-3 memory/overlap policy is tuned separately.
          fsdp_reshard_after_forward="never",
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=False,
      ),
      # GraphTrainer does not apply this eager AC config; its graph
      # rematerialization policy is selected in TPUConfig below.
      activation_checkpoint=SelectiveAC.Config(),
      tpu_config=TPUConfig(
          use_graph_trainer=True,
          compile_mode="aot_fx_trace",
          compile_optimizer=True,
          optimizer_in_graph=True,
          graph_trainer_memory_policy="default",
      ),
      compile=CompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_block_q=1024,
          sa_block_kv=1024,
          sa_block_dkv=1024,
          sa_block_kv_compute=1024,
          sa_block_q_dkv=1024,
          sa_block_kv_dkv=1024,
          sa_block_kv_dkv_compute=1024,
          sa_block_q_dq=1024,
          sa_block_kv_dq=1024,
      ),
      validator=Validator.Config(
          enable=False,
          freq=5,
          steps=10,
      ),
  )
  _enable_pallas_cross_entropy(config)
  return config


def qwen3_moe_30b() -> TPUTrainerConfig:
  """TPU specialization of the global-aux Qwen3 30B recipe."""
  cfg = TPUTrainerConfig.derive_from(qwen3_30b_a3b_global_aux())
  cfg.model_spec = model_registry(
      "30B-A3B-global-aux",
      converters=[
          TpuTokenDispatcherConverter.Config(),
          TpuGroupedExpertsConverter.Config(),
      ],
  )
  cfg.loss = CrossEntropyLoss.Config(
      global_vocab_size=decoder_vocab_size(cfg.model_spec),
  )
  cfg.model_spec.parallelize_fn = tpu_parallelize_qwen3
  cfg.dataloader = _c4_test_loader()
  _set_batch(
      cfg.training, local_batch_size=6, seq_len=2048, global_batch_size=768
  )
  cfg.parallelism = ParallelismConfig(
      use_simple_fsdp=True,
      data_parallel_shard_degree=-1,
      expert_parallel_degree=128,
  )
  cfg.tpu_config = TPUConfig(
      use_graph_trainer=True,
      compile_mode="aot_fx_trace",
      graph_trainer_memory_policy="full",
  )
  cfg.compile = CompileConfig(
      enable=True,
      backend="tpu",
      components=["model"],
  )
  cfg.splash_attention_kernel = SplashAttentionKernelConfig(
      use_splash_attention_kernel=True,
      sa_block_q=1024,
      sa_block_kv=1024,
      sa_block_dkv=1024,
      sa_block_kv_compute=1024,
      sa_block_q_dkv=1024,
      sa_block_kv_dkv=1024,
      sa_block_kv_dkv_compute=1024,
      sa_block_q_dq=1024,
      sa_block_kv_dq=1024,
  )
  cfg.loss_kernel = LossKernelConfig(use_loss_kernel=False)
  cfg.validator = Validator.Config(enable=False)
  return cfg


def _qwen3_moe_30b_fsdp_only_full_ac() -> TPUTrainerConfig:
  """Batch-6 FSDP128 baseline with local dropless expert dispatch."""
  cfg = qwen3_moe_30b()
  _enable_pallas_cross_entropy(cfg)
  cfg.parallelism = ParallelismConfig(
      use_simple_fsdp=True,
      enable_tpu_topo_mesh=True,
      data_parallel_shard_degree=-1,
      expert_parallel_degree=1,
      # The current best-performing benchmark recipe uses Zero-2. Keep it
      # explicit while Zero-3 memory/overlap policy is tuned separately.
      fsdp_reshard_after_forward="never",
  )
  assert cfg.model_spec is not None
  for _, experts, _, _ in cfg.model_spec.model.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    experts.use_pallas_paired_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 256
    experts.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)
    experts.use_kernel_native_weight_layout = True
  _set_batch(
      cfg.training, local_batch_size=6, seq_len=2048, global_batch_size=768
  )
  cfg.training.steps = 8
  cfg.tpu_config = TPUConfig(
      use_graph_trainer=True,
      compile_mode="aot_fx_trace",
      graph_trainer_memory_policy="full",
      enable_graph_trainer_remat_schedule=True,
      enable_fused_optimizer=True,
      compile_optimizer=False,
      optimizer_in_graph=False,
  )
  cfg.optimizer.implementation = "fused"
  cfg.compile = CompileConfig(
      enable=True,
      backend="tpu",
      components=["model"],
  )
  cfg.splash_attention_kernel = SplashAttentionKernelConfig(
      use_splash_attention_kernel=True,
      # Batch-native masking lets the always-vmapped forward avoid a serial
      # batch loop; use_vmap_bwd applies the same optimization to backward.
      # Runtime segment IDs still preserve exact packed-document boundaries.
      # At local batch 6, the combined change reduced XProf peak HBM from
      # 14.924 to 14.575 GiB (-0.349 GiB); no vmap memory penalty was observed.
      use_vmap_bwd=True,
      sa_use_document_aware_block_skipping=False,
      sa_q_layout="SEQ_MINOR",
      sa_k_layout="SEQ_MINOR",
      sa_block_q=1024,
      sa_block_kv=1024,
      sa_block_dkv=1024,
      # Matched Qwen3-30B/TPU7x profiling showed a 26.3-ms (1.60%) step win and
      # 0.199-GiB lower peak HBM. The attention calls themselves took 2.5 ms
      # longer, but module span fell 24.6 ms and exposed collectives fell
      # 7.4 ms: this is a whole-program resource/scheduling improvement.
      sa_block_kv_compute=512,
      sa_block_q_dkv=2048,
      sa_block_kv_dkv=2048,
      sa_block_kv_dkv_compute=1024,
      sa_block_q_dq=1024,
      sa_block_kv_dq=1024,
  )
  cfg.loss_kernel = LossKernelConfig(use_loss_kernel=False)
  cfg.metrics.log_freq = 1
  cfg.checkpoint = CheckpointManager.Config(enable=False)
  cfg.profiler = Profiler.Config(enable_profiling=False)
  cfg.validator = Validator.Config(enable=False)
  return cfg


def qwen3_moe_30b_fsdp_only_full_ac_profile() -> TPUTrainerConfig:
  """Profile the batch-6 FSDP128 baseline on rank zero."""
  cfg = _qwen3_moe_30b_fsdp_only_full_ac()
  cfg.profiler = Profiler.Config(
      enable_profiling=int(os.environ.get("RANK", "0")) == 0,
      profile_freq=5,
      profiler_warmup=2,
      profiler_active=1,
  )
  return cfg


def qwen3_moe_30b_fsdp_only_full_ac_benchmark() -> TPUTrainerConfig:
  """Profiler-free bracket for the batch-6 FSDP128 comparison."""
  cfg = _qwen3_moe_30b_fsdp_only_full_ac()
  cfg.training.steps = 30
  cfg.metrics.log_freq = 5
  cfg.tpu_config.enable_mfu_benchmark = True
  cfg.tpu_config.mfu_benchmark_start_step = 10
  return cfg


def qwen3_moe_30b_fsdp_only_attention_parity_benchmark() -> TPUTrainerConfig:
  """Q30 benchmark adding MaxText's BF16 RoPE numerical policy.

  The base FSDP recipe already enables the correctness-preserving batch-native
  masking and Qwen3-30B/TPU7x tile choices. This separate recipe changes only
  RoPE arithmetic to BF16 for MaxText parity. That change is not numerically
  equivalent to TorchTitan's FP32 RoPE and must remain opt-in unless longer
  convergence validation justifies promoting it.
  """
  cfg = qwen3_moe_30b_fsdp_only_full_ac_benchmark()
  # Match MaxText's BF16 rotary arithmetic. This is an explicit numerical
  # policy choice, not a numerics-preserving optimization; keep it out of the
  # default recipe pending a longer convergence comparison.
  cfg.override.imports.append(_BF16_ROPE_OVERRIDE)
  return cfg


def _enable_fsdp_only_pallas_experts(cfg: TPUTrainerConfig) -> None:
  """Local dropless dispatch with the Pallas grouped matmuls, no expert parallelism.

  Same expert-kernel and splash settings as ``_qwen3_moe_30b_fsdp_only_full_ac``;
  kept separate so a different model flavor can reuse them.
  """
  assert cfg.model_spec is not None
  for _, experts, _, _ in cfg.model_spec.model.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    experts.use_pallas_paired_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 256
    experts.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)
    experts.use_kernel_native_weight_layout = True
  cfg.splash_attention_kernel = SplashAttentionKernelConfig(
      use_splash_attention_kernel=True,
      use_vmap_bwd=True,
      sa_use_document_aware_block_skipping=False,
      sa_q_layout="SEQ_MINOR",
      sa_k_layout="SEQ_MINOR",
      sa_block_q=1024,
      sa_block_kv=1024,
      sa_block_dkv=1024,
      sa_block_kv_compute=512,
      sa_block_q_dkv=2048,
      sa_block_kv_dkv=2048,
      sa_block_kv_dkv_compute=1024,
      sa_block_q_dq=1024,
      sa_block_kv_dq=1024,
  )
  cfg.loss_kernel = LossKernelConfig(use_loss_kernel=False)
  _enable_pallas_cross_entropy(cfg)


def _enable_scan_layers(cfg: TPUTrainerConfig) -> None:
  """Run the decoder as one scan under the graph trainer with the stacked layer
  input offloaded inside the loop bodies (same recipe as llama3's
  ``llama3_405b_scan``), with the eager fused AdamW."""
  cfg.parallelism = ParallelismConfig(
      use_simple_fsdp=True,
      enable_tpu_topo_mesh=True,
      data_parallel_shard_degree=-1,
      expert_parallel_degree=1,
  )
  cfg.tpu_config = TPUConfig(
      use_graph_trainer=True,
      compile_mode="aot_fx_trace",
      graph_trainer_memory_policy="full_and_scan_offload",
      use_scan_layers=True,
      enable_fused_optimizer=True,
  )
  cfg.optimizer.implementation = "fused"
  cfg.compile = CompileConfig(
      enable=True,
      backend="tpu",
      components=["model"],
  )
  # The routing metrics are captured by the MoE layer writing sidecar
  # attributes during forward, which a scan body cannot do.
  cfg.metrics.enable_moe_routing = False
  # MaxText computes the load-balance loss from the local batch; match it.
  # (The global variant's per-layer all-reduce is synchronous but 0.1 ms.)
  for _, aux, _, _ in cfg.model_spec.model.traverse(MoELoadBalanceAuxLoss.Config):
    aux.reduce_over_dp = False
  cfg.checkpoint = CheckpointManager.Config(enable=False)
  cfg.validator = Validator.Config(enable=False)


def qwen3_235b_scan() -> TPUTrainerConfig:
  """Qwen3-235B-A22B, FSDP-only with local dropless Pallas experts, decoder as one scan.

  Uses the aux-loss balancer (the global-aux flavor, as the 30B recipes do):
  the expert-bias balancer counts routed tokens into a buffer inside the layer,
  which a scan body cannot write back.
  """
  cfg = qwen3_moe_30b()
  cfg.model_spec = model_registry(
      "235B-A22B-global-aux",
      converters=[
          TpuTokenDispatcherConverter.Config(),
          TpuGroupedExpertsConverter.Config(),
      ],
  )
  cfg.loss = CrossEntropyLoss.Config(
      global_vocab_size=decoder_vocab_size(cfg.model_spec),
  )
  cfg.model_spec.parallelize_fn = tpu_parallelize_qwen3
  _enable_fsdp_only_pallas_experts(cfg)
  # At 4096 x 1536 the paired dWeight kernel's full-K accumulators and output
  # slabs need 105 MB of VMEM (64 MB scoped); 4 K slabs bring it under 30 MB.
  for _, experts, _, _ in cfg.model_spec.model.traverse(TpuGroupedExperts.Config):
    experts.pallas_grouped_mm_tgmm_tiling = (256, 1024, 0)
  _enable_scan_layers(cfg)
  # 16 is MaxText's per-device batch for this model and the largest that fits
  # (peak 87.9 of 94.7 GiB on 64 v7x chips); 1,827 tokens/s/device there,
  # 1,903 with --training.mixed_precision_reduce=bfloat16.
  _set_batch(cfg.training, local_batch_size=16, seq_len=4096)
  cfg.training.steps = 1000
  cfg.metrics.log_freq = 5
  return cfg


def qwen3_235b_scan_loss_parity() -> TPUTrainerConfig:
  """`qwen3_235b_scan` pinned to the GPU lane's optimisation settings.

  Trajectory comparison against `qwen3_235b_a22b_loss_parity` on 64 B200 GPUs:
  same aux-loss balancer, same lr 3e-4 with a 20-step linear warmup over 200
  steps, the same 640-sequence global batch (5 x 128 TensorCores against the
  GPU lane's 10 x 64 GPUs), FP32 gradient reduction, seed 42, `c4_test` with
  the bundled test tokenizer. What differs is what is under test: the decoder
  runs as one compiled scan with in-body FSDP gathers, host-offloaded layer
  inputs and Pallas kernels.
  """
  cfg = qwen3_235b_scan()
  cfg.optimizer = default_adamw(lr=3e-4)
  cfg.optimizer.implementation = "fused"
  cfg.lr_scheduler = LRSchedulersContainer.Config(
      warmup_steps=20,
      total_steps=200,
      decay_type="linear",
      min_lr_factor=0.1,
  )
  _set_batch(cfg.training, local_batch_size=5, seq_len=4096)
  cfg.training.steps = 200
  cfg.training.mixed_precision_reduce = "float32"
  cfg.metrics.log_freq = 5
  # The GPU lane uses the canonical global-batch expert frequencies, so undo
  # the MaxText-parity per-device setting for this comparison. The reduction is
  # an all-reduce of [E] counts inside the scan body, measured at 0.1 ms.
  for _, aux, _, _ in cfg.model_spec.model.traverse(MoELoadBalanceAuxLoss.Config):
    aux.reduce_over_dp = True
  return cfg


def qwen3_moe_30b_scan_proxy() -> TPUTrainerConfig:
  """4 layers of the FSDP-only 30B-A3B recipe as one scan: single-host validation of scan + MoE."""
  cfg = _qwen3_moe_30b_fsdp_only_full_ac()
  assert cfg.model_spec is not None
  model_config = cast(Qwen3Model.Config, cfg.model_spec.model)
  model_config.layers = model_config.layers[:4]
  _enable_scan_layers(cfg)
  _set_batch(cfg.training, local_batch_size=2, seq_len=2048)
  cfg.training.steps = 5
  cfg.metrics.log_freq = 1
  return cfg


def qwen3_moe_30b_proxy() -> TPUTrainerConfig:
  """Proxy model for Qwen3 30B MoE (4 layers instead of 48) for testing/iterations."""
  cfg = qwen3_moe_30b()
  assert cfg.model_spec is not None
  model_config = cast(Qwen3Model.Config, cfg.model_spec.model)
  # Slice layers to only keep 4 layers to save memory and compile time
  model_config.layers = model_config.layers[:4]
  return cfg
