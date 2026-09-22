# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import os
from typing import cast
from torch import nn
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.qkv import TpuPackedQKVConverter
import torchtitan.experiments.tpu.qwen3 as tpu_qwen3
from torchtitan.experiments.tpu.qwen3 import model_registry
from torchtitan.experiments.tpu.qwen3.infra.parallelize import parallelize_qwen3 as tpu_parallelize_qwen3
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    LossKernelConfig,
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import make_aux_loss_config
from torchtitan.models.common.moe import MoE
from torchtitan.models.qwen3.config_registry import qwen3_30b_a3b_global_aux
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.profiler import Profiler


_FUSED_LOSS_OVERRIDE = "torchtitan.experiments.tpu.overrides.fused_loss"
_BF16_ROPE_OVERRIDE = "torchtitan.experiments.tpu.overrides.bf16_rope"
_DISCARDED_TOKEN_COUNTER_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.discarded_token_counter"
)


# Shared by both Qwen3-235B lines, the scan loop and the graph splitter.
_Q235_SHARED_FLAGS = tpu_flags.merge_flags(
    # Select TPU7x's throughput-oriented operating point for this compute-bound
    # recipe. This changes device frequency, not graph numerics.
    tpu_flags.TPU7X_THROUGHPUT_P_STATE,
    {
        # Raise the scoped VMEM budget to 128 MiB so the compiler can retain
        # more Q235 kernel intermediates in VMEM instead of spilling them.
        "xla_tpu_scoped_vmem_limit_kib": 131072,
        # Keep activation host copies off SparseCore, where they would contend
        # with the FSDP collectives. In the isolated A/B this moved MFU only
        # from 18.425% to 18.448% and did not itself create D2H overlap.
        "xla_tpu_enable_offloading_copy_to_sparsecore": False,
    },
)

# What the graph-splitter line adds on top.
_Q235_COMPILED_OFFLOAD_FLAGS = tpu_flags.merge_flags(
    _Q235_SHARED_FLAGS,
    # Aggregate SparseCore collectives and enable layer-level latency hiding so
    # FSDP communication can be scheduled alongside independent layer compute.
    tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
    # Use both SparseCores for reduce-scatter in the by-layer FSDP schedule.
    tpu_flags.DUAL_SPARSECORE_REDUCE_SCATTER,
    {
        # Prove the true copy/overwrite ordering across donated aliases. This
        # removed the defensive activation snapshot, moved all 93 measured D2H
        # waits to zero, and improved MFU from 18.528% to 19.490% in the A/B.
        "xla_tpu_copy_insertion_use_region_analysis": True,
        # Use the tested search budget for the large Q235 layer regions. This
        # value was evaluated together with region analysis, not in isolation.
        "xla_tpu_copy_insertion_use_region_analysis_limit": 1000,
    },
)


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Select the fused loss and its pre-construction LM-head override together."""
  config.loss = PallasCrossEntropyLoss.Config()
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)


def _use_global_batch_aux_loss(config: TPUTrainerConfig) -> None:
  """Use Qwen's global auxiliary load-balancing loss for every MoE layer."""
  assert config.model_spec is not None
  for _, moe, _, _ in config.model_spec.model.traverse(MoE.Config):
    moe.load_balance_coeff = None
    moe.aux_loss = make_aux_loss_config(
        type="batch_wise", coeff=1e-3, top_k=moe.router.top_k
    )


def qwen3_debugmodel() -> TPUTrainerConfig:
  """Qwen3 debug model configuration for TPU."""

  model_spec = model_registry("debugmodel", attn_backend="sdpa")
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


def qwen3_moe_testmodel() -> TPUTrainerConfig:
  """Qwen3 MoE test model configuration for TPU."""
  cfg = qwen3_debugmodel()
  config = tpu_qwen3.qwen3_configs["testmodel_moe"]
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


def qwen3_1_7b() -> TPUTrainerConfig:
  """Qwen3 1.7B model configuration for TPU."""
  model_spec = model_registry("1.7B", attn_backend="sdpa")
  model_config = cast(Qwen3Model.Config, model_spec.model)
  model_config.enable_weight_tying = False
  model_config.tok_embeddings.param_init = {
      "weight": partial(nn.init.normal_, std=1.0)
  }
  model_spec.parallelize_fn = tpu_parallelize_qwen3

  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=600,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          max_norm=1.0,
          local_batch_size=1,
          seq_len=2048,
          steps=15,
          mixed_precision_reduce="bfloat16",
      ),
      dataloader=HuggingFaceTextDataLoader.Config(
          dataset="c4_test",
      ),
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
  model_spec = model_registry("8B", attn_backend="sdpa")
  model_config = cast(Qwen3Model.Config, model_spec.model)
  model_config.enable_weight_tying = False
  model_config.tok_embeddings.param_init = {
      "weight": partial(nn.init.normal_, std=1.0)
  }
  model_spec.parallelize_fn = tpu_parallelize_qwen3

  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=600,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          max_norm=1.0,
          local_batch_size=6,
          seq_len=2048,
          steps=15,
          mixed_precision_reduce="float32",
      ),
      dataloader=HuggingFaceTextDataLoader.Config(
          dataset="c4_test",
      ),
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
          compile_grad_norm=True,
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
  model_spec = model_registry("8B", attn_backend="sdpa")
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
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=600,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          max_norm=1.0,
          local_batch_size=6,
          seq_len=2048,
          steps=15,
          mixed_precision_reduce="float32",
      ),
      dataloader=HuggingFaceTextDataLoader.Config(
          dataset="c4_test",
      ),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          enable_tpu_topo_mesh=True,
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
          sa_q_layout="SEQ_MINOR",
          sa_k_layout="SEQ_MINOR",
          sa_v_layout="HEAD_DIM_MINOR",
          sa_block_q=1024,
          sa_block_kv=1024,
          sa_block_dkv=1024,
          sa_block_kv_compute=512,
          sa_block_q_dkv=2048,
          sa_block_kv_dkv=2048,
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


def qwen3_8b_zero3() -> TPUTrainerConfig:
  """Qwen3 8B GraphTrainer recipe retaining sharded parameters after forward."""
  cfg = qwen3_8b()
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  cfg.tpu_config.graph_trainer_memory_policy = "full"
  cfg.tpu_config.enable_graph_trainer_remat_schedule = True
  cfg.tpu_config.graph_trainer_fsdp_schedule = "interleave"
  cfg.splash_attention_kernel.use_vmap_bwd = True
  cfg.splash_attention_kernel.sa_use_document_aware_block_skipping = False
  return cfg


def qwen3_8b_maxtext_parity() -> TPUTrainerConfig:
  """Qwen3 8B recipe matching currently available MaxText policies.

  MaxText keeps parameters sharded across its FSDP axis and rematerializes the
  layer body, which is operationally closest to Zero-3. Relative to the tuned
  Zero-3 recipe, this also rotates Q/K and reduces FSDP gradients in BF16.
  Layer scan is intentionally omitted, so this matches numerical and kernel
  policy but not the loop representation.
  """
  cfg = qwen3_8b_zero3()
  cfg.training.mixed_precision_reduce = "bfloat16"
  cfg.override.imports.append(_BF16_ROPE_OVERRIDE)
  return cfg


def qwen3_moe_30b() -> TPUTrainerConfig:
  """TPU specialization of the global-aux Qwen3 30B recipe."""
  cfg = TPUTrainerConfig.derive_from(qwen3_30b_a3b_global_aux())
  cfg.optimizer.implementation = "foreach"
  cfg.loss = CrossEntropyLoss.Config()
  cfg.model_spec = model_registry(
      "30B-A3B-global-aux",
      attn_backend="sdpa",
      converters=[
          TpuTokenDispatcherConverter.Config(),
          TpuGroupedExpertsConverter.Config(),
      ],
  )
  cfg.model_spec.parallelize_fn = tpu_parallelize_qwen3
  cfg.training.local_batch_size = 6
  cfg.training.global_batch_size = 768
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
  cfg.model_spec.model = (
      TpuPackedQKVConverter.Config().build().convert(cfg.model_spec.model)
  )
  for _, experts, _, _ in cfg.model_spec.model.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    experts.use_pallas_paired_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 256
    experts.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)
    experts.use_kernel_native_weight_layout = True
  cfg.training.local_batch_size = 6
  cfg.training.global_batch_size = 768
  cfg.training.seq_len = 2048
  cfg.training.steps = 8
  cfg.tpu_config = TPUConfig(
      use_graph_trainer=True,
      compile_mode="aot_fx_trace",
      graph_trainer_memory_policy="full",
      enable_graph_trainer_remat_schedule=True,
      compile_grad_norm=True,
      fsdp_allreduce_slice_parameter_patterns=(
          "layers.*.attention.qkv_linear.wqkv_packed.weight",
          "layers.*.attention.wo.weight",
      ),
      # The validated QKV and output-projection gradients are 40 and 32 MiB.
      # Larger reductions can lose to native reduce-scatter because all-reduce
      # sends roughly twice the ring traffic and materializes the full result.
      fsdp_allreduce_slice_max_input_bytes=40 * 1024**2,
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
  assert cfg.model_spec is not None
  dispatchers = list(
      cfg.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config)
  )
  if not dispatchers:
    raise ValueError("MaxText-parity Qwen3 MoE requires a TPU token dispatcher")
  for _, dispatcher, _, _ in dispatchers:
    dispatcher.fuse_route_score_backward = True
  return cfg


def _qwen3_moe_235b_fsdp_only() -> TPUTrainerConfig:
  """Build the common Qwen3 235B-A22B FSDP configuration."""
  cfg = qwen3_moe_30b_fsdp_only_full_ac_benchmark()
  # Q30's rewrite is tuned for its full-world FP32 graph. Q235's explicit
  # FSDP-128, HSDP, and BF16 recipes violate that pass's current validator.
  cfg.tpu_config.fsdp_allreduce_slice_parameter_patterns = ()
  cfg.tpu_config.fsdp_allreduce_slice_max_input_bytes = 0
  cfg.model_spec = model_registry(
      "235B-A22B",
      attn_backend="sdpa",
      converters=[
          TpuTokenDispatcherConverter.Config(),
          TpuGroupedExpertsConverter.Config(),
      ],
  )
  cfg.model_spec.parallelize_fn = tpu_parallelize_qwen3
  model_config = cast(Qwen3Model.Config, cfg.model_spec.model)
  # Replacing model_spec also replaces nested MoE policy inherited from the
  # 30B base recipe, so apply the intended Q235 routing policy afterwards.
  _use_global_batch_aux_loss(cfg)
  # The TPU Qwen parallelizer shards the uneven vocabulary and expert banks on
  # their divisible hidden axes when the FSDP mesh is wider than dim 0.
  for _, experts, _, _ in model_config.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 256
    experts.pallas_grouped_mm_tgmm_tiling = (256, 0, 512)
    experts.use_kernel_native_weight_layout = True
  cfg.training.local_batch_size = 16
  cfg.training.global_batch_size = 8192
  cfg.training.seq_len = 4096
  cfg.training.steps = 15
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  cfg.tpu_config.use_graph_split = True
  cfg.tpu_config.graph_split.forward_layers_per_executable = 1
  cfg.tpu_config.graph_split.forward_isolate_prefix = True
  cfg.tpu_config.graph_split.backward_partition_mode = "fsdp_pipeline"
  cfg.tpu_config.graph_split.backward_layers_per_executable = 1
  cfg.tpu_config.enable_graph_trainer_remat_schedule = False
  cfg.tpu_config.graph_trainer_fsdp_schedule = "none"
  cfg.tpu_config.compile_optimizer = False
  cfg.tpu_config.optimizer_in_graph = False
  cfg.hf_assets_path = "./assets/hf/Qwen3-235B-A22B"
  cfg.metrics.log_freq = 1
  return cfg


def _qwen3_moe_235b_fsdp_activation_offload() -> TPUTrainerConfig:
  """Add the common Q235 activation-offload execution policy."""
  cfg = _qwen3_moe_235b_fsdp_only()
  cfg.tpu_config.graph_trainer_memory_policy = "full_and_offload"
  cfg.tpu_config.graph_split.forward_layers_per_executable = 1
  # Current ``by_layer`` is the direct successor of the ``sequential`` mode
  # used by the measured schedule: one complete backward layer per
  # independently compiled child.
  cfg.tpu_config.graph_split.backward_partition_mode = "by_layer"
  cfg.tpu_config.graph_split.backward_layers_per_executable = 1
  # Issue each H2D pair in the child immediately before its consumer. This
  # removed 1.31 s of exposed H2D in the measured BF16 arm at a 0.96-GiB HBM
  # cost from carrying the completed activation across one child boundary.
  cfg.tpu_config.graph_split.cpu_offload_prefetch_executables = 1
  cfg.libtpu_init_flags = tpu_flags.merge_flags(
      cfg.libtpu_init_flags,
      _Q235_COMPILED_OFFLOAD_FLAGS,
  )
  return cfg


def qwen3_moe_235b_fsdp_activation_offload_maxtext() -> TPUTrainerConfig:
  """MaxText-numerics variant of the Q235 activation-offload recipe."""
  cfg = _qwen3_moe_235b_fsdp_activation_offload()
  cfg.training.mixed_precision_reduce = "bfloat16"
  if _BF16_ROPE_OVERRIDE not in cfg.override.imports:
    cfg.override.imports.append(_BF16_ROPE_OVERRIDE)
  assert cfg.model_spec is not None
  dispatchers = list(
      cfg.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config)
  )
  if not dispatchers:
    raise ValueError("MaxText-parity Qwen3 MoE requires a TPU token dispatcher")
  for _, dispatcher, _, _ in dispatchers:
    dispatcher.fuse_route_score_backward = True
  return cfg


def _configure_qwen3_moe_235b_fsdp_128(
    cfg: TPUTrainerConfig,
    *,
    replicate_degree: int,
) -> None:
  """Configure an FSDP-128 mesh and its matching global batch size."""
  cfg.parallelism.data_parallel_replicate_degree = replicate_degree
  cfg.parallelism.data_parallel_shard_degree = 128
  cfg.training.global_batch_size = 2048 * replicate_degree


def qwen3_moe_235b_hsdp_activation_offload_maxtext() -> TPUTrainerConfig:
  """MaxText-numerics Q235 offload recipe using 4x128 HSDP.

  Matching the FSDP shard degree to the 128-expert dimension keeps expert
  weights on their natural shard axis and avoids hidden-axis reconstruction.
  """
  cfg = qwen3_moe_235b_fsdp_activation_offload_maxtext()
  _configure_qwen3_moe_235b_fsdp_128(cfg, replicate_degree=4)
  return cfg


def qwen3_moe_235b_fsdp_activation_offload_4x4x4() -> TPUTrainerConfig:
  """FP32-reduction Q235 offload recipe for a 4x4x4 TPU7x slice.

  Pure FSDP over 128 ranks divides both the 128 experts and the 151,936-token
  vocabulary evenly. Keep local batch 16 and use one microbatch per step.
  """
  cfg = _qwen3_moe_235b_fsdp_activation_offload()
  _configure_qwen3_moe_235b_fsdp_128(cfg, replicate_degree=1)
  return cfg


def qwen3_moe_235b_fsdp_activation_offload_maxtext_4x4x4(
) -> TPUTrainerConfig:
  """MaxText-numerics Q235 offload recipe for a 4x4x4 TPU7x slice."""
  cfg = qwen3_moe_235b_fsdp_activation_offload_maxtext()
  _configure_qwen3_moe_235b_fsdp_128(cfg, replicate_degree=1)
  return cfg


def qwen3_moe_235b_fsdp_smoke() -> TPUTrainerConfig:
  """Four-layer FSDP512 smoke for the production 235B execution path."""
  cfg = _qwen3_moe_235b_fsdp_only()
  assert cfg.model_spec is not None
  model_config = cast(Qwen3Model.Config, cfg.model_spec.model)
  model_config.layers = model_config.layers[:4]
  cfg.training.local_batch_size = 1
  cfg.training.seq_len = 128
  cfg.training.steps = 2
  cfg.tpu_config.enable_mfu_benchmark = False
  return cfg


def qwen3_moe_235b_scan() -> TPUTrainerConfig:
  """Qwen3 235B-A22B on the graph trainer with its 94 layers scanned.

  Shaped after MaxText's published Ironwood recipe
  (`tpu-recipes/training/ironwood/qwen3-235b-a22b/4k-bf16-tpu7x-4x8x8`): 256
  chips, sequence 4096, batch 16 per device, activation offloading on, no
  expert parallelism. Batch and global batch are the two things to override on
  a smaller slice -- `global_batch_size` must equal ranks x `local_batch_size`
  or config validation fails with a message that reads like a scan error.
  """
  cfg = qwen3_moe_30b_fsdp_only_full_ac_benchmark()
  cfg.model_spec = model_registry(
      "235B-A22B",
      attn_backend="sdpa",
      converters=[
          TpuTokenDispatcherConverter.Config(),
          TpuGroupedExpertsConverter.Config(),
      ],
  )
  cfg.model_spec.parallelize_fn = tpu_parallelize_qwen3
  model_config = cast(Qwen3Model.Config, cfg.model_spec.model)
  _use_global_batch_aux_loss(cfg)
  # Clearing the coeff does not stop `MoE.forward` accumulating into a counter
  # nothing reads, and a scan body cannot write to a tensor it closed over.
  # TODO(#391): delete once `MoE.forward` guards that write.
  cfg.override.imports.append(_DISCARDED_TOKEN_COUNTER_OVERRIDE)
  for _, experts, _, _ in model_config.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 256
    experts.pallas_grouped_mm_tgmm_tiling = (256, 0, 512)
    experts.use_kernel_native_weight_layout = True
  cfg.training.local_batch_size = 16
  cfg.training.global_batch_size = 8192
  cfg.training.seq_len = 4096
  cfg.training.steps = 15
  # The 30B chain leaves this at the real `c4`, whose only builder config is
  # `default`; asking for `en` fails before the first step.
  cfg.dataloader = HuggingFaceTextDataLoader.Config(dataset="c4_test")
  # 94 layers of 128 experts do not fit without Zero-3 plus offloading, which
  # is also what MaxText does here (`remat_policy=custom`,
  # `decoder_layer_input=offload`).
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  cfg.tpu_config.use_scan_layers = True
  cfg.tpu_config.graph_trainer_memory_policy = "full_and_scan_offload"
  # The 30B chain coalesces its packed-QKV and `wo` reductions (#419). This
  # model packs no QKV and scan leaves no per-layer FQN to match, so the pass
  # would find none of its targets and refuse to run.
  cfg.tpu_config.fsdp_allreduce_slice_parameter_patterns = ()
  # The routing metrics stash tensors on the module, which a scan body cannot
  # do (https://github.com/google-pytorch/torchtitan/issues/399).
  cfg.metrics.enable_moe_routing = False
  cfg.metrics.log_freq = 1
  # Left to the command line these get forgotten, and a scan step measured
  # without them runs ~10% slow.
  cfg.libtpu_init_flags = tpu_flags.merge_flags(
      cfg.libtpu_init_flags,
      _Q235_SHARED_FLAGS,
  )
  return cfg


def qwen3_moe_235b_scan_proxy() -> TPUTrainerConfig:
  """`qwen3_moe_235b_scan` cut to 12 of its 94 layers, so it fits a 32-chip slice.

  Every layer is a real 235B layer -- dim 4096, 128 experts, top-k 8 -- so the
  scan body is the shape the full model would scan over. Only the trip count
  changes, which is the one thing scan is meant not to care about.
  """
  cfg = qwen3_moe_235b_scan()
  assert cfg.model_spec is not None
  model_config = cast(Qwen3Model.Config, cfg.model_spec.model)
  model_config.layers = model_config.layers[:12]
  cfg.training.local_batch_size = 1
  cfg.training.global_batch_size = 32
  return cfg


def qwen3_moe_235b_scan_maxtext() -> TPUTrainerConfig:
  """`qwen3_moe_235b_scan` with MaxText's numerics instead of torchtitan's.

  Three changes, all of them bf16 where torchtitan keeps f32: the gradient
  reduce, the rotary arithmetic, and the route-score backward. None is
  bit-preserving, so the base recipe stays the accuracy reference and this one
  is the speed comparison against MaxText's published Ironwood number.
  """
  cfg = qwen3_moe_235b_scan()
  cfg.training.mixed_precision_reduce = "bfloat16"
  cfg.override.imports.append(_BF16_ROPE_OVERRIDE)
  assert cfg.model_spec is not None
  dispatchers = list(
      cfg.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config)
  )
  if not dispatchers:
    raise ValueError("MaxText-parity Qwen3 MoE requires a TPU token dispatcher")
  for _, dispatcher, _, _ in dispatchers:
    dispatcher.fuse_route_score_backward = True
  return cfg


def qwen3_moe_30b_proxy() -> TPUTrainerConfig:
  """Proxy model for Qwen3 30B MoE (4 layers instead of 48) for testing/iterations."""
  cfg = qwen3_moe_30b()
  assert cfg.model_spec is not None
  model_config = cast(Qwen3Model.Config, cfg.model_spec.model)
  # Slice layers to only keep 4 layers to save memory and compile time
  model_config.layers = model_config.layers[:4]
  return cfg
