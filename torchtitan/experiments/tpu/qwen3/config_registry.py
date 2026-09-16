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
from torchtitan.models.qwen3.config_registry import qwen3_30b_a3b_global_aux
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.profiler import Profiler


_FUSED_LOSS_OVERRIDE = "torchtitan.experiments.tpu.overrides.fused_loss"
_BF16_ROPE_OVERRIDE = "torchtitan.experiments.tpu.overrides.bf16_rope"


# Compiler policy shared by the FP32 and MaxText-parity offload recipes.
_Q235_COMPILED_OFFLOAD_FLAGS = tpu_flags.merge_flags(
    # Aggregate SparseCore collectives and enable layer-level latency hiding so
    # FSDP communication can be scheduled alongside independent layer compute.
    tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
    # Use both SparseCores for reduce-scatter in the by-layer FSDP schedule.
    tpu_flags.DUAL_SPARSECORE_REDUCE_SCATTER,
    {
        # These three fusion disables are the validated workaround for a
        # deterministic libtpu crash while compiling Q235 layer executables.
        "xla_tpu_dot_dot_fusion": False,
        # Disable nested-dot fusion as the second part of that workaround.
        "xla_tpu_nested_dot_fusion": False,
        # Disable RWB fusion as the third part of that workaround.
        "xla_tpu_rwb_fusion": False,
        # Keep activation host copies off SparseCore, where they would contend
        # with the FSDP collectives above. This does not itself create overlap.
        "xla_tpu_enable_offloading_copy_to_sparsecore": False,
        # Prove the true copy/overwrite ordering across donated aliases. This
        # removes the defensive activation snapshot that otherwise blocks D2H.
        "xla_tpu_copy_insertion_use_region_analysis": True,
        # Let region analysis inspect the large Q235 layer regions instead of
        # declining because its search budget is too small.
        "xla_tpu_copy_insertion_use_region_analysis_limit": 1000,
    },
)

# Additional MaxText host-offload scheduler policy. Flags marked parity-only
# reproduce that policy but are not individually credited with this recipe's
# ordinary-copy overlap; those transfers use generic kCopy and leave
# xla_max_concurrent_host_copy at its default.
_Q235_COMPILED_OFFLOAD_MAXTEXT_FLAGS = tpu_flags.merge_flags(
    _Q235_COMPILED_OFFLOAD_FLAGS,
    {
        # Allow LHS to use the full shared-memory budget when evaluating
        # overlap schedules; the normal compiler default is 95%.
        "xla_tpu_scheduler_percent_shared_memory_limit": 100,
        # Disable libtpu's distributed diagnostic coordinator for the measured
        # 512-rank launch; this is operational and does not change model math.
        "xla_tpu_enable_diagnostic_coordinator": False,
        # Print the failed allocation and available/reserved bytes if this
        # near-capacity recipe OOMs; observability only.
        "tpu_log_allocations_on_oom": True,
        # Execute eligible FSDP all-gathers on SparseCore so TensorCore work can
        # overlap them.
        "xla_tpu_enable_sparse_core_collective_offload_all_gather": True,
        # Execute eligible FSDP all-reduces on SparseCore for the same reason.
        "xla_tpu_enable_sparse_core_collective_offload_all_reduce": True,
        # Select the one-SparseCore implementation for each offloaded
        # all-gather, matching the measured collective schedule.
        "xla_tpu_use_single_sparse_core_for_all_gather_offload": True,
        # Enable MaxText's umbrella set of experimental scheduler heuristics;
        # its individual contribution was not isolated in this recipe.
        "xla_tpu_enable_all_experimental_scheduler_features": True,
        # Include live-buffer pressure in scheduler choices so overlap does not
        # ignore this batch-16 recipe's tight HBM margin.
        "xla_tpu_enable_scheduler_memory_pressure_tracking": True,
        # Run the host-transfer pipelining path for eligible sliced/loop-carried
        # transfers. Ordinary split-child copies are asyncified separately.
        "xla_tpu_pipeline_memory_host_transfers": True,
        # Cap concurrent TPU dynamic-slice/update host transfers at 24. This
        # does not govern the ordinary kCopy transfers used by this recipe.
        "xla_tpu_host_transfer_overlap_limit": 24,
        # Remove optimization barriers aggressively before scheduling so they
        # do not fence otherwise independent asynchronous work.
        "xla_tpu_aggressive_opt_barrier_removal": "ENABLED",
        # Prefer maintaining asynchronous pipeline depth over minimizing the
        # scheduler's immediate stall estimate.
        "xla_lhs_prioritize_async_depth_over_stall": "ENABLED",
        # Permit backward all-gather pipelining for HLO shapes where the
        # collective pipeliner finds a legal candidate.
        "xla_tpu_enable_ag_backward_pipelining": True,
        # Let the collective pipeliner follow loop-variant parameters when
        # building a candidate chain; parity-only for split entry computations.
        "xla_should_allow_loop_variant_parameter_in_chain": "ENABLED",
        # Let that chain also include loop-invariant operands; likewise only
        # relevant if an eligible while/scan is present.
        "xla_should_add_loop_invariant_op_in_chain": "ENABLED",
        # Raise host send/recv concurrency for MaxText parity. It does not
        # control the generic kCopy activation transfers.
        "xla_max_concurrent_host_send_recv": 100,
        # Rerun latency-hiding scheduling after SparseCore assignment so the
        # final scheduler sees the selected collective resources.
        "xla_latency_hiding_scheduler_rerun": 2,
    },
)


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Select the fused loss and its pre-construction LM-head override together."""
  config.loss = PallasCrossEntropyLoss.Config()
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)


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


def qwen3_moe_235b_fsdp_only() -> TPUTrainerConfig:
  """Qwen3 235B-A22B FSDP recipe for a 4x8x8 TPU7x slice."""
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
  # The tokenizer/model vocabulary has 151,936 real classes, which is not
  # divisible by FSDP-512. Pad the physical embedding and LM-head matrices to
  # 152,064 (= 297 rows/rank), while keeping vocab_size as the logical size.
  # Token IDs never address the padded rows, and the loss/logit paths below
  # remove those rows before softmax so they cannot change model semantics.
  model_config.tok_embeddings.num_embeddings = 152064
  model_config.lm_head.out_features = 152064
  assert isinstance(cfg.loss, PallasCrossEntropyLoss.Config)
  cfg.loss.logical_vocab_size = model_config.vocab_size
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


def qwen3_moe_235b_fsdp_compiled_offload() -> TPUTrainerConfig:
  """FP32-reduction Q235 recipe with compiled activation CPU offload.

  This is the landing-policy counterpart of the BF16 MaxText-parity recipe
  below. It keeps one layer in each split executable and reuses the prior
  step's pinned-host activation buffers.
  """
  cfg = qwen3_moe_235b_fsdp_only()
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


def qwen3_moe_235b_fsdp_compiled_offload_maxtext() -> TPUTrainerConfig:
  """MaxText-numerics variant of the Q235 compiled-offload recipe."""
  cfg = qwen3_moe_235b_fsdp_compiled_offload()
  cfg.libtpu_init_flags = tpu_flags.merge_flags(
      cfg.libtpu_init_flags,
      _Q235_COMPILED_OFFLOAD_MAXTEXT_FLAGS,
  )
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


def qwen3_moe_235b_fsdp_smoke() -> TPUTrainerConfig:
  """Four-layer FSDP512 smoke for the production 235B execution path."""
  cfg = qwen3_moe_235b_fsdp_only()
  assert cfg.model_spec is not None
  model_config = cast(Qwen3Model.Config, cfg.model_spec.model)
  model_config.layers = model_config.layers[:4]
  cfg.training.local_batch_size = 1
  cfg.training.seq_len = 128
  cfg.training.steps = 2
  cfg.tpu_config.enable_mfu_benchmark = False
  return cfg


def qwen3_moe_235b_fsdp_pipeline_b8_profile() -> TPUTrainerConfig:
  """Batch-8 validation and rank-zero profile for the 235B pipeline."""
  cfg = qwen3_moe_235b_fsdp_only()
  cfg.training.local_batch_size = 8
  cfg.training.global_batch_size = 4096
  cfg.training.steps = 15
  cfg.profiler = Profiler.Config(
      enable_profiling=int(os.environ.get("RANK", "0")) == 0,
      profile_freq=5,
      profiler_warmup=2,
      profiler_active=1,
      profiler_repeat=1,
  )
  return cfg


def qwen3_moe_30b_proxy() -> TPUTrainerConfig:
  """Proxy model for Qwen3 30B MoE (4 layers instead of 48) for testing/iterations."""
  cfg = qwen3_moe_30b()
  assert cfg.model_spec is not None
  model_config = cast(Qwen3Model.Config, cfg.model_spec.model)
  # Slice layers to only keep 4 layers to save memory and compile time
  model_config.layers = model_config.layers[:4]
  return cfg
