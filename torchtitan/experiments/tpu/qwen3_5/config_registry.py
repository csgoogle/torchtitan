# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer configs for Qwen3.5 on TPU."""

from functools import partial
from typing import cast

from torch import nn
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.qwen3_5 import model_registry
from torchtitan.experiments.tpu.qwen3_5.infra.parallelize import (
    parallelize_qwen3_5 as tpu_parallelize_qwen3_5,
)
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuTokenDispatcherConverter,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    LossKernelConfig,
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.qwen3_5.model import Qwen35Model

# Devices in the `tpu_2x2x1_megachip` (v5p) CI slice. `dp_shard` resolves to
# this, and `global_batch_size` must equal `local_batch_size * dp_shard`.
_FORGE_V5P_DP_SHARD = 4


def _qwen35_tpu_base(
    flavor: str,
    *,
    local_batch_size: int,
    seq_len: int = 2048,
    steps: int = 12,
    use_tpu_converters: bool = True,
) -> TPUTrainerConfig:
  """Shared TPU trainer config for the Qwen3.5 flavors defined in this package."""
  converters = (
      [
          TpuTokenDispatcherConverter.Config(),
          TpuGroupedExpertsConverter.Config(),
      ]
      if use_tpu_converters
      else None
  )
  model_spec = model_registry(
      flavor, attn_backend="sdpa", converters=converters
  )
  model_spec.parallelize_fn = tpu_parallelize_qwen3_5

  model_config = cast(Qwen35Model.Config, model_spec.model)
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
          max_norm=1.0,
          local_batch_size=local_batch_size,
          global_batch_size=local_batch_size * _FORGE_V5P_DP_SHARD,
          seq_len=seq_len,
          steps=steps,
          mixed_precision_reduce="bfloat16",
      ),
      dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          enable_tpu_topo_mesh=False,
          data_parallel_shard_degree=-1,
          expert_parallel_degree=1,
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=False,
      ),
      activation_checkpoint=FullAC.Config(),
      tpu_config=TPUConfig(
          use_graph_trainer=False,
          compile_mode="layer",
      ),
      compile=CompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=False,
      ),
      loss_kernel=LossKernelConfig(use_loss_kernel=False),
      validator=Validator.Config(enable=False),
  )


def qwen35_moe_35b() -> TPUTrainerConfig:
  """Full 40-layer Qwen3.5-35B-A3B configuration for multi-slice TPU training (128 chips).

  Uses the full 40-layer `35B-A3B` model spec (`TpuGraphTrainer`,
  `aot_fx_trace`,
  `optimizer.implementation="fused"`, `fsdp_reshard_after_forward="never"`,
  `enable_graph_trainer_remat_schedule=True`, `compile_grad_norm=True`, and
  `TpuTokenDispatcherConverter` + `TpuGroupedExpertsConverter`) sized for a
  128-chip slice (`local_batch_size=6`, `global_batch_size=768`), matching
  `qwen3_moe_30b`.

  Returns:
    A trainer config for a 128-chip TPU slice.
  """
  cfg = _qwen35_tpu_base("35B-A3B", local_batch_size=6)
  cfg.training.global_batch_size = 768
  cfg.optimizer.implementation = "fused"
  cfg.parallelism.enable_tpu_topo_mesh = True
  cfg.parallelism.fsdp_reshard_after_forward = "never"
  cfg.tpu_config = TPUConfig(
      use_graph_trainer=True,
      compile_mode="aot_fx_trace",
      graph_trainer_memory_policy="full",
      enable_graph_trainer_remat_schedule=True,
      compile_grad_norm=True,
  )
  return cfg


def qwen35_moe_35b_proxy() -> TPUTrainerConfig:
  """Proxy model for Qwen3.5 35B MoE (4 layers instead of 40) for multi-slice testing."""
  cfg = qwen35_moe_35b()
  assert cfg.model_spec is not None
  model_config = cast(Qwen35Model.Config, cfg.model_spec.model)
  model_config.layers = model_config.layers[:4]
  return cfg


def qwen35_moe_35b_fsdp_only_full_ac_benchmark() -> TPUTrainerConfig:
  """Profiler-free 30-step MFU benchmark for 40-layer Qwen3.5-35B-A3B on 128 chips."""
  cfg = qwen35_moe_35b()
  cfg.training.steps = 30
  cfg.training.mixed_precision_reduce = "float32"
  cfg.metrics.log_freq = 5
  cfg.tpu_config.enable_mfu_benchmark = True
  cfg.tpu_config.mfu_benchmark_start_step = 10
  cfg.tpu_config.fsdp_allreduce_slice_parameter_patterns = (
      "layers.*.attn.wo.weight",
  )
  cfg.tpu_config.fsdp_allreduce_slice_max_input_bytes = 40 * 1024**2
  cfg.splash_attention_kernel = SplashAttentionKernelConfig(
      use_splash_attention_kernel=True,
      use_vmap_bwd=True,
      sa_use_document_aware_block_skipping=False,
      sa_q_layout="SEQ_MINOR",
      sa_k_layout="SEQ_MINOR",
      sa_block_q=1024,
      sa_block_kv=1024,
      sa_block_dkv=1024,
      sa_block_q_dkv=1024,
      sa_block_kv_dkv=1024,
      sa_block_q_dq=1024,
      sa_block_kv_dq=1024,
  )
  assert cfg.model_spec is not None
  for _, experts, _, _ in cfg.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.use_pallas_grouped_mm = True
    experts.use_pallas_paired_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 256
    experts.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)
    experts.use_kernel_native_weight_layout = True
  cfg.loss = PallasCrossEntropyLoss.Config()
  cfg.override.imports.append("torchtitan.experiments.tpu.overrides.fused_loss")
  return cfg


# -----------------------------------------------------------------------------
# CI (`bazel test`) variants of the 35B-A3B MoE recipe.
#
# CI schedules single-host slices only, so the largest pool reachable from a
# test is 4 v5p chips (`tpu_2x2x1_megachip`, 95GiB/chip). The production recipe
# uses 128 chips. Every config below therefore slices depth to 4 layers and
# rescales `global_batch_size = local_batch_size * 4`.
#
# Repro command (4x TPU v5p `tpu_2x2x1_megachip` on CI):
#   bazel test -c opt \
#     --norun_validations --nocache_test_results --test_output=errors \
#     --test_timeout=3600 \
#     //torchtitan/experiments/tpu:train_tpu_forge_tpu_tpu_2x2x1_megachip \
#     --test_arg=--nproc_per_node=4 \
#     --test_arg=--module=torchtitan.experiments.tpu.qwen3_5 \
#     --test_arg=--config=qwen35_moe_35b_4layer_graph_forge_b12 \
#     --test_arg=--hf_assets_path=tests/assets/tokenizer \
#     --test_arg=--dataloader.dataset=c4_test \
#     --test_arg=--dataloader.dataset_path=tests/assets/c4_test
# -----------------------------------------------------------------------------


def qwen35_moe_35b_4layer_forge() -> TPUTrainerConfig:
  """Qwen3.5-35B-A3B at 4 of 40 layers on a 4-chip v5p CI slice.

  The architecture is the real 35B-A3B: dim 2048, 256 routed experts at hidden
  dim 512 with top-k 8 plus a shared expert, 16 query heads over 2 KV heads at
  head_dim 256 with partial RoPE over the first 64 dims, and the hybrid decoder
  (three GatedDeltaNet layers then one full-attention layer, matching the
  model's 75/25 split). Only depth, the vision tower and the linear-attention
  kernel differ from upstream; see this package's docstring for why.

  Depth is the concession that makes it fit. The full 40-layer model is roughly
  35B parameters, which at 12 bytes per parameter for bf16 weights, bf16
  gradients and fp32 Adam moments is ~390GiB of state against 380GiB of HBM
  across four v5p chips, before any activation.

  Returns:
    A trainer config for the 4-chip v5p CI slice.
  """
  return _qwen35_tpu_base("35B-A3B-4layer", local_batch_size=1)


def qwen35_moe_35b_4layer_forge_b6() -> TPUTrainerConfig:
  """`qwen35_moe_35b_4layer_forge()` at per-device batch 6.

  Batch 1 leaves the MXUs badly underfed; on the Qwen3 MoE equivalent, moving
  to the recipe's per-device batch of 6 was worth roughly 4x MFU once the
  expert kernels were in place.

  Returns:
    A trainer config for the 4-chip v5p CI slice.
  """
  return _qwen35_tpu_base("35B-A3B-4layer", local_batch_size=6)


def qwen35_moe_35b_4layer_graph_forge_b6() -> TPUTrainerConfig:
  """`qwen35_moe_35b_4layer_forge_b6()` with TpuGraphTrainer and full trainer tuning.

  Mirrors `qwen3_moe_30b_forge_best()`: uses `use_graph_trainer=True`,
  `compile_mode="aot_fx_trace"`, `graph_trainer_memory_policy="full"`,
  `enable_graph_trainer_remat_schedule=True`, `compile_grad_norm=True`,
  `optimizer.implementation="fused"`, and `fsdp_reshard_after_forward="never"`
  (Zero-2 resharding).

  Returns:
    A trainer config for the 4-chip v5p CI slice.
  """
  cfg = _qwen35_tpu_base("35B-A3B-4layer", local_batch_size=6)
  cfg.optimizer.implementation = "fused"
  cfg.parallelism.fsdp_reshard_after_forward = "never"
  cfg.tpu_config = TPUConfig(
      use_graph_trainer=True,
      compile_mode="aot_fx_trace",
      graph_trainer_memory_policy="full",
      enable_graph_trainer_remat_schedule=True,
      compile_grad_norm=True,
  )
  return cfg


def qwen35_moe_35b_4layer_graph_forge_b12() -> TPUTrainerConfig:
  """`qwen35_moe_35b_4layer_graph_forge_b6()` at per-device batch 12."""
  cfg = qwen35_moe_35b_4layer_graph_forge_b6()
  cfg.training.local_batch_size = 12
  cfg.training.global_batch_size = 48
  return cfg
