# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

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
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.tpu.examples.gemma4 import (
    _build_flavor_config,
    model_registry,
)
from torchtitan.experiments.tpu.examples.gemma4.infra.parallelize import (
    parallelize_gemma4 as tpu_parallelize_gemma4,
)
from torchtitan.experiments.tpu.examples.gemma4.model import Gemma4Model
from torchtitan.experiments.tpu.examples.gemma4.state_dict_adapter import (
    Gemma4StateDictAdapter,
)
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
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
from torchtitan.protocols.model_spec import ModelSpec


def gemma4_debugmodel() -> TPUTrainerConfig:
  """Gemma4 dense debug model configuration for TPU."""
  model_spec = model_registry(
      "debugmodel", attn_backend="sdpa", enable_weight_tying=False
  )
  model_spec.parallelize_fn = tpu_parallelize_gemma4
  return TPUTrainerConfig(
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
          local_batch_size=4,
          seq_len=512,
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


def gemma4_moe_testmodel() -> TPUTrainerConfig:
  """Gemma4 MoE test model configuration for TPU."""
  cfg = gemma4_debugmodel()
  cfg.model_spec = ModelSpec(
      name="gemma4",
      flavor="testmodel_moe",
      model=_build_flavor_config("testmodel_moe", enable_weight_tying=False),
      parallelize_fn=tpu_parallelize_gemma4,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=Gemma4StateDictAdapter,
  )
  cfg.training.seq_len = 128
  cfg.training.local_batch_size = 4
  return cfg


def gemma4_moe_debug() -> TPUTrainerConfig:
  """Gemma4 MoE debug model configuration for TPU."""
  cfg = gemma4_debugmodel()
  cfg.model_spec = model_registry(
      "debugmodel_moe",
      attn_backend="sdpa",
      converters=[
          TpuTokenDispatcherConverter.Config(),
          TpuGroupedExpertsConverter.Config(),
      ],
      enable_weight_tying=False,
  )
  return cfg


def gemma4_31b() -> TPUTrainerConfig:
  """Gemma4 31B dense model configuration for TPU."""
  model_spec = model_registry(
      "31B", attn_backend="sdpa", enable_weight_tying=False
  )
  model_spec.parallelize_fn = tpu_parallelize_gemma4

  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./assets/hf/gemma-4-31B",
      model_spec=model_spec,
      optimizer=default_adamw(lr=3e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=200,
          decay_ratio=0.8,
          decay_type="cosine",
          min_lr_factor=0.1,
      ),
      training=TrainingConfig(
          max_norm=1.0,
          local_batch_size=2,
          seq_len=4096,
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
          data_parallel_shard_degree=-1,
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


def gemma4_26b_a4b() -> TPUTrainerConfig:
  """Gemma4 26B-A4B MoE model configuration for TPU."""
  cfg = gemma4_31b()
  cfg.hf_assets_path = "./assets/hf/gemma-4-26B-A4B"
  cfg.model_spec = model_registry(
      "26B-A4B",
      attn_backend="sdpa",
      converters=[
          TpuTokenDispatcherConverter.Config(),
          TpuGroupedExpertsConverter.Config(),
      ],
      enable_weight_tying=False,
  )
  model_config = cast(Gemma4Model.Config, cfg.model_spec.model)
  for _, experts, _, _ in model_config.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    experts.use_pallas_paired_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 256
    experts.pallas_grouped_mm_tgmm_tiling = (256, 0, 0)
    experts.use_kernel_native_weight_layout = True
  return cfg
