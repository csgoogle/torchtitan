"""TPU GraphTrainer integration for Qwen3.5 (text-only Dense and MoE)."""

import dataclasses

import torch.nn as nn
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    annotate_moe_ep_regions,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.tpu import qwen3_5 as tpu_qwen3_5
from torchtitan.experiments.tpu.qwen3_5 import gated_delta
from torchtitan.experiments.tpu.qwen3_5.infra.parallelize import (
    _validate_supported,
    apply_qwen3_5_simple_fsdp,
    configure_qwen3_5_simple_fsdp_shard_dims,
)
from torchtitan.models.common.moe import configure_moe_aux_loss_reduction
from torchtitan.protocols.model_spec import ModelSpec


def parallelize_qwen3_5(
    model: gated_delta.TpuQwen35Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> nn.Module:
  """Apply FQN annotation, SimpleFSDP, and GraphTrainer compilation to Qwen3.5."""
  del ac_config
  _validate_supported(model, parallel_dims)

  if any(layer.moe is not None for layer in model.config.layers):
    annotate_moe_ep_regions()
  annotate_module_fqns(model)

  if (
      parallelism.spmd_backend == "spmd_types"
      or parallel_dims.tp_enabled
      or parallel_dims.ep_enabled
  ):
    model.parallelize(parallel_dims)

  configure_moe_aux_loss_reduction(model, parallel_dims)
  configure_qwen3_5_simple_fsdp_shard_dims(model, parallel_dims)
  model = apply_qwen3_5_simple_fsdp(
      model, parallel_dims=parallel_dims, training=training
  )  # pyrefly: ignore[bad-assignment]
  for holder in model.modules():
    if hasattr(holder, "build_split_meshes"):
      holder.build_split_meshes()

  model = apply_compile(
      model,
      compile_config=compile_config,
      parallelism=parallelism,
      parallel_dims=parallel_dims,
      dump_folder=dump_folder,
  )
  return model


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    moe_comm_backend: str | None = None,
    seq_len: int | None = None,
    converters: list | None = None,
) -> ModelSpec:
  """GraphTrainer model registry for TPU text-only Qwen3.5 flavors."""
  model_spec = tpu_qwen3_5.model_registry(
      flavor,
      attn_backend=attn_backend,
      moe_comm_backend=moe_comm_backend,
      seq_len=seq_len,
      converters=converters,
  )
  return dataclasses.replace(
      model_spec,
      parallelize_fn=parallelize_qwen3_5,
      pipelining_fn=None,
  )
