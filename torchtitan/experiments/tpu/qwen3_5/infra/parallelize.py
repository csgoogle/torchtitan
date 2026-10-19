"""Parallelization utilities for Qwen3.5 models on TPU."""

import dataclasses

import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import configure_simple_fsdp_shard_dims
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts

# Installs the `attn_gym` stub before `torchtitan.models.qwen3_5` is imported.
from torchtitan.experiments.tpu.qwen3_5 import gated_delta  # pylint: disable=unused-import
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.models.common.moe import MoE
from torchtitan.models.qwen3_5 import parallelize as dtensor_parallelize
from torchtitan.tools.logging import logger


def configure_qwen3_5_simple_fsdp_shard_dims(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
) -> list[str]:
  """Use divisible hidden axes for uneven Qwen3.5 parameters under SimpleFSDP."""
  fsdp_degree = parallel_dims.dp_shard * parallel_dims.cp
  if fsdp_degree <= 1:
    return []

  hidden_axis_modules = configure_simple_fsdp_shard_dims(model, parallel_dims)
  stacks = any(hasattr(m, "plan_shard_dims") for m in model.modules())

  if not parallel_dims.ep_enabled:
    expert_modules = [
        (module_fqn, module)
        for module_fqn, module in model.named_modules()
        if isinstance(module, TpuGroupedExperts)
    ]
    for module_fqn, module in expert_modules:
      if module.num_experts % fsdp_degree == 0:
        continue

      parameters = list(module.named_parameters(recurse=False))
      if not parameters:
        if stacks:
          continue
        raise ValueError(f"{module_fqn} has no direct parameters to FSDP-shard")
      incompatible = [
          f"{name}{tuple(parameter.shape)}"
          for name, parameter in parameters
          if (
              parameter.ndim != 3
              or parameter.shape[0] != module.num_experts
              or parameter.shape[1] % fsdp_degree != 0
          )
      ]
      if incompatible:
        raise ValueError(
            f"Cannot FSDP-shard {module_fqn} on Qwen3.5's hidden dimension 1 "
            f"over {fsdp_degree} ranks; incompatible expert parameters: "
            f"{incompatible}"
        )
      setattr(module, "_fsdp_shard_dim", 1)
      hidden_axis_modules.append(module_fqn)

  if hidden_axis_modules:
    logger.info(
        "Using SimpleFSDP dimension 1 for %d uneven Qwen3.5 modules: %s",
        len(hidden_axis_modules),
        hidden_axis_modules,
    )
  return hidden_axis_modules


def _validate_supported(
    model: nn.Module, parallel_dims: torchtitan.distributed.ParallelDims
) -> None:
  if getattr(model, "vision_encoder", None) is not None:
    raise NotImplementedError(
        "Multimodal Qwen3.5 flavors are not supported on TPU yet; use a "
        "text-only flavor."
    )
  if parallel_dims.cp_enabled:
    raise NotImplementedError(
        "Context parallelism is not supported for Qwen3.5 on TPU yet."
    )
  if parallel_dims.pp_enabled:
    raise NotImplementedError(
        "Pipeline parallelism is not supported for Qwen3.5 on TPU yet."
    )


def parallelize_qwen3_5(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config: "torchtitan.distributed.activation_checkpoint.ActivationCheckpointingConfig | None",
    dump_folder: str,
    skip_dp: bool = False,
    tpu_compile_mode: str = "layer",
    **kwargs,
) -> nn.Module:
  """Parallelizes a text-only Qwen3.5 model on TPU, optionally using SimpleFSDP."""
  del kwargs  # Unused.
  _validate_supported(model, parallel_dims)

  compile_mode = tpu_compile_mode
  effective_compile_config = compile_config
  if effective_compile_config.enable and hasattr(model, "layers"):
    for layer in model.layers.values():
      has_moe = any(isinstance(m, MoE) for m in layer.modules())
      if not has_moe:
        apply_spmd_safe(layer)

  if compile_mode in ("fullgraph", "aot_fx_trace") and compile_config.enable:
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  if parallelism.use_simple_fsdp:
    dtensor_parallelize.parallelize_qwen3_5(
        model,  # pyrefly: ignore[bad-argument-type]
        ac_config=ac_config,  # pyrefly: ignore[bad-argument-type]
        compile_config=effective_compile_config,
        dump_folder=dump_folder,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        skip_dp=True,
        training=training,
    )
    configure_qwen3_5_simple_fsdp_shard_dims(model, parallel_dims)
    model = apply_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=training,
    )
    for holder in model.modules():
      if hasattr(holder, "build_split_meshes"):
        holder.build_split_meshes()
  else:
    dtensor_parallelize.parallelize_qwen3_5(
        model,  # pyrefly: ignore[bad-argument-type]
        ac_config=ac_config,  # pyrefly: ignore[bad-argument-type]
        compile_config=effective_compile_config,
        dump_folder=dump_folder,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        skip_dp=skip_dp,
        training=training,
    )

  if compile_mode == "fullgraph" and compile_config.enable:
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")
  return model
