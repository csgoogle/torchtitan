"""Parallelization utilities for Qwen3 models on TPU."""

import dataclasses
import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.models.qwen3 import parallelize as dtensor_parallelize

from torchtitan.experiments.graph_trainer.common_utils import (
    apply_simple_fsdp,
)
from torchtitan.experiments.tpu.grouped_experts import (
    configure_expert_dweight_output_dtype,
)
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.tools.logging import logger


def configure_qwen_simple_fsdp_expert_shard_dims(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
) -> None:
  """Use a divisible hidden axis for uneven Qwen expert banks under pure FSDP."""
  fsdp_degree = parallel_dims.dp_shard * parallel_dims.cp
  if fsdp_degree <= 1 or parallel_dims.ep_enabled:
    return

  modules = [
      (module_fqn, module)
      for module_fqn, module in model.named_modules()
      if module_fqn.endswith(".moe.experts")
  ]

  alternate_modules: list[str] = []
  for module_fqn, module in modules:
    parameters = list(module.named_parameters(recurse=False))
    if not parameters:
      raise ValueError(f"{module_fqn} has no direct parameters to FSDP-shard")
    if all(parameter.shape[0] % fsdp_degree == 0 for _, parameter in parameters):
      continue
    incompatible = [
        f"{name}{tuple(parameter.shape)}"
        for name, parameter in parameters
        if parameter.ndim < 2 or parameter.shape[1] % fsdp_degree != 0
    ]
    if incompatible:
      raise ValueError(
          f"Cannot FSDP-shard {module_fqn} on either dimension 0 or 1 over "
          f"{fsdp_degree} ranks; incompatible parameters: {incompatible}"
      )
    setattr(module, "_fsdp_shard_dim", 1)
    alternate_modules.append(module_fqn)

  if alternate_modules:
    logger.info(
        "Using SimpleFSDP dimension 1 for %d uneven Qwen expert modules",
        len(alternate_modules),
    )


def parallelize_qwen3(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config: "torchtitan.distributed.activation_checkpoint.ActivationCheckpointing.Config",
    dump_folder: str,
    skip_dp: bool = False,
    tpu_compile_mode: str = "layer",
    **kwargs,
) -> nn.Module:
  """Parallelizes the Qwen3 model, optionally applying Simple FSDP.

  This wrapper delegates to the upstream Qwen3 parallelize function. If
  `--parallelism.use_simple_fsdp` is set, it skips upstream Data Parallelism
  and instead applies the TPU-specific Simple FSDP implementation.
  """
  compile_mode = tpu_compile_mode
  effective_compile_config = compile_config
  reduce_dtype = torchtitan.config.TORCH_DTYPE_MAP[
      training.mixed_precision_reduce
  ]
  # Match unpaired expert dWeight stores to the final, post-CLI FSDP reduction
  # dtype before tracing or FSDP wrapping.
  configure_expert_dweight_output_dtype(model, reduce_dtype)
  if effective_compile_config.enable and hasattr(model, "layers"):
    from torchtitan.models.common.moe import MoE

    for layer in model.layers.values():
      has_moe = any(isinstance(m, MoE) for m in layer.modules())
      if not has_moe:
        apply_spmd_safe(layer)

  if compile_mode in ("fullgraph", "aot_fx_trace") and compile_config.enable:
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  if parallelism.use_simple_fsdp:
    # Apply all other parallelisms but skip data parallelism
    dtensor_parallelize.parallelize_qwen3(
          model,  # pyrefly: ignore[bad-argument-type]
          ac_config=ac_config,
          compile_config=effective_compile_config,
          dump_folder=dump_folder,
          parallel_dims=parallel_dims,
          parallelism=parallelism,
          skip_dp=True,
          training=training,
      )

    # Configure expert sharding after upstream parallelization has installed
    # any wrappers, but before SimpleFSDP selects parameter shard dimensions.
    configure_qwen_simple_fsdp_expert_shard_dims(model, parallel_dims)
    model = apply_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=training,
    )
  else:
    dtensor_parallelize.parallelize_qwen3(
          model,  # pyrefly: ignore[bad-argument-type]
          ac_config=ac_config,
          compile_config=effective_compile_config,
          dump_folder=dump_folder,
          parallel_dims=parallel_dims,
          parallelism=parallelism,
          skip_dp=skip_dp,
          training=training,
      )
  if compile_mode == "fullgraph":
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")
  return model
