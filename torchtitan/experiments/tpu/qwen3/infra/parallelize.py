"""Parallelization utilities for Qwen3 models on TPU."""

import dataclasses
import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.models.qwen3 import parallelize as dtensor_parallelize


from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    MixedPrecisionPolicy as SimpleFSDPMixedPrecisionPolicy,
    data_parallel as simple_fsdp_data_parallel,
)
from torchtitan.experiments.tpu.grouped_experts import (
    configure_expert_dweight_output_dtype,
)
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.tools.logging import logger

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

    # Mesh names changed upstream ('fsdp' -> 'dp_shard' x 'cp'); use the
    # graph trainer's helper, which resolves them and the MoE expert meshes.
    model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)
    logger.info("Applied SimpleFSDP to the model")
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
