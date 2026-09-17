"""Parallelization utilities for Qwen3.5 models on TPU."""

import dataclasses

import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    MixedPrecisionPolicy as SimpleFSDPMixedPrecisionPolicy,
    data_parallel as simple_fsdp_data_parallel,
)
from torchtitan.experiments.tpu.grouped_experts import (
    configure_expert_dweight_output_dtype,
)
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.models.qwen3_5 import parallelize as dtensor_parallelize
from torchtitan.tools.logging import logger


def parallelize_qwen3_5(
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
  """Parallelizes the Qwen3.5 model, optionally applying Simple FSDP.

  This wrapper delegates to the upstream Qwen3.5 parallelize function. If
  `--parallelism.use_simple_fsdp` is set, it skips upstream Data Parallelism
  and instead applies the TPU-specific Simple FSDP implementation.

  Args:
    model: Model to parallelize.
    parallel_dims: Resolved parallel dimensions.
    training: Training config, read for the mixed-precision dtypes.
    parallelism: Parallelism config.
    compile_config: torch.compile config.
    ac_config: Activation checkpointing config.
    dump_folder: Directory for compiler and AC dumps.
    skip_dp: Skip data parallelism entirely. Ignored when SimpleFSDP is on.
    tpu_compile_mode: TPU compilation strategy.
    **kwargs: Ignored; present for signature compatibility.

  Returns:
    The parallelized model.
  """
  del kwargs

  reduce_dtype = torchtitan.config.TORCH_DTYPE_MAP[
      training.mixed_precision_reduce
  ]
  # Match unpaired expert dWeight stores to the final, post-CLI FSDP reduction
  # dtype before tracing or FSDP wrapping.
  configure_expert_dweight_output_dtype(model, reduce_dtype)

  effective_compile_config = compile_config
  if (
      tpu_compile_mode in ("fullgraph", "aot_fx_trace")
      and compile_config.enable
  ):
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  if effective_compile_config.enable and hasattr(model, "layers"):
    from torchtitan.models.common.moe import MoE  # pylint: disable=g-import-not-at-top

    for layer in model.layers.values():
      if not any(isinstance(m, MoE) for m in layer.modules()):
        apply_spmd_safe(layer)

  if parallelism.use_simple_fsdp:
    dtensor_parallelize.parallelize_qwen3_5(
        model,  # pyrefly: ignore[bad-argument-type]
        ac_config=ac_config,
        compile_config=effective_compile_config,
        dump_folder=dump_folder,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        skip_dp=True,
        training=training,
    )
    dp_mesh_names = (
        ["dp_replicate", "fsdp"]
        if parallel_dims.dp_replicate_enabled
        else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
    dp_mode = (
        "hybrid_shard" if parallel_dims.dp_replicate_enabled else "fully_shard"
    )
    mp_policy = SimpleFSDPMixedPrecisionPolicy(
        param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            training.mixed_precision_param
        ],
        reduce_dtype=reduce_dtype,
    )
    model = simple_fsdp_data_parallel(
        model,
        dp_mesh,
        mode=dp_mode,
        mp_policy=mp_policy,
    )
    logger.info(f"Applied SimpleFSDP ({dp_mode}) to the model")
  else:
    dtensor_parallelize.parallelize_qwen3_5(
        model,  # pyrefly: ignore[bad-argument-type]
        ac_config=ac_config,
        compile_config=effective_compile_config,
        dump_folder=dump_folder,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        skip_dp=skip_dp,
        training=training,
    )

  if tpu_compile_mode == "fullgraph":
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")
  return model
