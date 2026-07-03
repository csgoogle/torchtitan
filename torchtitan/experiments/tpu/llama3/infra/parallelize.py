"""Parallelization utilities for Llama models on TPU.

This module is used in distributed training tests configured with
`model_name="llama3_tpu".
"""

from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.models.llama3 import parallelize as dtensor_parallelize
from torchtitan.tools.logging import logger
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    MixedPrecisionPolicy as SimpleFSDPMixedPrecisionPolicy,
    data_parallel as simple_fsdp_data_parallel,
)


def parallelize_llama(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config: torchtitan.config.ActivationCheckpointConfig,
    dump_folder: str,
    skip_dp: bool = False,
) -> nn.Module:
  """Parallelizes the Llama model based on the provided configuration and applies TPU-specific workarounds.

  Args:
    model: The nn.Module to be parallelized.
    parallel_dims: Contains the world mesh and other parallelization dimensions.
    training: Training configuration.
    parallelism: Parallelism configuration.
    compile_config: Compile configuration.
    ac_config: Activation checkpointing configuration.
    dump_folder: Folder for dumped artifacts.
    skip_dp: Whether to skip data parallelism wrapping.
  """
  # TODO b/499075866: re-enable CPU offload for TPU torchtitan run.
  if training.enable_cpu_offload:
    logger.warning(
        "CPU offload is not supported for TPU torchtitan run, setting it to False."
    )
    training.enable_cpu_offload = False

  if parallelism.use_simple_fsdp:
      # Apply all other parallelisms but skip data parallelism
      dtensor_parallelize.parallelize_llama(
          model,  # pyrefly: ignore[bad-argument-type]
          ac_config=ac_config,
          compile_config=compile_config,
          dump_folder=dump_folder,
          parallel_dims=parallel_dims,
          parallelism=parallelism,
          training=training,
          skip_dp=True,
      )

      dp_mesh_names = (
          ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
      )
      dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

      dp_mode = "hybrid_shard" if parallel_dims.dp_replicate_enabled else "fully_shard"
      mp_policy = SimpleFSDPMixedPrecisionPolicy(
          param_dtype=torchtitan.config.TORCH_DTYPE_MAP[training.mixed_precision_param],
          reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[training.mixed_precision_reduce],
      )
      model = simple_fsdp_data_parallel(
          model,
          dp_mesh,
          mode=dp_mode,
          mp_policy=mp_policy,
      )
      logger.info(f"Applied SimpleFSDP ({dp_mode}) to the model")
  else:
      dtensor_parallelize.parallelize_llama(
          model,  # pyrefly: ignore[bad-argument-type]
          ac_config=ac_config,
          compile_config=compile_config,
          dump_folder=dump_folder,
          parallel_dims=parallel_dims,
          parallelism=parallelism,
          training=training,
          skip_dp=skip_dp,
      )
  return model
