"""Parallelization utilities for Qwen3 models on TPU.

This module is used in distributed training tests configured with
`model_name="qwen3_tpu"`.
"""

from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.models.qwen3 import parallelize as dtensor_parallelize


def parallelize_qwen3(
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
  """Parallelizes the Qwen3 model based on the job configuration and applies TPU-specific workarounds.

  Args:
    model: The nn.Module to be parallelized.
    parallel_dims: Contains the world mesh and other parallelization dimensions.
    training: Training configuration.
    parallelism: Parallelism configuration.
    compile_config: Compile configuration.
    ac_config: Activation checkpointing configuration.
    dump_folder: Folder for dumped artifacts.
  """
  dtensor_parallelize.parallelize_qwen3(
      model,
      ac_config=ac_config,
      compile_config=compile_config,
      dump_folder=dump_folder,
      parallel_dims=parallel_dims,
      parallelism=parallelism,
      skip_dp=skip_dp,
      training=training,
  )
  return model
