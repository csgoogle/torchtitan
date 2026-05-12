"""Parallelization utilities for Llama models on TPU.

This module is used in distributed training tests configured with
`model_name="llama3_tpu".
"""

from torch import nn
import torchtitan.config
import torchtitan.distributed
import torchtitan.trainer
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.models.llama3 import parallelize as dtensor_parallelize
from torchtitan.tools.logging import logger


def parallelize_llama(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: (
        torchtitan.trainer.Trainer.Config
        | tpu_job_config.TPUJobConfig
        | tpu_job_config.TPUTrainerConfig
    ),
) -> nn.Module:
  """Parallelizes the Llama model based on the provided configuration and applies TPU-specific workarounds.

  Args:
    model: The nn.Module to be parallelized.
    parallel_dims: Contains the world mesh and other parallelization dimensions.
    job_config: The job configuration, potentially including TPU-specific
      settings.

  Returns:
    The parallelized nn.Module.
  """

  # TODO b/499075866: re-enable CPU offload for TPU torchtitan run.
  if job_config.training.enable_cpu_offload:
    logger.warning(
        "CPU offload is not supported for TPU torchtitan run, setting it to"
        " False."
    )
    job_config.training.enable_cpu_offload = False

  
  dtensor_parallelize.parallelize_llama(
      model,
      ac_config=job_config.activation_checkpoint,
      compile_config=job_config.compile,
      dump_folder="",
      parallel_dims=parallel_dims,
      parallelism=job_config.parallelism,
      training=job_config.training,
  )
  return model
