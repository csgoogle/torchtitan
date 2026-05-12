"""Parallelization utilities for Qwen3 models on TPU.

This module is used in distributed training tests configured with
`model_name="qwen3_tpu"`.
"""

from torch import nn
import torchtitan.config
import torchtitan.distributed
import torchtitan.trainer
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.models.qwen3 import parallelize as dtensor_parallelize


def parallelize_qwen3(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: (
        torchtitan.trainer.Trainer.Config
        | tpu_job_config.TPUJobConfig
        | tpu_job_config.TPUTrainerConfig
    ),
) -> nn.Module:
  """Parallelizes the Qwen3 model based on the job configuration and applies TPU-specific workarounds.

  Args:
    model: The nn.Module to be parallelized.
    parallel_dims: Contains the world mesh and other parallelization dimensions.
    job_config: The job configuration, used to determine the parallelization
      strategy.

  Returns:
    The parallelized nn.Module.
  """


  dtensor_parallelize.parallelize_qwen3(
      model,
      ac_config=job_config.activation_checkpoint,
      compile_config=job_config.compile,
      dump_folder="",
      parallel_dims=parallel_dims,
      parallelism=job_config.parallelism,
      skip_dp=False,
      training=job_config.training,
  )
  return model
