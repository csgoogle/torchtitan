"""Parallelization utilities for Qwen3 models on TPU.

This module is used in distributed training tests configured with
`model_name="qwen3_tpu"`.
"""

from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.experiments.tpu import workarounds
from torchtitan.models.qwen3.infra import parallelize as dtensor_parallelize
from torchtitan.tools.logging import logger


def parallelize_qwen3(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: (
        torchtitan.config.JobConfig
        | torchtitan.experiments.tpu.tpu_job_config.TPUJobConfig
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

  # Enable CPU histc workaround.
  workarounds.use_cpu_safe_histc_patch()

  if (
      isinstance(job_config, tpu_job_config.TPUJobConfig)
      and job_config.splash_attention_kernel.use_splash_attention_kernel
  ):
    workarounds.use_splash_attention_patch(model)
  if (
      isinstance(job_config, tpu_job_config.TPUJobConfig)
      and job_config.loss_kernel.use_loss_kernel
  ):
    workarounds.use_output_projection_patch(model)

  if (
      isinstance(job_config, tpu_job_config.TPUJobConfig)
      and job_config.qwen3.use_gmm_kernel
  ):
    workarounds.use_gmm_kernel_patch(model)

  if (
      isinstance(job_config, tpu_job_config.TPUJobConfig)
      and job_config.qwen3.use_fill_indices_kernel
  ):
    workarounds.use_fill_indices_patch(model)
  dtensor_parallelize.parallelize_qwen3(model, parallel_dims, job_config)
  return model
