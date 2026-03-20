"""Parallelization utilities for Qwen3 models on TPU.

NOTE: This module is used in distributed training tests configured with
`model_name="qwen3_tpu"` to invoke local parallelization modules.
"""

import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.tools.logging import logger
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.qwen3.infra import fairscale_parallelize
from torchtitan.models.qwen3.infra import parallelize as dtensor_parallelize


def parallelize_qwen3(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: (
        torchtitan.config.JobConfig
        | torchtitan.experiments.tpu.tpu_job_config.TPUJobConfig
    ),
) -> nn.Module:
  """Parallelizes the Qwen3 model based on the job configuration.

  This function dispatches to either fairscale or dtensor based parallelization
  strategies from TPU-specific modules (fairscale_parallelize and
  dtensor_parallelize).

  Args:
    model: The nn.Module to be parallelized.
    parallel_dims: Contains the world mesh and other parallelization dimensions.
    job_config: The job configuration, used to determine the parallelization
      strategy.

  Returns:
    The parallelized nn.Module.
  """
  if (
      isinstance(job_config, tpu_job_config.TPUJobConfig)
      and job_config.tpu_config.use_splash_attention_kernel
  ):
    workarounds.use_splash_attention_patch(model)
  if (
      isinstance(job_config, tpu_job_config.TPUJobConfig)
      and job_config.tpu_config.use_loss_kernel
  ):
    workarounds.use_output_projection_patch(model)

  if (
      isinstance(job_config, tpu_job_config.TPUJobConfig)
      and job_config.tpu_config.use_gmm_kernel
  ):
    workarounds.use_gmm_kernel_patch(model)

  if (
      isinstance(job_config, tpu_job_config.TPUJobConfig)
      and job_config.tpu_config.use_fill_indices_kernel
  ):
    workarounds.use_fill_indices_patch(model)

  if tpu_job_config.use_fairscale(job_config):
    rank = torch.distributed.get_rank()
    fairscale_parallelize.parallelize_qwen3(
        model, parallel_dims, job_config, rank)
  else:
    dtensor_parallelize.parallelize_qwen3(model, parallel_dims, job_config)
  return model
