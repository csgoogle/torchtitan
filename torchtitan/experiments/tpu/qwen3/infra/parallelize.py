"""Parallelization utilities for Qwen3 models on TPU.

NOTE: This module is used in distributed training tests configured with
`model_name="qwen3_tpu"` to invoke local parallelization modules.
"""

import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.models.qwen3.infra import parallelize as dtensor_parallelize
from torchtitan.experiments.tpu.qwen3.infra import fairscale_parallelize


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
  if tpu_job_config.use_fairscale(job_config):
    rank = torch.distributed.get_rank()
    fairscale_parallelize.parallelize_qwen3(
        model, parallel_dims, job_config, rank)
  else:
    dtensor_parallelize.parallelize_qwen3(model, parallel_dims, job_config)
  return model
