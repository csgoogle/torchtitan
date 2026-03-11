"""Parallelization utilities for Flux models on TPU."""

from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.models.flux.infra.parallelize import parallelize_flux as native_parallelize_flux


def parallelize_flux(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: torchtitan.config.JobConfig,
) -> nn.Module:
  """Parallelizes the Flux model based on the provided configuration.

  Args:
    model: The nn.Module to be parallelized.
    parallel_dims: Contains the world mesh and other parallelization dimensions.
    job_config: The job configuration.

  Returns:
    The parallelized nn.Module.
  """
  # For now, we delegate to the native implementation.
  # If TPU-specific optimizations are needed, we can implement them here.
  return native_parallelize_flux(model, parallel_dims, job_config)
