"""TPU specific loss functions and builders.

This module provides loss functions tailored for TPU execution, including
a Pallas implementation of cross-entropy loss with fallback to XLA.
"""

import torch
from torchtitan.components import loss as components_loss
import torchtitan.config
import torchtitan.experiments.tpu.tpu_job_config as tpu_job_config_module
import torchtitan.tools.logging as torchtitan_logging


def pallas_cross_entropy_loss(
    pred: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
  """Pallas cross entropy loss with fallback to XLA for unsupported shapes."""
  if isinstance(pred, tuple) and len(pred) == 2:
    x, weights = pred
    if x.ndim == 3:
      x = x.flatten(0, 1)
    if labels.ndim == 2:
      labels = labels.flatten(0, 1)

    implementation = "mosaic_tpu"
    if x.shape[0] % 1024 != 0:
      implementation = "xla"
      torchtitan_logging.logger.warning(
          "Falling back to XLA implementation for Pallas loss "
          "due to unsupported shape: %s "
          "(batch size must be a multiple of 1024)",
          x.shape,
      )
    from torchtitan.experiments.tpu.kernels import linear_softmax_cross_entropy_loss  # pylint: disable=g-import-not-at-top
    return linear_softmax_cross_entropy_loss.linear_softmax_cross_entropy_loss(
        x, labels, weights, implementation=implementation
    )
  else:
    raise ValueError("Pallas loss requires (x, weights) as input")


def build_cross_entropy_loss(
    job_config: (
        torchtitan.config.JobConfig | tpu_job_config_module.TPUJobConfig
    ),
    **kwargs
):
  if hasattr(job_config, "tpu_config") and getattr(
      job_config.tpu_config, "use_loss_kernel", True
  ):
    return pallas_cross_entropy_loss
  return components_loss.build_cross_entropy_loss(job_config, **kwargs)
