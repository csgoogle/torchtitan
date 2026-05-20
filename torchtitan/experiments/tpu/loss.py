"""TPU specific loss functions and builders.

This module provides loss functions tailored for TPU execution, including
a Pallas implementation of cross-entropy loss with fallback to XLA.
"""

import torch
from torchtitan.components import loss as components_loss
import torchtitan.config
import torchtitan.trainer
import torchtitan.experiments.tpu.tpu_job_config as tpu_job_config_module
import torchtitan.tools.logging as torchtitan_logging


def pallas_cross_entropy_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    job_config: tpu_job_config_module.TPUTrainerConfig | None = None,
) -> torch.Tensor:
  """Pallas cross entropy loss with fallback to XLA for unsupported shapes."""
  if isinstance(pred, tuple) and len(pred) == 2:
    x, weights = pred
    if x.ndim == 3:
      x = x.flatten(0, 1)
    if labels.ndim == 2:
      labels = labels.flatten(0, 1)

    implementation = "mosaic_tpu"
    b_block_size = 1024
    h_block_size = 512
    v_block_size = 2048
    if job_config and hasattr(job_config, "loss_kernel"):
      b_block_size = job_config.loss_kernel.loss_b_block_size
      h_block_size = job_config.loss_kernel.loss_h_block_size
      v_block_size = job_config.loss_kernel.loss_v_block_size
      if not job_config.loss_kernel.enable_pallas_loss_kernel:
        implementation = "xla"

    if x.shape[0] % 1024 != 0 and implementation == "mosaic_tpu":
      implementation = "xla"
      torchtitan_logging.logger.warning(
          "Falling back to XLA implementation for Pallas loss "
          "due to unsupported shape: %s "
          "(batch size must be a multiple of 1024)",
          x.shape,
      )

    from torchtitan.experiments.tpu.kernels import linear_softmax_cross_entropy_loss  # pylint: disable=g-import-not-at-top

    return linear_softmax_cross_entropy_loss.linear_softmax_cross_entropy_loss(
        x,
        labels,
        weights,
        implementation=implementation,
        b_block_size=b_block_size,
        h_block_size=h_block_size,
        v_block_size=v_block_size,
    )
  else:
    raise ValueError("Pallas loss requires (x, weights) as input")


def build_cross_entropy_loss(
    job_config: (
        torchtitan.trainer.Trainer.Config | tpu_job_config_module.TPUTrainerConfig
    ),
    **kwargs
):
  if hasattr(job_config, "loss_kernel") and getattr(
      job_config.loss_kernel, "use_loss_kernel", True
  ):

    def loss_fn(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
      return pallas_cross_entropy_loss(pred, labels, job_config)

    return loss_fn
  return components_loss.CrossEntropyLoss(components_loss.CrossEntropyLoss.Config())
