"""TPU specific loss functions and builders.

This module provides loss functions tailored for TPU execution, including
fused linear cross entropy implemented with Pallas.
"""

from dataclasses import dataclass
from typing import Literal

import math
import torch
from torchtitan.components import loss as components_loss
import torchtitan.config
import torchtitan.trainer
import torchtitan.experiments.tpu.tpu_job_config as tpu_job_config_module


def pallas_cross_entropy_loss(
    pred: tuple[torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    *,
    implementation: Literal["mosaic_tpu", "xla"] = "mosaic_tpu",
    b_block_size: int = 1024,
    h_block_size: int = 512,
    v_block_size: int = 2048,
) -> torch.Tensor:
  """Compute sum-reduced linear cross entropy without materializing logits.

  ``FusedLossLinearWrapper`` supplies hidden states and the transposed LM-head
  weight instead of computing logits itself. This function passes those
  operands to the fused kernel, which performs the projection, softmax, and
  cross entropy together. It also flattens TorchTitan's batched shapes to the
  token-major shapes required by Tokamax and rejects incompatible Mosaic tiles
  instead of silently switching kernels.
  """
  if not isinstance(pred, tuple) or len(pred) != 2:
    raise ValueError("Pallas loss requires (x, weights) as input")

  x, weights = pred
  if x.ndim == 3:
    x = x.flatten(0, 1)
  if labels.ndim == 2:
    labels = labels.flatten(0, 1)

  if x.ndim != 2 or labels.ndim != 1 or weights.ndim != 2:
    raise ValueError(
        "Pallas loss requires x [tokens, hidden], labels [tokens], and "
        "weights [hidden, vocab] or [vocab, hidden]"
    )
  if x.shape[1] == weights.shape[0]:
    weights_are_transposed = False
    vocab_size = weights.shape[1]
  elif x.shape[1] == weights.shape[1]:
    weights_are_transposed = True
    vocab_size = weights.shape[0]
  else:
    weights_are_transposed = False
    vocab_size = -1
  if x.shape[0] != labels.shape[0] or vocab_size < 0:
    raise ValueError(
        "Pallas loss received incompatible x, labels, and weight shapes: "
        f"{tuple(x.shape)}, {tuple(labels.shape)}, {tuple(weights.shape)}"
    )
  if min(b_block_size, h_block_size, v_block_size) <= 0:
    raise ValueError("Pallas loss block sizes must be positive")
  if implementation not in ("mosaic_tpu", "xla"):
    raise ValueError(f"Unsupported Pallas loss implementation: {implementation}")

  # ChatDataset marks prompt and padding tokens with IGNORE_INDEX (-100). Keep
  # the fixed token count required by the kernel, but zero those hidden states
  # and use an in-range label. Each ignored row then contributes exactly
  # log(vocab_size), which is removed from the sum below, with zero gradient.
  ignored = labels == components_loss.IGNORE_INDEX
  x = torch.where(ignored.unsqueeze(-1), torch.zeros_like(x), x)
  labels = torch.where(ignored, torch.zeros_like(labels), labels)

  # Do not silently switch implementations: an accidental XLA fallback changes
  # the memory and performance contract that selecting the fused loss promises.
  # Mosaic tiles must cover the token dimension exactly, so validate against the
  # configured tile rather than assuming the default b_block_size of 1024.
  if implementation == "mosaic_tpu" and x.shape[0] % b_block_size != 0:
    raise ValueError(
        f"Pallas loss requires the token count ({x.shape[0]}) to be a "
        f"multiple of b_block_size ({b_block_size}); use implementation='xla' "
        "or select a compatible block size"
    )

  from torchtitan.experiments.tpu.kernels import linear_softmax_cross_entropy_loss  # pylint: disable=g-import-not-at-top

  loss_sum = linear_softmax_cross_entropy_loss.linear_softmax_cross_entropy_loss(
      x,
      labels,
      weights,
      implementation=implementation,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
      weights_are_transposed=weights_are_transposed,
  )
  return loss_sum - ignored.sum().to(loss_sum.dtype) * math.log(vocab_size)


class PallasCrossEntropyLoss(components_loss.BaseLoss):
  """Expose fused linear cross entropy through TorchTitan's loss protocol.

  Kernel selection and tile sizes live in the loss config rather than the
  legacy side-band TPU config, so GraphTrainer builds and optionally compiles
  the same component as every other TorchTitan loss. ``BaseLoss`` also applies
  TorchTitan's global-valid-token normalization after the kernel's sum
  reduction.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(components_loss.BaseLoss.Config):
    implementation: Literal["mosaic_tpu", "xla"] = "mosaic_tpu"
    b_block_size: int = 1024
    h_block_size: int = 512
    v_block_size: int = 2048

  def __init__(
      self,
      config: Config,
      *,
      compile_config: torchtitan.config.CompileConfig | None = None
  ):
    self.fn = lambda pred, labels: pallas_cross_entropy_loss(
        pred,
        labels,
        implementation=config.implementation,
        b_block_size=config.b_block_size,
        h_block_size=config.h_block_size,
        v_block_size=config.v_block_size,
    )
    self.implementation = config.implementation
    self.b_block_size = config.b_block_size
    self.h_block_size = config.h_block_size
    self.v_block_size = config.v_block_size
    self._maybe_compile(compile_config)


def _legacy_pallas_loss_config(
    job_config: tpu_job_config_module.TPUTrainerConfig,
) -> PallasCrossEntropyLoss.Config:
  loss_kernel = job_config.loss_kernel
  return PallasCrossEntropyLoss.Config(
      implementation=(
          "mosaic_tpu" if loss_kernel.enable_pallas_loss_kernel else "xla"
      ),
      b_block_size=loss_kernel.loss_b_block_size,
      h_block_size=loss_kernel.loss_h_block_size,
      v_block_size=loss_kernel.loss_v_block_size,
  )


def build_cross_entropy_loss(
    job_config: (
        torchtitan.trainer.Trainer.Config
        | tpu_job_config_module.TPUTrainerConfig
    ),
    **kwargs
):
  """Build explicit loss config, with legacy TPU recipe compatibility.

  New recipes select ``PallasCrossEntropyLoss.Config`` directly. The
  ``loss_kernel`` branch preserves older regular-TPU callers during migration;
  all other recipes retain ordinary cross entropy.
  """
  if isinstance(job_config.loss, PallasCrossEntropyLoss.Config):
    return job_config.loss.build(compile_config=job_config.compile)

  # TODO: Remove this backward-compatibility fallback once all TPU recipes
  # select PallasCrossEntropyLoss.Config directly.
  if isinstance(job_config, tpu_job_config_module.TPUTrainerConfig) and getattr(
      job_config.loss_kernel, "use_loss_kernel", False
  ):
    return _legacy_pallas_loss_config(job_config).build(
        compile_config=job_config.compile
    )
  return components_loss.CrossEntropyLoss(components_loss.CrossEntropyLoss.Config())
