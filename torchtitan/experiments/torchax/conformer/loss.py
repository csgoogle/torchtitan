"""JAX CTC Loss for Conformer in torchax."""

import dataclasses
import jax.numpy as jnp
import optax
import torch
import torchax
import torchtitan.components.loss as components_loss
import torchtitan.config as torchtitan_config


def conformer_jax_ctc_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
  """Computes JAX-native CTC loss using optax under Torchax JIT."""
  env = torchax.default_env()

  # Get JAX views of the input tensors
  logits_jax = torchax.interop.jax_view(logits)
  labels_jax = torchax.interop.jax_view(labels)

  # Get vocab_size dynamically from the logits view
  vocab_size = logits_jax.shape[-1]

  # Ignore index for padding (matches PyTorch's default -100)
  ignore_mask = (labels_jax == -100)

  # Replace -100 with 0 and modulo positive label IDs by vocab_size
  safe_labels = jnp.where(ignore_mask, 0, labels_jax)
  safe_labels = (safe_labels % vocab_size).astype(jnp.int32)
  label_paddings = ignore_mask.astype(jnp.float32)

  # Assuming inputs are not padded at the end (logit_paddings is all zeros)
  # matches tpu train_minimal where example full length is used.
  logit_paddings = jnp.zeros(logits_jax.shape[:-1], dtype=jnp.float32)

  # Compute Optax CTC loss: returns (B,) loss array
  loss_per_seq = optax.losses.ctc_loss(
      logits_jax, logit_paddings, safe_labels, label_paddings, blank_id=0
  )

  # Return the sum of loss across sequences in the batch
  total_loss = jnp.sum(loss_per_seq)

  # Convert back to PyTorch Tensor view
  return env.j2t_iso(total_loss)


class CTCLoss(components_loss.BaseLoss):
  """JAX CTC Loss component for Torchax Conformer."""

  @dataclasses.dataclass(kw_only=True, slots=True)
  class Config(components_loss.BaseLoss.Config):
    pass

  def __init__(self, config: Config, *, compile_config: torchtitan_config.CompileConfig | None = None):
    self.fn = conformer_jax_ctc_loss
    self._maybe_compile(compile_config)
