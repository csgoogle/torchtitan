"""JAX CTC Loss for Conformer in torchax."""

import dataclasses
import jax
import jax.numpy as jnp
import optax
import torch
import torchax
import torchtitan.components.loss as components_loss
import torchtitan.config as torchtitan_config


def _jax_ctc_loss_fwd_helper(logits_jax, labels_jax):
  """Computes CTC loss using optax."""
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
  return jnp.sum(loss_per_seq)


@jax.custom_vjp
def jax_ctc_loss_with_vjp(logits_jax, labels_jax):
  """Computes CTC loss with custom VJP."""
  return _jax_ctc_loss_fwd_helper(logits_jax, labels_jax)


@jax.named_call
def _jax_ctc_loss_fwd(logits_jax, labels_jax):
  """Forward pass for custom VJP of CTC loss."""
  loss, vjp_fn = jax.vjp(_jax_ctc_loss_fwd_helper, logits_jax, labels_jax)
  return loss, vjp_fn


@jax.named_call
def _jax_ctc_loss_bwd(vjp_fn, g):
  """Backward pass for custom VJP of CTC loss."""
  with jax.named_scope("ctc_loss_backward"):
    grad_logits, grad_labels = vjp_fn(g)
    return grad_logits, grad_labels


jax_ctc_loss_with_vjp.defvjp(_jax_ctc_loss_fwd, _jax_ctc_loss_bwd)


@jax.named_call
def conformer_jax_ctc_loss(logits, labels):
  """Computes JAX-native CTC loss using optax under Torchax JIT."""
  env = torchax.default_env()

  # Get JAX views of the input tensors
  logits_jax = torchax.interop.jax_view(logits)
  labels_jax = torchax.interop.jax_view(labels)

  total_loss = jax_ctc_loss_with_vjp(logits_jax, labels_jax)

  # Convert back to PyTorch Tensor view
  return env.j2t_iso(total_loss)


class CTCLoss(components_loss.BaseLoss):
  """JAX CTC Loss component for Torchax Conformer."""

  @dataclasses.dataclass(kw_only=True, slots=True)
  class Config(components_loss.BaseLoss.Config):
    pass

  def __init__(
      self,
      config: Config,
      *,
      compile_config: torchtitan_config.CompileConfig | None = None,
  ):
    """Initializes the CTCLoss component."""
    self.fn = conformer_jax_ctc_loss
    self._maybe_compile(compile_config)
