"""Unit test comparing JAX CTC loss outputs and gradients to PyTorch CTC loss."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
import torch
from torchtitan.experiments.tpu.conformer import jax_ctc_loss


class JAXLossEquivalenceTest(absltest.TestCase):
  """Test case for JAX CTC loss equivalence on CPU."""

  def test_equivalence(self):
    batch = 4
    seq_len = 128
    vocab_size = 64
    target_len = 30

    torch.manual_seed(42)
    # Standard random logits
    logits_pt = torch.randn(
        batch, seq_len, vocab_size, dtype=torch.float32, requires_grad=True
    )

    # Targets (avoiding blank token 0, using 1..vocab_size-1)
    targets_pt = torch.randint(
        1, vocab_size, (batch, target_len), dtype=torch.int32
    )

    # PyTorch CTC loss expects input lengths and target lengths
    output_lengths = torch.full((batch,), seq_len, dtype=torch.int32)
    target_lengths = torch.full((batch,), target_len, dtype=torch.int32)

    # --- 1. PyTorch Loss & Gradient ---
    logits_f32 = logits_pt.float()
    log_probs = torch.nn.functional.log_softmax(logits_f32, dim=-1).transpose(
        0, 1
    )

    loss_pt, _ = torch.ops.aten._ctc_loss.Tensor(
        log_probs,
        targets_pt.to(torch.long),  # Aten expects long targets
        output_lengths.to(torch.long),
        target_lengths.to(torch.long),
        blank=0,
        zero_infinity=False,
    )
    loss_pt_sum = loss_pt.sum()

    # Backward pass
    loss_pt_sum.backward()
    grad_pt = logits_pt.grad.clone()

    # --- 2. JAX Loss & Gradient ---
    # Convert PyTorch inputs to JAX arrays
    logits_jax = jnp.array(logits_pt.detach().numpy())
    targets_jax = jnp.array(targets_pt.numpy())

    # Forward pass (direct JAX call)
    loss_jax = jax_ctc_loss.ctc_loss_forward_jax(logits_jax, targets_jax)

    # Backward pass (direct JAX call)
    d_loss = jnp.array(1.0, dtype=jnp.float32)
    grad_jax = jax_ctc_loss.ctc_loss_backward_jax(logits_jax, targets_jax, d_loss)

    # --- 3. Assertions ---
    loss_pt_val = loss_pt_sum.item()
    loss_jax_val = float(loss_jax)

    # Test loss difference
    loss_diff = abs(loss_pt_val - loss_jax_val)
    self.assertLess(
        loss_diff,
        1e-3,
        f"Loss difference too large: PT={loss_pt_val}, JAX={loss_jax_val}",
    )

    # Test gradient difference
    grad_pt_np = grad_pt.numpy()
    grad_jax_np = np.array(grad_jax)
    max_grad_diff = np.max(np.abs(grad_pt_np - grad_jax_np))
    print(f"Loss difference: {loss_diff}")
    print(f"Max gradient difference: {max_grad_diff}")

    self.assertLess(
        max_grad_diff,
        5e-4,
        f"Max gradient difference too large: {max_grad_diff}",
    )


if __name__ == "__main__":
  absltest.main()
