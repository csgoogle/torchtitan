"""Tests for splash attention on TPU."""

from absl import logging
from absl.testing import absltest
import torch
from torch.nn import attention
import torch.nn.functional as F
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa


class SplashAttentionTest(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for the splash_sdpa kernel on TPU.

  This class contains tests to ensure the correctness of the `splash_sdpa`
  function, including parity with PyTorch's scaled_dot_product_attention
  (SDPA) in both forward and backward passes for Multi-Head Attention (MHA)
  and Grouped Query Attention (GQA) configurations.
  """

  def test_mha_parity(self):
    device = self.accelerator_device
    b, n_heads, seq_len, head_dim = 2, 8, 128, 64

    # Generate on CPU first
    q = torch.randn(b, n_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(b, n_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(b, n_heads, seq_len, head_dim, dtype=torch.float32)

    # Move to TPU
    q_tpu, k_tpu, v_tpu = q.to(device), k.to(device), v.to(device)

    # Reference Math SDPA on TPU
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      expected_out = F.scaled_dot_product_attention(
          q_tpu, k_tpu, v_tpu, is_causal=True
      )

    # TPU Splash SDPA
    actual_out = splash_sdpa(
        q_tpu, k_tpu, v_tpu, is_causal=True, enable_gqa=False
    )

    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=5e-2, atol=5e-2
    )
    logging.info("MHA Parity test passed. Output shape: %s", actual_out.shape)

  def test_mqa_parity(self):
    device = self.accelerator_device
    b, n_heads, n_kv_heads, seq_len, head_dim = 2, 8, 2, 128, 64

    # Generate on CPU first
    q = torch.randn(b, n_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(b, n_kv_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(b, n_kv_heads, seq_len, head_dim, dtype=torch.float32)

    # Move to TPU
    q_tpu, k_tpu, v_tpu = q.to(device), k.to(device), v.to(device)

    # Reference Math SDPA on TPU (manual GQA expansion)
    n_rep = n_heads // n_kv_heads
    k_exp = k_tpu.repeat_interleave(n_rep, dim=1)
    v_exp = v_tpu.repeat_interleave(n_rep, dim=1)

    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      expected_out = F.scaled_dot_product_attention(
          q_tpu, k_exp, v_exp, is_causal=True
      )

    # TPU Splash SDPA
    actual_out = splash_sdpa(
        q_tpu, k_tpu, v_tpu, is_causal=True, enable_gqa=True
    )

    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=5e-2, atol=5e-2
    )
    logging.info(
        "MQA/GQA Parity test passed. Output shape: %s", actual_out.shape
    )

  def test_mha_backward(self):
    """Gradients from splash attention backward match reference SDPA."""
    device = self.accelerator_device
    b, n_heads, seq_len, head_dim = 2, 8, 128, 64

    q = torch.randn(
        b, n_heads, seq_len, head_dim, dtype=torch.float32, device=device
    )
    k = torch.randn(
        b, n_heads, seq_len, head_dim, dtype=torch.float32, device=device
    )
    v = torch.randn(
        b, n_heads, seq_len, head_dim, dtype=torch.float32, device=device
    )

    # Reference backward via math SDPA.
    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    v_ref = v.clone().requires_grad_(True)
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          q_ref, k_ref, v_ref, is_causal=True
      )
    out_ref.sum().backward()

    # Splash backward.
    q_test = q.clone().requires_grad_(True)
    k_test = k.clone().requires_grad_(True)
    v_test = v.clone().requires_grad_(True)
    out_test = splash_sdpa(
        q_test, k_test, v_test, is_causal=True, enable_gqa=False
    )
    out_test.sum().backward()

    # Confirm the backward kernel was invoked and produced non-zero gradients.
    # If _SplashAttentionFn.backward was never called, these would be None.
    self.assertIsNotNone(
        q_test.grad, "grad_q is None — splash backward not called"
    )
    self.assertIsNotNone(
        k_test.grad, "grad_k is None — splash backward not called"
    )
    self.assertIsNotNone(
        v_test.grad, "grad_v is None — splash backward not called"
    )
    self.assertGreater(q_test.grad.abs().sum().item(), 0, "grad_q is all zeros")
    self.assertGreater(k_test.grad.abs().sum().item(), 0, "grad_k is all zeros")
    self.assertGreater(v_test.grad.abs().sum().item(), 0, "grad_v is all zeros")

    torch.testing.assert_close(
        q_test.grad.cpu(), q_ref.grad.cpu(), rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(
        k_test.grad.cpu(), k_ref.grad.cpu(), rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(
        v_test.grad.cpu(), v_ref.grad.cpu(), rtol=5e-2, atol=5e-2
    )
    logging.info("MHA backward test passed.")

  def test_mqa_backward(self):
    """Gradients from splash GQA backward match reference SDPA."""
    device = self.accelerator_device
    b, n_heads, n_kv_heads, seq_len, head_dim = 2, 8, 2, 128, 64
    n_rep = n_heads // n_kv_heads

    q = torch.randn(
        b, n_heads, seq_len, head_dim, dtype=torch.float32, device=device
    )
    k = torch.randn(
        b, n_kv_heads, seq_len, head_dim, dtype=torch.float32, device=device
    )
    v = torch.randn(
        b, n_kv_heads, seq_len, head_dim, dtype=torch.float32, device=device
    )

    # Reference backward (expand k/v to full heads for math SDPA).
    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    v_ref = v.clone().requires_grad_(True)
    k_exp = k_ref.repeat_interleave(n_rep, dim=1)
    v_exp = v_ref.repeat_interleave(n_rep, dim=1)
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          q_ref, k_exp, v_exp, is_causal=True
      )
    out_ref.sum().backward()
    # k_ref/v_ref grads are summed over the repeated heads.
    k_ref_grad = k_ref.grad
    v_ref_grad = v_ref.grad

    # Splash GQA backward.
    q_test = q.clone().requires_grad_(True)
    k_test = k.clone().requires_grad_(True)
    v_test = v.clone().requires_grad_(True)
    out_test = splash_sdpa(
        q_test, k_test, v_test, is_causal=True, enable_gqa=True
    )
    out_test.sum().backward()

    self.assertIsNotNone(
        q_test.grad, "grad_q is None — splash backward not called"
    )
    self.assertIsNotNone(
        k_test.grad, "grad_k is None — splash backward not called"
    )
    self.assertIsNotNone(
        v_test.grad, "grad_v is None — splash backward not called"
    )
    self.assertGreater(q_test.grad.abs().sum().item(), 0, "grad_q is all zeros")
    self.assertGreater(k_test.grad.abs().sum().item(), 0, "grad_k is all zeros")
    self.assertGreater(v_test.grad.abs().sum().item(), 0, "grad_v is all zeros")

    torch.testing.assert_close(
        q_test.grad.cpu(), q_ref.grad.cpu(), rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(
        k_test.grad.cpu(), k_ref_grad.cpu(), rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(
        v_test.grad.cpu(), v_ref_grad.cpu(), rtol=5e-2, atol=5e-2
    )
    logging.info("MQA/GQA backward test passed.")


if __name__ == "__main__":
  absltest.main()
