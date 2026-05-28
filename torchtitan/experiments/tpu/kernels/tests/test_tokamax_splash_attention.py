"""Tests for Tokamax splash attention on TPU."""

from collections.abc import Callable
import functools
import logging

from absl.testing import absltest
import torch
from torch.nn import attention
import torch.nn.functional as F
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import tokamax_splash_attention


class SplashAttentionTokamaxTest(base_device_test.BaseAcceleratorDeviceTest):

  def check_forward_parity(
      self,
      splash_fn: Callable[
          [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
      ],
      b: int,
      n_heads: int,
      n_kv_heads: int,
      seq_len: int,
      head_dim: int,
  ):
    device = self.accelerator_device

    # Generates inputs in (B, S, H, D) layout standard
    q = torch.randn(b, seq_len, n_heads, head_dim, dtype=torch.float32)
    k = torch.randn(b, seq_len, n_kv_heads, head_dim, dtype=torch.float32)
    v = torch.randn(b, seq_len, n_kv_heads, head_dim, dtype=torch.float32)
    q_tpu, k_tpu, v_tpu = q.to(device), k.to(device), v.to(device)

    # Expands kv_head dim if necessary (head dim is at axis 2 now)
    n_rep = n_heads // n_kv_heads
    k_exp = k_tpu.repeat_interleave(n_rep, dim=2)
    v_exp = v_tpu.repeat_interleave(n_rep, dim=2)

    # Reference SDPA Math expects (B, H, S, D)
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      expected_out = F.scaled_dot_product_attention(
          q_tpu.transpose(1, 2),
          k_exp.transpose(1, 2),
          v_exp.transpose(1, 2),
          is_causal=True,
      )
      expected_out = expected_out.transpose(1, 2)

    actual_out = splash_fn(q_tpu, k_tpu, v_tpu)

    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=5e-2, atol=5e-2
    )

  def test_mha_parity(self):
    splash_fn = functools.partial(
        tokamax_splash_attention.splash_sdpa, is_causal=True, enable_gqa=False
    )

    self.check_forward_parity(
        splash_fn=splash_fn,
        b=2,
        n_heads=8,
        n_kv_heads=8,
        seq_len=128,
        head_dim=64,
    )

  def test_mqa_parity(self):
    splash_fn = functools.partial(
        tokamax_splash_attention.splash_sdpa, is_causal=True, enable_gqa=True
    )

    self.check_forward_parity(
        splash_fn=splash_fn,
        b=2,
        n_heads=8,
        n_kv_heads=2,
        seq_len=128,
        head_dim=64,
    )

  def test_mqa_parity_one_kv_head(self):
    splash_fn = functools.partial(
        tokamax_splash_attention.splash_sdpa, is_causal=True, enable_gqa=True
    )

    self.check_forward_parity(
        splash_fn=splash_fn,
        b=2,
        n_heads=8,
        n_kv_heads=1,
        seq_len=128,
        head_dim=64,
    )

  def test_block_sizes_parity(self):
    splash_fn = functools.partial(
        tokamax_splash_attention.splash_sdpa,
        is_causal=True,
        enable_gqa=False,
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
    )

    self.check_forward_parity(
        splash_fn=splash_fn,
        b=2,
        n_heads=8,
        n_kv_heads=8,
        seq_len=128,
        head_dim=64,
    )

  def test_gpu_available(self):
    device = self.accelerator_device
    self.assertEqual(device.type, "tpu")

  def test_mha_backward(self):
    """Gradients from splash attention backward match reference SDPA."""
    device = self.accelerator_device
    b, n_heads, seq_len, head_dim = 2, 8, 128, 64

    # (B, S, H, D)
    q = torch.randn(
        b, seq_len, n_heads, head_dim, dtype=torch.float32, device=device
    )
    k = torch.randn(
        b, seq_len, n_heads, head_dim, dtype=torch.float32, device=device
    )
    v = torch.randn(
        b, seq_len, n_heads, head_dim, dtype=torch.float32, device=device
    )

    # Reference backward via math SDPA (expects B, H, S, D).
    q_ref = q.transpose(1, 2).clone().requires_grad_(True)
    k_ref = k.transpose(1, 2).clone().requires_grad_(True)
    v_ref = v.transpose(1, 2).clone().requires_grad_(True)
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          q_ref, k_ref, v_ref, is_causal=True
      )
    out_ref.sum().backward()

    # Splash backward.
    q_test = q.clone().requires_grad_(True)
    k_test = k.clone().requires_grad_(True)
    v_test = v.clone().requires_grad_(True)
    out_test = tokamax_splash_attention.splash_sdpa(
        q_test, k_test, v_test, is_causal=True, enable_gqa=False
    )

    out_test.sum().backward()

    # Confirm the backward kernel was invoked and produced non-zero gradients.
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

    # Transpose the reference gradients back to compare with our raw (B, S, H, D) gradients!
    torch.testing.assert_close(
        q_test.grad.cpu(),
        q_ref.grad.transpose(1, 2).cpu(),
        rtol=5e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        k_test.grad.cpu(),
        k_ref.grad.transpose(1, 2).cpu(),
        rtol=5e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        v_test.grad.cpu(),
        v_ref.grad.transpose(1, 2).cpu(),
        rtol=5e-2,
        atol=5e-2,
    )
    logging.info("MHA backward test passed.")

  def test_mqa_backward(self):
    """Gradients from splash GQA backward match reference SDPA."""
    device = self.accelerator_device
    b, n_heads, n_kv_heads, seq_len, head_dim = 2, 8, 2, 128, 64
    n_rep = n_heads // n_kv_heads

    # (B, S, H, D)
    q = torch.randn(
        b, seq_len, n_heads, head_dim, dtype=torch.float32, device=device
    )
    k = torch.randn(
        b, seq_len, n_kv_heads, head_dim, dtype=torch.float32, device=device
    )
    v = torch.randn(
        b, seq_len, n_kv_heads, head_dim, dtype=torch.float32, device=device
    )

    # Reference backward (expand k/v to full heads and transpose to B, H, S, D).
    q_ref = q.transpose(1, 2).clone().requires_grad_(True)
    k_ref = k.transpose(1, 2).clone().requires_grad_(True)
    v_ref = v.transpose(1, 2).clone().requires_grad_(True)
    k_exp = k_ref.repeat_interleave(n_rep, dim=1)
    v_exp = v_ref.repeat_interleave(n_rep, dim=1)
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          q_ref, k_exp, v_exp, is_causal=True
      )
    out_ref.sum().backward()
    k_ref_grad = k_ref.grad.transpose(1, 2)
    v_ref_grad = v_ref.grad.transpose(1, 2)

    # Splash GQA backward.
    q_test = q.clone().requires_grad_(True)
    k_test = k.clone().requires_grad_(True)
    v_test = v.clone().requires_grad_(True)
    out_test = tokamax_splash_attention.splash_sdpa(
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
        q_test.grad.cpu(),
        q_ref.grad.transpose(1, 2).cpu(),
        rtol=5e-2,
        atol=5e-2,
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
