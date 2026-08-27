"""Tests for splash attention on TPU."""

import functools
from typing import Callable

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

    # Generates inputs on cpu and moves to tpu.
    q = torch.randn(b, n_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(b, n_kv_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(b, n_kv_heads, seq_len, head_dim, dtype=torch.float32)
    q_tpu, k_tpu, v_tpu = q.to(device), k.to(device), v.to(device)

    # Expands kv_head dim if necessary.
    n_rep = n_heads // n_kv_heads
    k_exp = k_tpu.repeat_interleave(n_rep, dim=1)
    v_exp = v_tpu.repeat_interleave(n_rep, dim=1)

    # Computes expected result using Math SDPA on TPU.
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      expected_out = F.scaled_dot_product_attention(
          q_tpu, k_exp, v_exp, is_causal=True
      )

    actual_out = splash_fn(q_tpu, k_tpu, v_tpu)

    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=5e-2, atol=5e-2
    )

  def test_mha_parity(self):
    splash_fn = functools.partial(splash_sdpa, is_causal=True, enable_gqa=False)
    self.check_forward_parity(
        splash_fn=splash_fn,
        b=2,
        n_heads=8,
        n_kv_heads=8,
        seq_len=128,
        head_dim=64,
    )

  def make_packed_inputs(self):
    """Helper function to create packed inputs and reference block causal
    mask for SDPA MATH.
    """
    device = self.accelerator_device
    b, n_heads, seq_len, head_dim = 2, 8, 128, 64

    q, k, v = (
        torch.randn(
            b, n_heads, seq_len, head_dim, dtype=torch.float32, device=device
        )
        for _ in range(3)
    )

    segment_ids = torch.zeros(b, seq_len, dtype=torch.int32, device=device)
    segment_ids[0, 64:] = 1
    segment_ids[1, 32:64] = 1
    segment_ids[1, 64:96] = 2
    segment_ids[1, 96:] = 3

    # We build the final reference mask by building a document mask and a
    # causal mask and ANDing them together.
    # Shape: (B, S, S)

    # We build the document mask from segment_ids tensor of shape (B, S)
    # unsqueeze(2) adds a dimension at index 2 -> (B, S, 1)
    # unsqueeze(1) adds a dimension at index 1 -> (B, 1, S)
    # The comparison broadcasts the two tensors into resulting shape (B, S, S)
    # where true only if two tokens at i, j belong to the same document
    document_mask = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )
    attn_mask = document_mask & causal_mask.unsqueeze(0)  # (B, S, S)

    # SDPA takes attn_mask of shape (B, H, S, S), (B, S, S) or (S, S). Passing
    # (B, S, S) with H=8 could broadcast B against H, so unsqueeze to
    # (B, 1, S, S) to broadcast over heads explicitly.
    return q, k, v, segment_ids, attn_mask.unsqueeze(1).cpu()

  def test_segment_ids_parity(self):
    """Verifies SplashAttention with segment_ids matches SDPA MATH backend with

    custom attn_mask.
    """
    q, k, v, segment_ids, attn_mask_cpu = self.make_packed_inputs()
    q_cpu, k_cpu, v_cpu = q.cpu(), k.cpu(), v.cpu()

    # For the reference SDPA MATH implementation, we pass the manually
    # constructed final mask (document + causal) and must set `is_causal=False`
    # as required by PyTorch so our custom mask is honored.
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      expected_out_cpu = F.scaled_dot_product_attention(
          q_cpu, k_cpu, v_cpu, attn_mask=attn_mask_cpu, is_causal=False
      )

    # For our `splash_sdpa` wrapper, we test the functionality of constructing
    # the final mask by passing both `segment_ids` and `is_causal=True`, and the
    # mask construction is handled in the kernel.
    actual_out = splash_sdpa(
        q, k, v, segment_ids=segment_ids, is_causal=True, enable_gqa=False
    )

    torch.testing.assert_close(
        actual_out.cpu(), expected_out_cpu, rtol=5e-2, atol=5e-2
    )

  def test_segment_ids_backward_parity(self):
    """Gradients from SplashAttention backward with segment_ids match SDPA MATH

    backend with custom attn_mask."""
    q, k, v, segment_ids, attn_mask_cpu = self.make_packed_inputs()

    # Reference backward via math SDPA on CPU
    q_ref = q.cpu().requires_grad_(True)
    k_ref = k.cpu().requires_grad_(True)
    v_ref = v.cpu().requires_grad_(True)

    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          q_ref, k_ref, v_ref, attn_mask=attn_mask_cpu, is_causal=False
      )
    out_ref.sum().backward()

    # Splash backward.
    q_test = q.clone().requires_grad_(True)
    k_test = k.clone().requires_grad_(True)
    v_test = v.clone().requires_grad_(True)
    out_test = splash_sdpa(
        q_test,
        k_test,
        v_test,
        segment_ids=segment_ids,
        is_causal=True,
        enable_gqa=False,
    )
    out_test.sum().backward()

    self.assertIsNotNone(
        q_test.grad, "grad_q is None — splash backward not called"
    )
    self.assertGreater(q_test.grad.abs().sum().item(), 0, "grad_q is all zeros")

    torch.testing.assert_close(
        q_test.grad.cpu(), q_ref.grad.cpu(), rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(
        k_test.grad.cpu(), k_ref.grad.cpu(), rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(
        v_test.grad.cpu(), v_ref.grad.cpu(), rtol=5e-2, atol=5e-2
    )
    logging.info("Segment IDs backward test passed.")

  def test_mqa_parity(self):
    splash_fn = functools.partial(splash_sdpa, is_causal=True, enable_gqa=True)
    self.check_forward_parity(
        splash_fn=splash_fn,
        b=2,
        n_heads=8,
        n_kv_heads=2,
        seq_len=128,
        head_dim=64,
    )

  def test_mqa_parity_one_kv_head(self):
    splash_fn = functools.partial(splash_sdpa, is_causal=True, enable_gqa=True)
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
        splash_sdpa,
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
        seq_len=256,
        head_dim=64,
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
