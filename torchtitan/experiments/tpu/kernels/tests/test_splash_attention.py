"""Tests for splash attention on TPU."""

from absl import logging
from absl.testing import absltest
import torch
from torch.nn import attention
import torch.nn.functional as F
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa


class SplashAttentionTest(base_device_test.BaseAcceleratorDeviceTest):

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


if __name__ == "__main__":
  absltest.main()
