# Copyright 2026 Google LLC

"""Tests for splash attention with bias + PRNG dropout on TPU.

Conventions follow test_tokamax_splash_attention.py: absltest +
BaseAcceleratorDeviceTest, (B, S, H, D) layouts, SDPA MATH-backend
references. `SbdConfig` values ride as static op arguments (part of the
trace cache key), so config changes are safe at any shape; the per-key
test reuses the full-rank test's shape on purpose to lock that in.
"""

import functools
import unittest

from absl.testing import absltest
import torch
from torch.nn import attention
import torch.nn.functional as F
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import splash_bias_dropout
from torchtitan.experiments.tpu.kernels import tokamax_splash_attention


def _sdpa_ref(q, k, v, attn_mask=None, is_causal=False):
  """Math-backend SDPA reference on CPU (torch_tpu overrides SDPA on-device
  with a fused op whose backward cannot produce attn_mask grads); inputs
  and outputs are CPU (B, S, H, D) tensors."""
  with attention.sdpa_kernel([attention.SDPBackend.MATH]):
    out = F.scaled_dot_product_attention(
        q.cpu().transpose(1, 2),
        k.cpu().transpose(1, 2),
        v.cpu().transpose(1, 2),
        attn_mask=attn_mask.cpu() if attn_mask is not None else None,
        is_causal=is_causal,
    )
  return out.transpose(1, 2)


def _causal_bias_mask(bias_hss, seq_len, device, dtype=torch.float32):
  """Additive attn_mask (H, S, S): bias + -inf above the diagonal."""
  causal = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
  causal.masked_fill_(
      torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), 1),
      float("-inf"),
  )
  return bias_hss.to(dtype) + causal[None]


class SplashBiasDropoutTest(base_device_test.BaseAcceleratorDeviceTest):

  def _qkv(self, b, s, h, h_kv, d, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, s, h, d, generator=g, dtype=torch.float32).to(device)
    k = torch.randn(b, s, h_kv, d, generator=g, dtype=torch.float32).to(device)
    v = torch.randn(b, s, h_kv, d, generator=g, dtype=torch.float32).to(device)
    return q, k, v

  def _expand_kv(self, k, v, n_heads):
    n_rep = n_heads // k.shape[2]
    return k.repeat_interleave(n_rep, dim=2), v.repeat_interleave(n_rep, dim=2)

  # -------- plain causal attention (no bias, no dropout) --------

  def test_mha_parity(self):
    device = self.accelerator_device
    q, k, v = self._qkv(2, 512, 4, 4, 64, device)
    expected = _sdpa_ref(q, k, v, is_causal=True)
    actual = splash_bias_dropout.sbd_splash_attention(q, k, v)
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=5e-2, atol=5e-2
    )

  def test_gqa_parity(self):
    device = self.accelerator_device
    q, k, v = self._qkv(2, 512, 4, 2, 64, device)
    k_exp, v_exp = self._expand_kv(k, v, 4)
    expected = _sdpa_ref(q, k_exp, v_exp, is_causal=True)
    actual = splash_bias_dropout.sbd_splash_attention(q, k, v)
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=5e-2, atol=5e-2
    )

  def test_mqa_parity(self):
    device = self.accelerator_device
    q, k, v = self._qkv(2, 512, 4, 1, 64, device, seed=29)
    k_exp, v_exp = self._expand_kv(k, v, 4)
    expected = _sdpa_ref(q, k_exp, v_exp, is_causal=True)
    actual = splash_bias_dropout.sbd_splash_attention(q, k, v)
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=5e-2, atol=5e-2
    )

    # Backward: the reference expands KV inside the graph so autograd folds
    # the per-group gradients back onto the single KV head.
    q_ref = q.cpu().clone().requires_grad_(True)
    k_ref = k.cpu().clone().requires_grad_(True)
    v_ref = v.cpu().clone().requires_grad_(True)
    out_ref = _sdpa_ref(
        q_ref,
        k_ref.repeat_interleave(4, dim=2),
        v_ref.repeat_interleave(4, dim=2),
        is_causal=True,
    )
    out_ref.sum().backward()

    q_t = q.clone().requires_grad_(True)
    k_t = k.clone().requires_grad_(True)
    v_t = v.clone().requires_grad_(True)
    out = splash_bias_dropout.sbd_splash_attention(q_t, k_t, v_t)
    out.sum().backward()
    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad),
        ("dk", k_t.grad, k_ref.grad),
        ("dv", v_t.grad, v_ref.grad),
    ):
      self.assertIsNotNone(got, f"{name} is None")
      torch.testing.assert_close(
          got.cpu(), want.cpu(), rtol=8e-2, atol=8e-2, msg=name
      )

  # -------- packed-document segment ids --------

  def _segment_ids_case(self, h_kv, causal, use_window=False):
    """Kernel with segment_ids vs SDPA over the dense document(+causal) mask,
    forward and backward; also checks the ids actually changed the output
    (a vacuous pass -- ids silently dropped -- would still match the
    unmasked reference otherwise)."""
    device = self.accelerator_device
    b, s, h, d = 2, 512, 4, 64
    q, k, v = self._qkv(b, s, h, h_kv, d, device, seed=41)
    segment_ids = torch.zeros(b, s, dtype=torch.int32)
    segment_ids[0, 128:] = 1  # one boundary on a block edge
    segment_ids[1, 40:100] = 1  # three boundaries inside blocks
    segment_ids[1, 100:200] = 2
    segment_ids[1, 200:] = 3
    allowed = segment_ids[:, :, None] == segment_ids[:, None, :]  # (B, Sq, Skv)
    if causal:
      allowed &= torch.tril(torch.ones(s, s, dtype=torch.bool))[None]
    window = 64 if use_window else None
    if use_window:
      i = torch.arange(s)
      allowed &= ((i[:, None] - i[None, :]) <= window) & (i[:, None] >= i[None, :])
    mask = torch.zeros(b, 1, s, s, dtype=torch.float32)
    mask.masked_fill_(~allowed[:, None], float("-inf"))

    def reference(q_, k_, v_):
      k_e, v_e = self._expand_kv(k_, v_, h)
      return _sdpa_ref(q_, k_e, v_e, attn_mask=mask)

    q_ref = q.cpu().clone().requires_grad_(True)
    k_ref = k.cpu().clone().requires_grad_(True)
    v_ref = v.cpu().clone().requires_grad_(True)
    out_ref = reference(q_ref, k_ref, v_ref)
    out_ref.sum().backward()

    q_t = q.clone().requires_grad_(True)
    k_t = k.clone().requires_grad_(True)
    v_t = v.clone().requires_grad_(True)
    out = splash_bias_dropout.sbd_splash_attention(
        q_t, k_t, v_t, segment_ids=segment_ids.to(device), is_causal=causal,
        local_window_size=window,
    )
    out.sum().backward()
    torch.testing.assert_close(out.cpu(), out_ref.cpu(), rtol=5e-2, atol=5e-2)
    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad), ("dk", k_t.grad, k_ref.grad), ("dv", v_t.grad, v_ref.grad),
    ):
      self.assertIsNotNone(got, f"{name} is None")
      torch.testing.assert_close(got.cpu(), want.cpu(), rtol=8e-2, atol=8e-2, msg=name)

    # Guard against a vacuous pass: the same call without ids must differ.
    out_no_ids = splash_bias_dropout.sbd_splash_attention(
        q, k, v, is_causal=causal, local_window_size=window
    )
    self.assertGreater(
        float((out_no_ids.cpu() - out.cpu()).abs().max()), 1e-2,
        "segment_ids had no effect on the output",
    )

  def test_segment_ids_mha_causal(self):
    self._segment_ids_case(h_kv=4, causal=True)

  def test_segment_ids_mha_bidirectional(self):
    self._segment_ids_case(h_kv=4, causal=False)

  def test_segment_ids_gqa_causal(self):
    self._segment_ids_case(h_kv=2, causal=True)

  def test_segment_ids_mqa_causal(self):
    # The kernel tiles kv segment ids differently on its rank-reduced entry.
    self._segment_ids_case(h_kv=1, causal=True)

  def test_segment_ids_window(self):
    self._segment_ids_case(h_kv=4, causal=True, use_window=True)

  def test_segment_ids_with_dropout(self):
    """Dropout on top of segment ids must not disturb the document mask.

    Deterministic form (a mean-over-seeds check is a coin flip at 262k
    elements): the second document's values carry a +1000 offset, so any
    cross-document leakage would show up as a large first-document output;
    with dropout active that output must stay within the undropped,
    unoffset output's range, while within-document outputs do change with
    the seed.
    """
    device = self.accelerator_device
    b, s, h, d = 2, 512, 4, 64
    q, k, v = self._qkv(b, s, h, h, d, device, seed=43)
    segment_ids = torch.zeros(b, s, dtype=torch.int32)
    segment_ids[:, 256:] = 1
    seg = segment_ids.to(device)
    v_offset = v.clone()
    v_offset[:, 256:] += 1000.0  # document 2 only
    base = splash_bias_dropout.sbd_splash_attention(q, k, v, segment_ids=seg)
    splash_bias_dropout.SbdConfig.update(dropout_rate=0.1)
    try:
      out0 = splash_bias_dropout.sbd_splash_attention(q, k, v_offset, segment_ids=seg, dropout_seed=0)
      out1 = splash_bias_dropout.sbd_splash_attention(q, k, v_offset, segment_ids=seg, dropout_seed=1)
    finally:
      splash_bias_dropout.SbdConfig.update(dropout_rate=0.0)
    # Document 1's queries never see document 2's +1000: bounded by the
    # undropped output scaled by the worst-case dropout rescale (1 / 0.9).
    doc1_bound = float(base[:, :256].abs().max()) / 0.9 + 1e-3
    self.assertLess(float(out0[:, :256].abs().max()), doc1_bound, "cross-document leakage under dropout")
    self.assertLess(float(out1[:, :256].abs().max()), doc1_bound, "cross-document leakage under dropout")
    # Dropout is active: a different seed gives a different output.
    self.assertGreater(float((out0 - out1).abs().max()), 1e-3, "seeds gave identical masks")

  def test_noncausal_parity(self):
    device = self.accelerator_device
    q, k, v = self._qkv(2, 512, 4, 4, 64, device, seed=31)
    expected = _sdpa_ref(q, k, v, is_causal=False)
    actual = splash_bias_dropout.sbd_splash_attention(q, k, v, is_causal=False)
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=5e-2, atol=5e-2
    )

    q_ref = q.cpu().clone().requires_grad_(True)
    k_ref = k.cpu().clone().requires_grad_(True)
    v_ref = v.cpu().clone().requires_grad_(True)
    out_ref = _sdpa_ref(q_ref, k_ref, v_ref, is_causal=False)
    out_ref.sum().backward()

    q_t = q.clone().requires_grad_(True)
    k_t = k.clone().requires_grad_(True)
    v_t = v.clone().requires_grad_(True)
    out = splash_bias_dropout.sbd_splash_attention(q_t, k_t, v_t, is_causal=False)
    out.sum().backward()
    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad),
        ("dk", k_t.grad, k_ref.grad),
        ("dv", v_t.grad, v_ref.grad),
    ):
      self.assertIsNotNone(got, f"{name} is None")
      torch.testing.assert_close(
          got.cpu(), want.cpu(), rtol=8e-2, atol=8e-2, msg=name
      )

  def _assert_matches_tokamax(self, q, k, v, name, **kwargs):
    """Forward AND backward bit-exact against the stock tokamax wrapper."""
    q_a = q.clone().requires_grad_(True)
    k_a = k.clone().requires_grad_(True)
    v_a = v.clone().requires_grad_(True)
    out_a = splash_bias_dropout.sbd_splash_attention(q_a, k_a, v_a, **kwargs)
    out_a.sum().backward()

    q_b = q.clone().requires_grad_(True)
    k_b = k.clone().requires_grad_(True)
    v_b = v.clone().requires_grad_(True)
    out_b = tokamax_splash_attention.splash_sdpa(q_b, k_b, v_b, **kwargs)
    out_b.sum().backward()

    torch.testing.assert_close(
        out_a.detach().cpu(), out_b.detach().cpu(), rtol=0.0, atol=0.0, msg=name
    )
    for gname, ga, gb in (
        ("dq", q_a.grad, q_b.grad),
        ("dk", k_a.grad, k_b.grad),
        ("dv", v_a.grad, v_b.grad),
    ):
      self.assertIsNotNone(ga, f"{name}:{gname} is None")
      torch.testing.assert_close(
          ga.cpu(), gb.cpu(), rtol=0.0, atol=0.0, msg=f"{name}:{gname}"
      )

  # TODO(jeremiahhsu): Update tokamax pin. This test requires tokamax pin
  # 8eadbf6cfdd9 otherwise fails with ValueError: too many values to unpack
  # (expected 8)
  @unittest.skip("Requires tokamax pin 8eadbf6cfdd9")
  def test_matches_tokamax(self):
    """The no-regression contract: forward and backward parity with the
    stock tokamax wrapper across its full user surface."""
    device = self.accelerator_device
    q, k, v = self._qkv(2, 512, 4, 4, 64, device, seed=3)
    self._assert_matches_tokamax(q, k, v, "causal", is_causal=True)
    self._assert_matches_tokamax(q, k, v, "scale", is_causal=True, scale=0.05)
    self._assert_matches_tokamax(
        q, k, v, "window", is_causal=True, local_window_size=128
    )
    q1, k1, v1 = self._qkv(2, 512, 4, 1, 64, device, seed=5)
    self._assert_matches_tokamax(q1, k1, v1, "mqa", is_causal=True)

  def test_sliding_window_parity(self):
    device = self.accelerator_device
    s, w = 512, 128
    q, k, v = self._qkv(2, s, 4, 4, 64, device, seed=37)
    i = torch.arange(s)[:, None]
    j = torch.arange(s)[None, :]
    banded = torch.zeros(s, s)
    banded.masked_fill_(~((j <= i) & (j >= i - w)), float("-inf"))

    q_ref = q.cpu().clone().requires_grad_(True)
    k_ref = k.cpu().clone().requires_grad_(True)
    v_ref = v.cpu().clone().requires_grad_(True)
    out_ref = _sdpa_ref(q_ref, k_ref, v_ref, attn_mask=banded)
    out_ref.sum().backward()

    q_t = q.clone().requires_grad_(True)
    k_t = k.clone().requires_grad_(True)
    v_t = v.clone().requires_grad_(True)
    out = splash_bias_dropout.sbd_splash_attention(
        q_t, k_t, v_t, local_window_size=w
    )
    out.sum().backward()

    torch.testing.assert_close(
        out.detach().cpu(), out_ref.detach().cpu(), rtol=5e-2, atol=5e-2
    )
    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad),
        ("dk", k_t.grad, k_ref.grad),
        ("dv", v_t.grad, v_ref.grad),
    ):
      self.assertIsNotNone(got, f"{name} is None")
      torch.testing.assert_close(
          got.cpu(), want.cpu(), rtol=8e-2, atol=8e-2, msg=name
      )

  # ---------------- full-rank bias (+ exact dbias) ----------------

  def test_fullrank_bias_forward_and_backward(self):
    device = self.accelerator_device
    b, s, h, d = 2, 512, 4, 64
    q, k, v = self._qkv(b, s, h, h, d, device, seed=7)
    g = torch.Generator().manual_seed(11)
    bias = (torch.randn(h, s, s, generator=g) * 0.5).to(device)

    # Reference: SDPA with additive causal+bias mask, autograd on CPU.
    q_ref = q.cpu().clone().requires_grad_(True)
    k_ref = k.cpu().clone().requires_grad_(True)
    v_ref = v.cpu().clone().requires_grad_(True)
    bias_ref = bias.cpu().clone().requires_grad_(True)
    mask = _causal_bias_mask(bias_ref, s, torch.device("cpu"))
    out_ref = _sdpa_ref(q_ref, k_ref, v_ref, attn_mask=mask)
    out_ref.sum().backward()

    q_t = q.clone().requires_grad_(True)
    k_t = k.clone().requires_grad_(True)
    v_t = v.clone().requires_grad_(True)
    bias_t = bias.clone().requires_grad_(True)
    out = splash_bias_dropout.sbd_splash_attention(q_t, k_t, v_t, bias_t)
    out.sum().backward()

    torch.testing.assert_close(
        out.detach().cpu(), out_ref.detach().cpu(), rtol=5e-2, atol=5e-2
    )
    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad),
        ("dk", k_t.grad, k_ref.grad),
        ("dv", v_t.grad, v_ref.grad),
        ("dbias", bias_t.grad, bias_ref.grad),
    ):
      self.assertIsNotNone(got, f"{name} is None")
      torch.testing.assert_close(
          got.cpu(), want.cpu(), rtol=8e-2, atol=8e-2, msg=name
      )

  # ---------------- per-key bias (+ trainable rank-reduced dbias) --------

  def test_perkey_bias_dbias(self):
    device = self.accelerator_device
    # Deliberately the SAME shape as the full-rank bias test: the per-key
    # config must retrace via the static args, not via a shape change.
    b, s, h, d = 2, 512, 4, 64
    rows = splash_bias_dropout.PER_KEY_ROWS
    splash_bias_dropout.SbdConfig.update(
        bias_per_key_rows=rows, bias_per_key_dbias=True
    )
    try:
      q, k, v = self._qkv(b, s, h, h, d, device, seed=13)
      g = torch.Generator().manual_seed(17)
      row = (torch.randn(h, s, generator=g) * 0.5).to(device)
      bias_pk = row[:, None, :].expand(h, rows, s).contiguous()

      q_ref = q.cpu().clone().requires_grad_(True)
      k_ref = k.cpu().clone().requires_grad_(True)
      v_ref = v.cpu().clone().requires_grad_(True)
      bias_pk_ref = bias_pk.cpu().clone().requires_grad_(True)
      # The kernel contract: row 0 is the logical bias, broadcast over q.
      mask = _causal_bias_mask(
          bias_pk_ref[:, 0:1, :].expand(h, s, s), s, torch.device("cpu")
      )
      out_ref = _sdpa_ref(q_ref, k_ref, v_ref, attn_mask=mask)
      out_ref.sum().backward()

      q_t = q.clone().requires_grad_(True)
      k_t = k.clone().requires_grad_(True)
      v_t = v.clone().requires_grad_(True)
      bias_t = bias_pk.clone().requires_grad_(True)
      out = splash_bias_dropout.sbd_splash_attention(q_t, k_t, v_t, bias_t)
      out.sum().backward()

      torch.testing.assert_close(
          out.detach().cpu(), out_ref.detach().cpu(), rtol=5e-2, atol=5e-2
      )
      self.assertIsNotNone(bias_t.grad, "per-key dbias is None")
      # Row 0 carries the rank-reduced gradient; pad rows are exactly zero.
      # The reference's autograd concentrates its grad in row 0 as well
      # (rows 1.. are unused by the reference mask).
      torch.testing.assert_close(
          bias_t.grad[:, 0, :].cpu(),
          bias_pk_ref.grad[:, 0, :].cpu(),
          rtol=8e-2,
          atol=8e-2,
      )
      self.assertEqual(
          bias_t.grad[:, 1:, :].abs().sum().item(), 0.0,
          "per-key dbias pad rows must be exactly zero",
      )
    finally:
      splash_bias_dropout.SbdConfig.update(
          bias_per_key_rows=0, bias_per_key_dbias=False
      )

  # ---------------- PRNG dropout ----------------

  def test_dropout_determinism_and_grads(self):
    device = self.accelerator_device
    b, s, h, d = 4, 512, 4, 64
    splash_bias_dropout.SbdConfig.update(dropout_rate=0.1)
    try:
      q, k, v = self._qkv(b, s, h, h, d, device, seed=23)

      out_a = splash_bias_dropout.sbd_splash_attention(
          q, k, v, dropout_seed=1234
      )
      out_b = splash_bias_dropout.sbd_splash_attention(
          q, k, v, dropout_seed=1234
      )
      out_c = splash_bias_dropout.sbd_splash_attention(
          q, k, v, dropout_seed=999
      )
      # Same seed => identical; different seed => different masks.
      torch.testing.assert_close(out_a.cpu(), out_b.cpu(), rtol=0.0, atol=0.0)
      self.assertGreater(
          (out_a - out_c).abs().sum().item(), 0.0,
          "different dropout seeds produced identical outputs",
      )
      # Dropout keeps the output in the no-dropout ballpark (fold-rescale
      # preserves scale): mean magnitude within 20% of the reference.
      ref = _sdpa_ref(q, k, v, is_causal=True)
      self.assertLess(
          (out_a.abs().mean() - ref.abs().mean()).abs().item(),
          0.2 * ref.abs().mean().item(),
      )

      # Gradients flow and are finite through the dropout path.
      q_t = q.clone().requires_grad_(True)
      k_t = k.clone().requires_grad_(True)
      v_t = v.clone().requires_grad_(True)
      out = splash_bias_dropout.sbd_splash_attention(
          q_t, k_t, v_t, dropout_seed=1234
      )
      out.sum().backward()
      for name, grad in (("dq", q_t.grad), ("dk", k_t.grad), ("dv", v_t.grad)):
        self.assertIsNotNone(grad, f"{name} is None")
        self.assertTrue(
            torch.isfinite(grad).all().item(), f"{name} has non-finite values"
        )
        self.assertGreater(grad.abs().sum().item(), 0.0, f"{name} is all zeros")
    finally:
      splash_bias_dropout.SbdConfig.update(dropout_rate=0.0)


if __name__ == "__main__":
  absltest.main()
