"""Tests for Tokamax splash attention on TPU."""

from collections.abc import Callable
import functools
import logging

from absl.testing import absltest
from absl.testing import parameterized
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
    # Block sizes reach this kernel through the process-global TokamaxConfig, not
    # through per-call kwargs - splash_sdpa never forwards the per-call values. This
    # test previously passed them per-call, so it ran at the 512 defaults and
    # asserted nothing about block sizes; set them the way the kernel actually reads
    # them, and restore afterwards since the config is process-global.
    cfg = tokamax_splash_attention.TokamaxConfig
    saved = (cfg.block_q, cfg.block_kv, cfg.block_kv_compute)
    self.addCleanup(
        lambda: cfg.update(
            block_q=saved[0], block_kv=saved[1], block_kv_compute=saved[2]
        )
    )
    cfg.update(block_q=128, block_kv=128, block_kv_compute=128)

    splash_fn = functools.partial(
        tokamax_splash_attention.splash_sdpa,
        is_causal=True,
        enable_gqa=False,
    )

    self.check_forward_parity(
        splash_fn=splash_fn,
        b=2,
        n_heads=8,
        n_kv_heads=8,
        seq_len=128,
        head_dim=64,
    )

  @parameterized.named_parameters(
      ("gqa_shapes_flag_false_raises", 8, 2, False, True),
      ("gqa_shapes_flag_true_ok", 8, 2, True, False),
      ("mha_shapes_flag_false_ok", 8, 8, False, False),
      ("mha_shapes_flag_true_raises", 8, 8, True, True),
  )
  def test_enable_gqa_is_validated(self, n_heads, n_kv_heads, flag, should_raise):
    """enable_gqa must be enforced, not ignored.

    The stock splash_sdpa enforces `enable_gqa == (n_heads != n_kv_heads)`, and the
    two kernels are interchangeable via a config flag - so accepting a wrong value
    here would make them disagree on identical input: stock raises, tokamax runs on.
    Shapes are token-major (B, S, H, D); validation happens before any device work.
    """
    b, seq_len, head_dim = 2, 128, 64
    mk = lambda h: torch.zeros(b, seq_len, h, head_dim, dtype=torch.bfloat16)
    q, k, v = mk(n_heads), mk(n_kv_heads), mk(n_kv_heads)

    if should_raise:
      with self.assertRaisesRegex(ValueError, "enable_gqa"):
        tokamax_splash_attention.splash_sdpa(q, k, v, enable_gqa=flag)
    else:
      try:
        tokamax_splash_attention.splash_sdpa(q, k, v, enable_gqa=flag)
      except ValueError as e:
        self.assertNotIn("enable_gqa", str(e))
      except Exception:  # pylint: disable=broad-except
        pass  # got past validation; device/kernel errors are not this test's concern

  def test_custom_scale_backward(self):
    """Regression: dq must carry the chain-rule factor for the custom-scale
    q pre-multiply in the backward (it was off by default_scale/scale)."""
    device = self.accelerator_device
    torch.manual_seed(11)
    b, s, h, d = 2, 128, 4, 64
    scale = 0.05
    q = torch.randn(b, s, h, d, dtype=torch.float32)
    k = torch.randn(b, s, h, d, dtype=torch.float32)
    v = torch.randn(b, s, h, d, dtype=torch.float32)

    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    v_ref = v.clone().requires_grad_(True)
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          q_ref.transpose(1, 2),
          k_ref.transpose(1, 2),
          v_ref.transpose(1, 2),
          is_causal=True,
          scale=scale,
      )
    out_ref.sum().backward()

    q_t = q.to(device).requires_grad_(True)
    k_t = k.to(device).requires_grad_(True)
    v_t = v.to(device).requires_grad_(True)
    out = tokamax_splash_attention.splash_sdpa(
        q_t, k_t, v_t, is_causal=True, scale=scale
    )
    out.sum().backward()

    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad),
        ("dk", k_t.grad, k_ref.grad),
        ("dv", v_t.grad, v_ref.grad),
    ):
      self.assertIsNotNone(got, f"{name} is None")
      torch.testing.assert_close(
          got.cpu(), want, rtol=5e-2, atol=5e-2, msg=name
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

  def _set_tokamax_config(self, **overrides):
    """Set process-global TokamaxConfig fields, restored after the test."""
    cfg = tokamax_splash_attention.TokamaxConfig
    saved = {k: getattr(cfg, k) for k in overrides}
    self.addCleanup(lambda: [setattr(cfg, k, v) for k, v in saved.items()])
    for k, v in overrides.items():
      setattr(cfg, k, v)

  def _check_packed_parity(self, n_kv_heads, segment_ids):
    """Forward and dq/dk/dv match a block-causal SDPA reference."""
    device = self.accelerator_device
    torch.manual_seed(7)
    b, seq_len = segment_ids.shape
    n_heads, head_dim = 8, 64
    n_rep = n_heads // n_kv_heads

    # (B, S, H, D).
    q = torch.randn(b, seq_len, n_heads, head_dim, dtype=torch.float32)
    k = torch.randn(b, seq_len, n_kv_heads, head_dim, dtype=torch.float32)
    v = torch.randn(b, seq_len, n_kv_heads, head_dim, dtype=torch.float32)

    # (B, 1, S, S): same document AND causal.
    document_mask = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    attn_mask = (document_mask & causal_mask).unsqueeze(1)

    q_ref = q.transpose(1, 2).clone().requires_grad_(True)
    k_ref = k.transpose(1, 2).clone().requires_grad_(True)
    v_ref = v.transpose(1, 2).clone().requires_grad_(True)
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          q_ref,
          k_ref.repeat_interleave(n_rep, dim=1),
          v_ref.repeat_interleave(n_rep, dim=1),
          attn_mask=attn_mask,
          is_causal=False,
      )
    out_ref.sum().backward()

    q_t = q.to(device).requires_grad_(True)
    k_t = k.to(device).requires_grad_(True)
    v_t = v.to(device).requires_grad_(True)
    out = tokamax_splash_attention.splash_sdpa(
        q_t,
        k_t,
        v_t,
        segment_ids=segment_ids.to(device),
        is_causal=True,
        enable_gqa=n_kv_heads != n_heads,
    )
    out.sum().backward()

    torch.testing.assert_close(
        out.detach().cpu(), out_ref.detach().transpose(1, 2), rtol=5e-2, atol=5e-2
    )
    for name, got, want in (
        ("dq", q_t.grad, q_ref.grad),
        ("dk", k_t.grad, k_ref.grad),
        ("dv", v_t.grad, v_ref.grad),
    ):
      self.assertIsNotNone(got, f"{name} is None")
      torch.testing.assert_close(
          got.cpu(), want.transpose(1, 2), rtol=5e-2, atol=5e-2, msg=name
      )

    # Guard against a vacuous pass: without segment_ids the kernel must differ
    # from the packed reference.
    out_unpacked = tokamax_splash_attention.splash_sdpa(
        q_t.detach(),
        k_t.detach(),
        v_t.detach(),
        is_causal=True,
        enable_gqa=n_kv_heads != n_heads,
    )
    self.assertFalse(
        torch.allclose(
            out_unpacked.cpu(),
            out_ref.detach().transpose(1, 2),
            rtol=5e-2,
            atol=5e-2,
        )
    )

  @parameterized.named_parameters(
      ("mha_scan_bwd", 8, False),
      ("mha_vmap_bwd", 8, True),
      ("gqa_scan_bwd", 2, False),
      ("gqa_vmap_bwd", 2, True),
      ("mqa_scan_bwd", 1, False),
      ("mqa_vmap_bwd", 1, True),
  )
  def test_segment_ids_parity(self, n_kv_heads, use_vmap_bwd):
    """Batch-native masking: packed documents via segment IDs, no skipping."""
    seq_len = 256
    self._set_tokamax_config(
        use_vmap_bwd=use_vmap_bwd, use_document_aware_block_skipping=False
    )
    # Row 0 packs two documents, row 1 packs four with a boundary inside a
    # 128-token block, so a kernel that only honored the static causal mask
    # would visibly disagree with the reference.
    segment_ids = torch.zeros(2, seq_len, dtype=torch.int32)
    segment_ids[0, 128:] = 1
    segment_ids[1, 40:100] = 1
    segment_ids[1, 100:200] = 2
    segment_ids[1, 200:] = 3
    self._check_packed_parity(n_kv_heads, segment_ids)

  @parameterized.named_parameters(
      ("mha_scan_bwd", 8, False),
      ("mha_vmap_bwd", 8, True),
      ("gqa_scan_bwd", 2, False),
      ("gqa_vmap_bwd", 2, True),
      ("mqa_scan_bwd", 1, False),
      ("mqa_vmap_bwd", 1, True),
  )
  def test_document_aware_block_skipping_parity(self, n_kv_heads, use_vmap_bwd):
    """Skipping cross-document tiles leaves the result exact."""
    # 128-token tiles over 512 tokens give several skippable tiles per row.
    seq_len = 512
    self._set_tokamax_config(
        use_vmap_bwd=use_vmap_bwd,
        use_document_aware_block_skipping=True,
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
    )
    segment_ids = torch.zeros(2, seq_len, dtype=torch.int32)
    segment_ids[0, 128:] = 1
    segment_ids[0, 300:] = 2
    segment_ids[1, 40:] = 1
    segment_ids[1, 100:] = 2
    segment_ids[1, 200:] = 3
    segment_ids[1, 420:] = 4
    self._check_packed_parity(n_kv_heads, segment_ids)

  def test_document_aware_mask_info_drops_cross_document_tiles(self):
    """The expected tiles are dropped from both tokamax MaskInfo forms."""
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

    kernel_lib = tokamax_splash_attention.splash_attention_kernel
    seq_len, block = 512, 128
    kernel = kernel_lib.make_splash_mha_single_device(
        mask=tokamax_splash_attention.splash_attention_mask.CausalMask(
            (seq_len, seq_len)
        ),
        config=kernel_lib.SplashConfig(
            block_q=block,
            block_kv=block,
            block_kv_compute=block,
            block_q_dkv=block,
            block_kv_dkv=block,
            block_kv_dkv_compute=block,
            use_fused_bwd_kernel=True,
        ),
    )
    # Blocks hold documents {0}, {1}, {1, 2}, {2}, so of the 10 causal
    # (q, kv) tiles, (1,0), (2,0), (3,0), (3,1) are entirely cross-document.
    segment_ids = jnp.concatenate([
        jnp.zeros(128, jnp.int32),
        jnp.ones(172, jnp.int32),
        jnp.full(212, 2, jnp.int32),
    ])
    expected = [(0, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)]

    def active_tiles(mask_info, is_dkv):
      if mask_info.active_rows is not None:
        n = int(mask_info.num_active_blocks[0])
        rows = [int(r) for r in mask_info.active_rows]
        cols = [int(c) for c in mask_info.active_cols]
        self.assertTrue(all(r == -1 for r in rows[n:]), rows)
        pairs = list(zip(rows[:n], cols[:n]))
        # dkv lists are kv-major: (kv, q).
        return sorted((c, r) for r, c in pairs) if is_dkv else pairs
      grid = jnp.asarray(mask_info.block_mask).reshape(4, 4)
      if is_dkv:
        grid = grid.T
      return [(i, j) for i in range(4) for j in range(4) if int(grid[i, j])]

    for name, mask_info, is_dkv in (
        ("fwd", kernel.fwd_mask_info, False),
        ("dkv", kernel.dkv_mask_info, True),
    ):
      got = active_tiles(
          tokamax_splash_attention._document_aware_mask_info(
              mask_info, segment_ids, block, block, is_dkv
          ),
          is_dkv,
      )
      self.assertEqual(sorted(got), expected, name)
      # Kept tiles stay in the kernel's original order.
      if not is_dkv and mask_info.active_rows is not None:
        self.assertEqual(got, expected, name)

  def test_config_change_retraces_kernel(self):
    """A TokamaxConfig change must not reuse a kernel compiled for the old one.

    jax_op caches compiled kernels by argument values, so the config has to be
    passed as op arguments; read inside the op, two calls with equal shapes
    would share one compiled kernel whatever the config said.
    """
    from unittest import mock  # pylint: disable=g-import-not-at-top

    device = self.accelerator_device
    cfg = tokamax_splash_attention.TokamaxConfig
    saved = cfg.use_vmap_bwd
    self.addCleanup(setattr, cfg, "use_vmap_bwd", saved)

    # A shape no other test uses, so the first call below must trace too.
    q, k, v = (
        torch.randn(1, 1024, 2, 64, device=device).requires_grad_(True)
        for _ in range(3)
    )
    real = tokamax_splash_attention._make_splash_attention_fn
    seen = []

    def record(*args, **kwargs):
      seen.append(kwargs["use_vmap_bwd"])
      return real(*args, **kwargs)

    with mock.patch.object(
        tokamax_splash_attention, "_make_splash_attention_fn", record
    ):
      for use_vmap_bwd in (False, True):
        cfg.use_vmap_bwd = use_vmap_bwd
        tokamax_splash_attention.splash_sdpa(q, k, v).sum().backward()

    # fwd + bwd each traced once per config value.
    self.assertEqual(sorted(seen), [False, False, True, True])


if __name__ == "__main__":
  absltest.main()
