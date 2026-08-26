# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for torchtitan.experiments.tpu.rope_real_arithmetic.

The real-arithmetic rewrite exists only to remove a differentiable
`view_as_complex` from the scan combine_fn (see the module docstring). It must
therefore be a *pure* refactor: same values, same gradients. These tests pin
that, on CPU, so a regression is caught without a TPU.
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torchtitan.experiments.tpu import rope_real_arithmetic
from torchtitan.models.common import rope


def _make_inputs(dtype, batch=2, seqlen=8, n_heads=4, n_kv_heads=2, head_dim=16):
  generator = torch.Generator().manual_seed(0)
  query = torch.randn(
      batch, seqlen, n_heads, head_dim, generator=generator, dtype=torch.float32
  ).to(dtype)
  key = torch.randn(
      batch, seqlen, n_kv_heads, head_dim, generator=generator,
      dtype=torch.float32,
  ).to(dtype)
  # The cache as ComplexRoPE builds it: unit-modulus e^{i theta}, broadcast over
  # the batch and head axes.
  theta = torch.randn(1, seqlen, 1, head_dim // 2, generator=generator)
  cache = torch.polar(torch.ones_like(theta), theta)
  return query, key, cache


class RopeRealArithmeticTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("float32", torch.float32),
      ("bfloat16", torch.bfloat16),
  )
  def test_matches_complex_form(self, dtype):
    query, key, cache = _make_inputs(dtype)

    want_q, want_k = rope.ComplexRoPE.apply_rotary_emb(query, key, cache)
    got_q, got_k = rope_real_arithmetic._apply_rotary_emb_real(query, key, cache)  # pylint: disable=protected-access

    self.assertEqual(got_q.shape, want_q.shape)
    self.assertEqual(got_k.shape, want_k.shape)
    self.assertEqual(got_q.dtype, want_q.dtype)
    self.assertEqual(got_k.dtype, want_k.dtype)
    torch.testing.assert_close(got_q, want_q, rtol=0, atol=0)
    torch.testing.assert_close(got_k, want_k, rtol=0, atol=0)

  def test_matches_complex_form_in_backward(self):
    query, key, cache = _make_inputs(torch.float32)
    grad_q = torch.randn_like(query)
    grad_k = torch.randn_like(key)

    def grads(fn):
      q = query.detach().clone().requires_grad_(True)
      k = key.detach().clone().requires_grad_(True)
      out_q, out_k = fn(q, k, cache)
      (out_q * grad_q).sum().add_((out_k * grad_k).sum()).backward()
      return q.grad, k.grad

    want_dq, want_dk = grads(rope.ComplexRoPE.apply_rotary_emb)
    got_dq, got_dk = grads(rope_real_arithmetic._apply_rotary_emb_real)  # pylint: disable=protected-access

    torch.testing.assert_close(got_dq, want_dq, rtol=0, atol=0)
    torch.testing.assert_close(got_dk, want_dk, rtol=0, atol=0)

  def test_patch_is_idempotent_and_installs(self):
    original = rope.ComplexRoPE.apply_rotary_emb
    try:
      self.assertTrue(rope_real_arithmetic.use_real_arithmetic_rope_patch())
      patched = rope.ComplexRoPE.apply_rotary_emb
      self.assertIs(patched, rope_real_arithmetic._apply_rotary_emb_real)  # pylint: disable=protected-access
      self.assertTrue(rope_real_arithmetic.use_real_arithmetic_rope_patch())
      self.assertIs(rope.ComplexRoPE.apply_rotary_emb, patched)
    finally:
      rope.ComplexRoPE.apply_rotary_emb = original
      rope_real_arithmetic._APPLIED = False  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
