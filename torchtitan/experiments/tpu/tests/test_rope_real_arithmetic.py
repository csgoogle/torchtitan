# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`ComplexRoPE.apply_rotary_emb` is a pure refactor of the complex multiply.

The method is written as an expanded real multiply so that it survives
functional re-tracing inside a `scan` body (see its docstring). That is only
sound if it is bit-identical to the complex form it replaced, which is what
these tests pin -- against a local copy of the original implementation, on
whatever device the suite is pointed at.

    python -m torchtitan.experiments.tpu.tests.test_rope_real_arithmetic
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.models.common import rope


def _apply_rotary_emb_complex(query, key, rope_cache):
  """The original implementation, kept here as the reference."""
  xq_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
  xq_out = torch.view_as_real(xq_ * rope_cache).flatten(3)
  xk_out = torch.view_as_real(xk_ * rope_cache).flatten(3)
  return xq_out.type_as(query), xk_out.type_as(key)


def _make_inputs(dtype, device, head_dim=16):
  generator = torch.Generator().manual_seed(0)
  query = torch.randn(2, 8, 4, head_dim, generator=generator).to(dtype)
  key = torch.randn(2, 8, 2, head_dim, generator=generator).to(dtype)
  # The cache as ComplexRoPE builds it: unit-modulus e^{i theta}, broadcast
  # over the batch and head axes.
  theta = torch.randn(1, 8, 1, head_dim // 2, generator=generator)
  cache = torch.polar(torch.ones_like(theta), theta)
  return query.to(device), key.to(device), cache.to(device)


class RopeRealArithmeticTest(BaseAcceleratorDeviceTest):
  """Bit-for-bit equivalence with the complex form, forward and backward."""

  @parameterized.named_parameters(
      ("float32", torch.float32),
      ("bfloat16", torch.bfloat16),
  )
  def test_forward_is_bit_identical_to_the_complex_form(self, dtype):
    query, key, cache = _make_inputs(dtype, self.accelerator_device)

    want_q, want_k = _apply_rotary_emb_complex(query, key, cache)
    got_q, got_k = rope.ComplexRoPE.apply_rotary_emb(query, key, cache)

    self.assertEqual((got_q.shape, got_q.dtype), (want_q.shape, want_q.dtype))
    self.assertEqual((got_k.shape, got_k.dtype), (want_k.shape, want_k.dtype))
    torch.testing.assert_close(got_q.cpu(), want_q.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(got_k.cpu(), want_k.cpu(), rtol=0, atol=0)

  def test_backward_is_bit_identical_to_the_complex_form(self):
    query, key, cache = _make_inputs(torch.float32, self.accelerator_device)
    grad_q, grad_k = torch.randn_like(query), torch.randn_like(key)

    def grads(fn):
      q = query.detach().clone().requires_grad_(True)
      k = key.detach().clone().requires_grad_(True)
      out_q, out_k = fn(q, k, cache)
      ((out_q * grad_q).sum() + (out_k * grad_k).sum()).backward()
      return q.grad.cpu(), k.grad.cpu()

    want_dq, want_dk = grads(_apply_rotary_emb_complex)
    got_dq, got_dk = grads(rope.ComplexRoPE.apply_rotary_emb)

    torch.testing.assert_close(got_dq, want_dq, rtol=0, atol=0)
    torch.testing.assert_close(got_dk, want_dk, rtol=0, atol=0)

  def test_survives_functional_retracing(self):
    """The reason for the rewrite: no `view_as_complex_copy` on a grad input."""
    query, key, cache = _make_inputs(torch.float32, self.accelerator_device)
    query = query.detach().requires_grad_(True)
    key = key.detach().requires_grad_(True)

    from torch.fx.experimental.proxy_tensor import make_fx  # pylint: disable=g-import-not-at-top

    def loss(q, k):
      out_q, out_k = rope.ComplexRoPE.apply_rotary_emb(q, k, cache)
      return out_q.sum() + out_k.sum()

    graph = make_fx(torch.func.functionalize(loss))(query, key)
    targets = {str(n.target) for n in graph.graph.nodes}
    self.assertNotIn("aten.view_as_complex_copy.default", targets)


if __name__ == "__main__":
  absltest.main()
