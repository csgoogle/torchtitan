# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical parity tests for the chunkwise gated delta rule.

`models.qwen3_5.model._torch_native_gated_delta` steps the recurrence one token
at a time and is the definition of correct behaviour, so it serves as the
oracle here.
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torchtitan.experiments.tpu import gated_delta
from torchtitan.models.qwen3_5 import model as qwen35_model


def _random_inputs(
    batch: int,
    seq_len: int,
    n_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    seed: int = 0,
):
  """Draws inputs in the ranges the model actually produces."""
  generator = torch.Generator().manual_seed(seed)

  def randn(*shape):
    return torch.randn(*shape, generator=generator, dtype=torch.float32)

  q = randn(batch, seq_len, n_heads, key_head_dim)
  k = randn(batch, seq_len, n_heads, key_head_dim)
  v = randn(batch, seq_len, n_heads, value_head_dim)
  # g is a log decay, always non-positive.
  g = -torch.nn.functional.softplus(randn(batch, seq_len, n_heads))
  # beta is an update gate in (0, 1).
  beta = torch.sigmoid(randn(batch, seq_len, n_heads))
  return q, k, v, g, beta


class ChunkedGatedDeltaRuleTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("exact_chunks", 16, 4),
      ("ragged_tail", 19, 8),
      ("single_chunk", 5, 8),
      ("chunk_of_one", 7, 1),
  )
  def test_matches_sequential_reference(self, seq_len: int, chunk_size: int):
    q, k, v, g, beta = _random_inputs(
        batch=2, seq_len=seq_len, n_heads=3, key_head_dim=8, value_head_dim=8
    )

    expected = qwen35_model._torch_native_gated_delta(q, k, v, g, beta)
    actual = gated_delta.chunked_gated_delta_rule(
        q, k, v, g, beta, chunk_size=chunk_size
    )

    self.assertEqual(actual.shape, expected.shape)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

  def test_chunk_size_does_not_change_result(self):
    q, k, v, g, beta = _random_inputs(
        batch=1, seq_len=24, n_heads=2, key_head_dim=8, value_head_dim=16
    )

    reference = gated_delta.chunked_gated_delta_rule(
        q, k, v, g, beta, chunk_size=24
    )
    for chunk_size in (1, 2, 3, 4, 6, 8, 12):
      with self.subTest(chunk_size=chunk_size):
        torch.testing.assert_close(
            gated_delta.chunked_gated_delta_rule(
                q, k, v, g, beta, chunk_size=chunk_size
            ),
            reference,
            rtol=1e-4,
            atol=1e-4,
        )

  def test_strong_decay_is_stable(self):
    """Long chunks with heavy decay must not overflow.

    The naive algebra divides keys by the accumulated decay, which explodes
    here; the implementation is expected to use bounded pairwise exponents
    instead.
    """
    q, k, v, _, beta = _random_inputs(
        batch=1, seq_len=128, n_heads=2, key_head_dim=8, value_head_dim=8
    )
    g = torch.full((1, 128, 2), -8.0)

    actual = gated_delta.chunked_gated_delta_rule(
        q, k, v, g, beta, chunk_size=64
    )

    self.assertTrue(torch.isfinite(actual).all())
    torch.testing.assert_close(
        actual,
        qwen35_model._torch_native_gated_delta(q, k, v, g, beta),
        rtol=1e-4,
        atol=1e-4,
    )

  def test_preserves_input_dtype(self):
    q, k, v, g, beta = _random_inputs(
        batch=1, seq_len=8, n_heads=1, key_head_dim=4, value_head_dim=4
    )

    actual = gated_delta.chunked_gated_delta_rule(
        q.bfloat16(), k.bfloat16(), v.bfloat16(), g, beta, chunk_size=4
    )

    self.assertEqual(actual.dtype, torch.bfloat16)

  def test_grouped_value_heads(self):
    """Value heads may outnumber key heads after the caller's expansion."""
    q, k, v, g, beta = _random_inputs(
        batch=2, seq_len=12, n_heads=4, key_head_dim=8, value_head_dim=16
    )

    torch.testing.assert_close(
        gated_delta.chunked_gated_delta_rule(q, k, v, g, beta, chunk_size=4),
        qwen35_model._torch_native_gated_delta(q, k, v, g, beta),
        rtol=1e-4,
        atol=1e-4,
    )


if __name__ == "__main__":
  absltest.main()
