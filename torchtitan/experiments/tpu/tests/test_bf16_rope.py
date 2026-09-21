# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the opt-in activation-dtype TPU RoPE override."""

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu.overrides.bf16_rope import (
    TpuActivationDtypeCosSinRoPE,
    tpu_activation_dtype_cossin_rope,
)
from torchtitan.models.common import CosSinRoPE


class TpuActivationDtypeCosSinRoPETest(absltest.TestCase):

  def test_override_preserves_config_and_builds_tpu_rope(self):
    source = CosSinRoPE.Config(dim=8, max_context_length=16, theta=5000.0)
    replacement = tpu_activation_dtype_cossin_rope(source)

    self.assertIsInstance(replacement, TpuActivationDtypeCosSinRoPE.Config)
    self.assertEqual(replacement.dim, source.dim)
    self.assertEqual(
        replacement.max_context_length, source.max_context_length
    )
    self.assertEqual(replacement.theta, source.theta)
    self.assertIsInstance(replacement.build(), TpuActivationDtypeCosSinRoPE)

  def test_forward_matches_explicit_bf16_rotation(self):
    torch.manual_seed(0)
    rope = TpuActivationDtypeCosSinRoPE.Config(
        dim=8, max_context_length=16
    ).build()
    # Upstream d771abb0 folds the batch axis away: RoPE now takes rank-3
    # `[T, H, Dh]` query/key and 1-D `[T]` positions.
    num_tokens = 8
    query = torch.randn(num_tokens, 3, 8, dtype=torch.bfloat16)
    key = torch.randn(num_tokens, 1, 8, dtype=torch.bfloat16)
    positions = torch.arange(4).repeat(2)

    query_out, key_out = rope(query, key, positions)
    cache = rope._reshape_cache(query, positions).bfloat16()
    cos, sin = cache[..., :8], cache[..., 8:]
    expected_query = query * cos + rope._rotate_half(query) * sin
    expected_key = key * cos + rope._rotate_half(key) * sin

    torch.testing.assert_close(query_out, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(key_out, expected_key, rtol=0, atol=0)
    self.assertEqual(query_out.dtype, torch.bfloat16)
    self.assertEqual(key_out.dtype, torch.bfloat16)


if __name__ == "__main__":
  absltest.main()
