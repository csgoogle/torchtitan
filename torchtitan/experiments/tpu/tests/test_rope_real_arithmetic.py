# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``workarounds.use_real_rope_arithmetic_patch`` must be a rewrite, not a
new convention.

``ComplexRoPE`` rotates adjacent dimension pairs; ``CosSinRoPE`` rotates
halves. Drifting from one to the other silently changes what an existing
checkpoint means, so compare against the complex multiply itself rather than
against the other RoPE.
"""

import unittest

import torch

from torchtitan.experiments.tpu.workarounds import use_real_rope_arithmetic_patch
from torchtitan.models.common.rope import ComplexRoPE


class TestRealRopeArithmetic(unittest.TestCase):
    def setUp(self):
        # Read out of `__dict__`, not by attribute: `apply_rotary_emb` is a
        # `staticmethod`, and attribute access unwraps it to a plain function
        # that would rebind as an instance method when put back.
        self.unpatched = ComplexRoPE.__dict__["apply_rotary_emb"]
        use_real_rope_arithmetic_patch()
        self.addCleanup(
            setattr, ComplexRoPE, "apply_rotary_emb", self.unpatched
        )

        torch.manual_seed(42)
        head_dim = 64
        shape = (2, 16, 4, head_dim)
        self.xq = torch.randn(*shape)
        self.xk = torch.randn(*shape)
        self.rope_cache = torch.polar(
            torch.ones(16, head_dim // 2), torch.randn(16, head_dim // 2)
        ).view(1, 16, 1, head_dim // 2)

    def _complex_reference(self, x):
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(x_ * self.rope_cache).flatten(3).type_as(x)

    def test_the_patch_is_installed(self):
        self.assertIsNot(
            ComplexRoPE.__dict__["apply_rotary_emb"], self.unpatched
        )

    def test_matches_the_complex_multiply_bit_for_bit(self):
        xq_out, xk_out = ComplexRoPE.apply_rotary_emb(
            self.xq, self.xk, self.rope_cache
        )
        self.assertEqual(
            (xq_out - self._complex_reference(self.xq)).abs().max().item(), 0.0
        )
        self.assertEqual(
            (xk_out - self._complex_reference(self.xk)).abs().max().item(), 0.0
        )

    def test_no_differentiable_complex_view_remains(self):
        """The reason the patch exists: `scan` re-traces its body through
        functionalization, which has no derivative for `view_as_complex_copy`.
        """
        xq = self.xq.clone().requires_grad_(True)
        with torch.utils._python_dispatch._disable_current_modes():
            traced = torch.fx.experimental.proxy_tensor.make_fx(
                lambda q: ComplexRoPE.apply_rotary_emb(q, q, self.rope_cache)[0],
                pre_dispatch=True,
            )(xq)
        targets = {str(n.target) for n in traced.graph.nodes}
        self.assertFalse(
            any("view_as_complex" in t for t in targets),
            f"a complex view survived: {sorted(t for t in targets if 'complex' in t)}",
        )

    def test_output_dtype_and_shape_match_input(self):
        xq = self.xq.bfloat16()
        xq_out, _ = ComplexRoPE.apply_rotary_emb(xq, xq, self.rope_cache)
        self.assertEqual(xq_out.dtype, torch.bfloat16)
        self.assertEqual(xq_out.shape, xq.shape)


if __name__ == "__main__":
    unittest.main()
