# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fp32 activation evaluation in FeedForward, GELU and SiLU (TPU v7x numerics)."""

import unittest

import torch
import torch.nn.functional as F

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import GELU, SiLU


def _ffn_config(**kwargs) -> FeedForward.Config:
    return FeedForward.Config(
        w1=Linear.Config(in_features=16, out_features=32),
        w2=Linear.Config(in_features=32, out_features=16),
        w3=Linear.Config(in_features=16, out_features=32),
        **kwargs,
    )


class TestFp32Activation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # Large values: bf16 SiLU and fp32 SiLU round differently here.
        self.x = (torch.randn(8, 16) * 40).to(torch.bfloat16)

    def test_default_is_off(self):
        ffn = _ffn_config().build()
        self.assertFalse(ffn.activation_in_fp32)
        self.assertFalse(GELU.Config().build().activation_in_fp32)
        self.assertFalse(SiLU.Config().build().activation_in_fp32)

    def test_feed_forward_swiglu_in_fp32(self):
        ffn = _ffn_config(activation_in_fp32=True).build().to(torch.bfloat16)
        gate, up = ffn.w1(self.x), ffn.w3(self.x)
        expected = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        self.assertEqual(ffn.swiglu(gate, up).dtype, torch.bfloat16)
        self.assertTrue(torch.equal(ffn.swiglu(gate, up), expected))
        self.assertTrue(torch.equal(ffn(self.x), ffn.w2(expected)))

    def test_feed_forward_default_matches_stock(self):
        ffn = _ffn_config().build().to(torch.bfloat16)
        stock = ffn.w2(F.silu(ffn.w1(self.x)) * ffn.w3(self.x))
        self.assertTrue(torch.equal(ffn(self.x), stock))

    def test_activation_modules_in_fp32(self):
        for module, ref in ((GELU, F.gelu), (SiLU, F.silu)):
            act = module.Config(activation_in_fp32=True).build()
            expected = ref(self.x.float()).to(torch.bfloat16)
            self.assertTrue(torch.equal(act(self.x), expected), module.__name__)
            # fp32 inputs are left alone.
            self.assertTrue(torch.equal(act(self.x.float()), ref(self.x.float())))


if __name__ == "__main__":
    unittest.main()
