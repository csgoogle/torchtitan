# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

__all__ = ["FeedForward", "SigmoidGatedFeedForward", "compute_ffn_hidden_dim"]


def compute_ffn_hidden_dim(
    dim: int,
    *,
    multiple_of: int = 1,
    ffn_dim_multiplier: float | None = None,
) -> int:
    """Compute the SwiGLU hidden dimension for Llama3/4-style models.

    This applies the 2/3 scaling, optional multiplier, and rounds up to multiple_of.
    """
    hidden_dim = int(2 * 4 * dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class FeedForward(Module):
    """SwiGLU feed-forward module shared across models.

    Config takes the **final** hidden_dim (no internal 2/3 scaling).
    Use compute_ffn_hidden_dim() for Llama3/4-style dim computation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w1: Linear.Config
        w2: Linear.Config
        w3: Linear.Config
        # Compute SiLU(w1 x) * w3 x in fp32 and cast the product back to the
        # activation dtype; the matmuls keep their dtype. Needed on accelerators
        # that evaluate bf16 SiLU natively in bf16 (TPU v7x): with Qwen3-4B's
        # large early-layer activations the bf16 evaluation raises the
        # hidden-state error of the forward ~3x and the logits error ~5x against
        # an fp32 reference.
        activation_in_fp32: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.w1 = config.w1.build()
        self.w2 = config.w2.build()
        self.w3 = config.w3.build()
        self.activation_in_fp32 = config.activation_in_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.swiglu(self.w1(x), self.w3(x)))

    def swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """``SiLU(gate) * up``, in fp32 when ``activation_in_fp32`` is set."""
        if self.activation_in_fp32:
            return (F.silu(gate.float()) * up.float()).to(gate.dtype)
        return F.silu(gate) * up


class SigmoidGatedFeedForward(FeedForward):
    """SwiGLU feed-forward with a per-token sigmoid gate.

    The output is ``sigmoid(gate(x)) * ffn(x)``. Inherits ``w1/w2/w3`` from
    FeedForward so weight FQNs are flat (no nested ``ffn.`` level), which keeps
    the weights directly shardable.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        gate: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.gate = config.gate.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return torch.sigmoid(self.gate(x)) * out
