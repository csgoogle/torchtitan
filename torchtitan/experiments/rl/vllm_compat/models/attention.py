# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
import torch


class VLLMCompatibleFlashAttention(torch.nn.Module):
    """Wrapper around FlashAttention. Optimized for TPU via native SDPA."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass using native PyTorch SDPA.
        
        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_kv_heads, seq_len, head_dim)
            v: Value tensor (batch, num_kv_heads, seq_len, head_dim)
            scale: Softmax scale factor
        """
        # TorchTitan models typically pass: (batch, num_heads, seq_len, head_dim)
        # SDPA expects the same. No transpose needed if already in this format.
        
        # Handle Grouped Query Attention (GQA) by repeating K/V heads to match Q
        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        if num_heads != num_kv_heads:
            num_groups = num_heads // num_kv_heads
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)
        
        if scale is None:
            scale = 1.0 / math.sqrt(q.size(-1))

        # Use PyTorch native SDPA which is fully supported and optimized on TPU
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale
        )
