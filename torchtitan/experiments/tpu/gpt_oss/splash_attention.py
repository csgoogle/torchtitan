"""Opt-in GPT-OSS attention adapter for Tokamax's native sink support."""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor

from torchtitan.config import derive, override
from torchtitan.experiments.tpu.muse_glimmer.attention import _unflatten_to_4d
from torchtitan.models.common.attention import segment_ids_from_positions
from torchtitan.models.gpt_oss.model import Attention


class SinkSplashAttention(Attention):
    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        splash_block_size: int = 128

    def __init__(self, config):
        super().__init__(config)
        self.window_size = config.sliding_window_size
        self.splash_block_size = config.splash_block_size

    def forward(self, x, attention_masks, positions=None, segment_ids=None):
        # Same projections/RoPE as upstream; pass sinks explicitly instead of
        # the output-rescaling callback so the kernel owns their gradient.
        if attention_masks is not None:
            raise ValueError("GPT-OSS Splash requires position-based causal masks")
        from .splash import sink_splash

        q, k, v = self.qkv_linear(x)
        q, k = self.rope(q, k, positions)
        q, k, v, _ = _unflatten_to_4d(q, k, v)
        if segment_ids is None:
            segment_ids = (
                segment_ids_from_positions(positions)
                if positions is not None
                else torch.zeros(q.shape[:2], dtype=torch.int32, device=q.device)
            )
        segment_ids = segment_ids.reshape(q.shape[:2]).to(torch.int32)
        sinks = self.sinks
        if isinstance(sinks, DTensor):
            sinks = sinks.to_local(grad_placements=sinks.placements)
        out, _ = sink_splash(
            q,
            k,
            v,
            sinks,
            segment_ids,
            -1 if self.window_size is None else self.window_size,
            self.softmax_scale,
            self.splash_block_size,
        )
        return self.wo(out.reshape(x.shape[0], -1).contiguous())


@override(
    target=Attention.Config,
    exact=True,
    description="Use Tokamax attention with native GPT-OSS sinks and packed masks",
)
def use_sink_splash(config: Attention.Config) -> SinkSplashAttention.Config:
    return derive(config, SinkSplashAttention.Config)
