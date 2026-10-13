"""SDPA baseline for GPT-OSS's learned sinks and packed sliding windows."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from torchtitan.experiments.tpu.muse_glimmer.attention import (
    packed_sliding_window_causal_mask,
)
from torchtitan.experiments.tpu.unflatten_shim import unflatten_tensor
from torchtitan.models.common.attention import (
    ScaledDotProductInnerAttention,
    segment_ids_from_positions,
)


class SinkSDPA(ScaledDotProductInnerAttention):
    """Honor upstream's sink callback, including its LSE gradient.

    SDPA does not expose differentiable logsumexp. This correctness baseline
    computes it separately and uses upstream's sink rescaling callback. Keeping
    the class name distinct from the generic Splash patch's match prevents a
    kernel without sink support from silently replacing this implementation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ScaledDotProductInnerAttention.Config):
        window_size: int | None = None

    def __init__(self, config: Config):
        # Do not probe TPU availability while building a CPU reference model.
        # The dispatcher selects the appropriate enabled backend from q.device.
        torch.nn.Module.__init__(self)
        self.sdpa_backends = [SDPBackend.OVERRIDEABLE, SDPBackend.MATH]
        if config.window_size is not None and config.window_size < 1:
            raise ValueError("window_size must be positive or None")
        self.window_size = config.window_size

    def forward(
        self,
        q,
        k,
        v,
        *,
        attention_masks=None,
        scale=None,
        enable_gqa=False,
        is_causal=True,
        positions=None,
        segment_ids=None,
        out_transform=None,
        **kwargs,
    ):
        if attention_masks is not None or not is_causal:
            raise ValueError("GPT-OSS SDPA requires causal position-based masking")
        if out_transform is None:
            raise ValueError("GPT-OSS SDPA requires the learned-sink callback")
        if segment_ids is None and positions is not None:
            segment_ids = segment_ids_from_positions(positions)
        folded = q.ndim == 3
        if folded:
            q, k, v = (unflatten_tensor(t) for t in (q, k, v))
            if q.ndim == 3:
                q, k, v = (t.unsqueeze(0) for t in (q, k, v))
        batch, length = q.shape[:2]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        mask = packed_sliding_window_causal_mask(
            length,
            self.window_size,
            q.device,
            segment_ids_BL=(
                None if segment_ids is None else segment_ids.reshape(batch, length)
            ),
        )
        scale = q.shape[-1] ** -0.5 if scale is None else scale
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        # Match upstream Flex/Varlen's natural-log LSE, in [T, H] order.
        lse_k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        # Explicit casts alone do not prevent autocast from rounding QK back
        # to bf16. The sink normalizer must retain the fp32 score accumulator.
        with torch.autocast(device_type=q.device.type, enabled=False):
            scores = (q.float() @ lse_k.float().transpose(-2, -1)) * scale
            lse = scores.masked_fill(~mask, float("-inf")).logsumexp(-1)
        out = out.transpose(1, 2)
        lse = lse.transpose(1, 2)
        if folded:
            out = out.reshape(-1, *out.shape[2:])
            lse = lse.reshape(-1, lse.shape[-1])
        return out_transform(out, lse)
