"""GPT-OSS on TPU, reusing upstream architecture and checkpoint layout."""

import torchtitan.models.gpt_oss as upstream
from torchtitan.config.override import derive

from .attention import SinkSDPA
from .experts import TpuGptOssGroupedExperts, TpuGptOssMoE
from .infra.parallelize import parallelize_gptoss


def model_registry(flavor: str, *, seq_len: int | None = None, converters=None):
    """Build the SDPA baseline; production shapes remain upstream's shapes."""
    spec = upstream.model_registry(flavor, seq_len=seq_len, attn_backend="sdpa")
    for layer in spec.model.layers:
        layer.moe = derive(layer.moe, TpuGptOssMoE.Config)
        layer.attention.inner_attention = SinkSDPA.Config(
            window_size=layer.attention.sliding_window_size
        )
        layer.moe.routed_experts.inner_experts = derive(
            layer.moe.routed_experts.inner_experts, TpuGptOssGroupedExperts.Config
        )
    if converters:
        from torchtitan.models.utils import validate_converter_order

        validate_converter_order(converters)
        for converter in converters:
            spec.model = converter.build().convert(spec.model)
    spec.parallelize_fn = parallelize_gptoss
    spec.pipelining_fn = None
    return spec


from .config_registry import (
    gpt_oss_debugmodel,
    gpt_oss_debugmodel_graphtrainer,
    gpt_oss_debugmodel_splash,
    gpt_oss_20b,
    gpt_oss_120b,
)

__all__ = [
    "model_registry",
    "gpt_oss_debugmodel",
    "gpt_oss_debugmodel_graphtrainer",
    "gpt_oss_debugmodel_splash",
    "gpt_oss_20b",
    "gpt_oss_120b",
]
