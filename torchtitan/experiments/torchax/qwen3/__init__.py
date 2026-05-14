"""qwen3 model with torchax."""

from .sharding import sharding_map_original
from .sharding import sharding_map_scan
from .sharding import sharding_map_scan_moe

from torchtitan.models.qwen3 import Qwen3Model, qwen3_configs

# Materialize upstream factory configs into per-flavor model-args instances.
# Upstream ``qwen3_configs[flavor]`` is a callable ``(attn_backend) -> Config``.
args = {k: v("sdpa") for k, v in qwen3_configs.items()}
# Use the upstream ``Qwen3Model`` directly. A torchax-specific MoE-substitution
# wrapper used to live at ``torchax/qwen3/model.py``, written against the
# pre-restructure upstream API; it now imports broken paths. Re-port that
# wrapper (subclass upstream ``Qwen3TransformerBlock``, swap ``self.moe`` for
# the torchax MoE on a per-layer basis) if torchax-side MoE optimizations are
# needed.
model = Qwen3Model


__all__ = [
    'sharding_map_original',
    'sharding_map_scan',
    'sharding_map_scan_moe',
    'args',
    'model',
]
