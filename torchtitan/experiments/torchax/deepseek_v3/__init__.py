"""DeepSeek-v3 model with torchax."""

from .sharding import sharding_map_original
from .sharding import sharding_map_scan
from .sharding import sharding_map_scan_moe

from torchtitan.models.deepseek_v3 import DeepSeekV3Model, deepseekv3_configs

# Materialize upstream factory configs into per-flavor model-args instances.
# Upstream ``deepseekv3_configs[flavor]`` is a callable
# ``(attn_backend, moe_comm_backend, non_blocking_capacity_factor=None) -> Config``.
args = {k: v("sdpa", "standard") for k, v in deepseekv3_configs.items()}
# Use the upstream ``DeepSeekV3Model`` directly. A torchax-specific
# MoE-substitution wrapper used to live at ``torchax/deepseek_v3/model.py``,
# written against the pre-restructure upstream API; it now imports broken
# paths. Re-port that wrapper if torchax-side MoE optimizations are needed.
model = DeepSeekV3Model


__all__ = [
    'sharding_map_original',
    'sharding_map_scan',
    'sharding_map_scan_moe',
    'args',
    'model',
]
