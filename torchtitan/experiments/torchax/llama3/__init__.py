"""Llama3 model with torchax."""

from .sharding import sharding_map_original
from .sharding import sharding_map_scan
from .sharding import sharding_map_scan_moe

from torchtitan.models.llama3 import Llama3Model, llama3_configs

# Materialize upstream factory configs into per-flavor model-args instances.
# Upstream ``llama3_configs[flavor]`` is a callable ``(attn_backend) -> Config``;
# torchax overrides flash attention separately (see
# ``splash_attn.declare_splash_attention``), so default to ``"sdpa"`` here.
args = {k: v("sdpa") for k, v in llama3_configs.items()}
model = Llama3Model

__all__ = [
    'sharding_map_original',
    'sharding_map_scan',
    'sharding_map_scan_moe',
    'args',
    'model',
]
