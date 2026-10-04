"""Llama3 model with torchax."""

from .sharding import sharding_map_original
from .sharding import sharding_map_scan
from .sharding import sharding_map_scan_moe

from torchtitan.models.llama3 import Llama3Model, llama3_configs

# Materialize upstream factory configs into per-flavor model-args instances.
# Upstream d771abb0 changed ``llama3_configs[flavor]`` from a callable
# ``(attn_backend) -> Config`` into a ``(builder, max_context_length)`` pair,
# with ``seq_len`` now a required keyword-only argument; each flavor is built at
# its own max context length. torchax overrides flash attention separately (see
# ``splash_attn.declare_splash_attention``), so default to ``"sdpa"`` here.
args = {
    flavor: build_config(attn_backend="sdpa", seq_len=max_context_length)
    for flavor, (build_config, max_context_length) in llama3_configs.items()
}
model = Llama3Model

__all__ = [
    'sharding_map_original',
    'sharding_map_scan',
    'sharding_map_scan_moe',
    'args',
    'model',
]
