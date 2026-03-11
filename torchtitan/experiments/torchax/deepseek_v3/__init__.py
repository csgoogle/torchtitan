"""qwen3 model with torchax."""

from .sharding import sharding_map_original
from .sharding import sharding_map_scan
from .sharding import sharding_map_scan_moe

import torchtitan.models.deepseek_v3
from .model import DeepSeekV3Model

args = torchtitan.models.deepseek_v3.deepseekv3_args
model = DeepSeekV3Model


__all__ = [
    'sharding_map_original',
    'sharding_map_scan',
    'sharding_map_scan_moe',
    'args',
    'model',
]
