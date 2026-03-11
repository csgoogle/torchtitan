"""qwen3 model with torchax."""

from .sharding import sharding_map_original
from .sharding import sharding_map_scan
from .sharding import sharding_map_scan_moe

import torchtitan.models.qwen3
from .model import Qwen3Model

args = torchtitan.models.qwen3.qwen3_args
model = Qwen3Model


__all__ = [
    'sharding_map_original',
    'sharding_map_scan',
    'sharding_map_scan_moe',
    'args',
    'model',
]
