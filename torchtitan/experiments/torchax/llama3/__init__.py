"""Llama3 model with torchax."""

from .sharding import sharding_map_original
from .sharding import sharding_map_scan
from .sharding import sharding_map_scan_moe

import torchtitan.models.llama3

args = torchtitan.models.llama3.llama3_args
model = torchtitan.models.llama3.model.model.Transformer

__all__ = [
    'sharding_map_original',
    'sharding_map_scan',
    'sharding_map_scan_moe',
    'args',
    'model',
]
