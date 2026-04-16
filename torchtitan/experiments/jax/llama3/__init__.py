"""Llama3 model for JAX experiment."""

from .model import ModelArgs, Transformer
from .model import llama3_8b, llama3_70b
from .sharding import sharding_map_original, sharding_map_scan

args = {
    '8B': llama3_8b,
    '70B': llama3_70b,
}

__all__ = [
    'ModelArgs',
    'Transformer',
    'args',
    'sharding_map_original',
    'sharding_map_scan',
]
