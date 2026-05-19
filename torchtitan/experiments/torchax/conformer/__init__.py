"""Conformer model with torchax."""

from .sharding import sharding_map_original
from .sharding import sharding_map_scan

from .model import Conformer, ConformerModelArgs

args = {
    "test": ConformerModelArgs(
        vocab_size=64,
        hidden_dim=512,
        num_layers=17,
        num_heads=8,
        kernel_size=31,
    )
}

model = Conformer

__all__ = [
    'sharding_map_original',
    'sharding_map_scan',
    'args',
    'model',
]
