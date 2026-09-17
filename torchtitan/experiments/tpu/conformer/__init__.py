"""Registration of Conformer model for TorchTitan."""

from . import model
from .infra.parallelize import parallelize_conformer

__all__ = [
    "conformer_args",
    "parallelize_conformer",
]

conformer_args = {
    "test": model.ConformerModelArgs(
        vocab_size=64,
        hidden_dim=512,
        num_layers=17,
        num_heads=8,
        kernel_size=31,
    ),
    "debugmodel": model.ConformerModelArgs(
        vocab_size=64,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        kernel_size=31,
    ),
}
