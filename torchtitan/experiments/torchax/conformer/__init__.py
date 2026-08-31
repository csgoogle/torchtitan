"""Conformer model with torchax."""

from torchtitan.experiments.torchax.conformer import sharding as conformer_sharding
import torchtitan.experiments.tpu.conformer as tpu_conformer
import torchtitan.experiments.tpu.conformer.model as tpu_conformer_model

sharding_map_original = conformer_sharding.sharding_map_original
sharding_map_scan = conformer_sharding.sharding_map_scan


# Materialize arguments from module definition (following conventions)
args = tpu_conformer.conformer_args

model = tpu_conformer_model.Conformer

__all__ = [
    "sharding_map_original",
    "sharding_map_scan",
    "args",
    "model",
]
