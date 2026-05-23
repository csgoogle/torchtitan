"""Conformer model with torchax."""

import torch
from torchtitan.experiments.torchax.conformer import sharding as conformer_sharding
import torchtitan.experiments.tpu.conformer as tpu_conformer
import torchtitan.experiments.tpu.conformer.model as tpu_conformer_model

sharding_map_original = conformer_sharding.sharding_map_original
sharding_map_scan = conformer_sharding.sharding_map_scan


# ==============================================================================
# Torchax Dtype Preservation Utilities
# ==============================================================================

class TorchaxConformer(tpu_conformer_model.Conformer):
  """Conformer model adapted for Torchax via dynamic patching."""

  def forward(self, inputs: torch.Tensor, **kwargs):
    # Clamp inputs to vocab size to support large tokenizers
    inputs = inputs % self.config.vocab_size
    return super().forward(inputs, **kwargs)


# Materialize arguments from module definition (following conventions)
args = tpu_conformer.conformer_args

model = TorchaxConformer

__all__ = [
    "sharding_map_original",
    "sharding_map_scan",
    "args",
    "model",
]
