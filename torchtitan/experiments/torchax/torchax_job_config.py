"""Torchax job config."""

import dataclasses

import torchtitan.config
from torchtitan.experiments.jax.jax_job_config import JaxConfig


@dataclasses.dataclass
class TorchaxConfig(JaxConfig):
  """Torchax-specific config. Inherits all JAX/splash-attention fields
  from ``JaxConfig`` and adds torchax-only settings."""

  use_torchax: bool = True
  # separate keys by `|` if there are multiple keys to offload
  offload_keys: str | None = 'decoder_layer_input'


@dataclasses.dataclass
class TorchaxJobConfig(torchtitan.config.JobConfig):
  torchax_config: TorchaxConfig = dataclasses.field(
      default_factory=TorchaxConfig
  )
