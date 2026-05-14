"""Torchax job config."""

import dataclasses

from torchtitan.experiments.jax.jax_job_config import JaxConfig
from torchtitan.experiments.tpu.tpu_job_config import BaseTPUTrainerConfig


@dataclasses.dataclass
class TorchaxConfig(JaxConfig):
  """Torchax-specific runtime config."""

  use_torchax: bool = True
  # separate keys by `|` if there are multiple keys to offload
  offload_keys: str | None = 'decoder_layer_input'
  # For LoRA flavors: replicate the *base model* weights on every chip
  # instead of sharding them on the fsdp axis. This removes the per-layer
  # all-gather (~5% of step on redacted 3B on v6e-4) at the cost of holding
  # the full base model in each chip's HBM (6 GB for redacted 3B at bf16).
  # LoRA adapters are already replicated (see sharding_map_scan_lora);
  # this flag extends the "DDP" treatment to the base model as well.
  use_ddp_sharding: bool = False
  # Compile the forward+backward and the optimizer update as *two separate*
  # XLA programs instead of one monolithic jit over the whole training step.
  # This mirrors torchtpu's default where `torch.compile(model)` compiles
  # just the model graph and the optimizer runs outside compile. Useful for
  # inspection / comparing HLO module sizes between frameworks.
  split_compile: bool = False


@dataclasses.dataclass(kw_only=True, slots=True)
class TorchaxJobConfig(BaseTPUTrainerConfig):
  torchax_config: TorchaxConfig = dataclasses.field(
      default_factory=TorchaxConfig
  )
