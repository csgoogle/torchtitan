"""Torchax job config."""

import dataclasses
import torchtitan.config


@dataclasses.dataclass
class TorchaxConfig:
  use_torchax: bool = True
  use_scan: bool = True
  # Informs torchax training job
  # if megacore is enabled. (Defaults to True, as v4/v5p defaults to using
  # megacore in xpk and launcher. To disable megacore, you must pass
  # --deepsea_chip_config_name=legacy as an XLA flag)
  # TODO(jeremiahhsu): See if we can determine megacore from environment var.
  tpu_megacore: bool = True
  # separate keys by `|` if there are multiple keys to offload
  offload_keys: str | None = 'decoder_layer_input'
  tpu_num_slices: int = 1
  # only for debugging with weaker chips
  model_layer_override: int | None = None


@dataclasses.dataclass
class TorchaxJobConfig(torchtitan.config.JobConfig):
  torchax_config: TorchaxConfig = dataclasses.field(
      default_factory=TorchaxConfig
  )
