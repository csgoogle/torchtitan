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

  # Splash Attention block sizes
  sa_block_q: int = 1024
  sa_block_kv: int = 512
  sa_block_dkv: int = 512
  sa_block_kv_compute: int = 512
  sa_block_q_dkv: int = 2048
  sa_block_kv_dkv: int = 512
  sa_block_kv_dkv_compute: int = 512
  sa_block_q_dq: int = 2048
  sa_block_kv_dq: int = 512
  sa_use_fused_bwd_kernel: bool = True
  sa_q_layout: str = 'HEAD_DIM_MINOR'
  sa_k_layout: str = 'HEAD_DIM_MINOR'
  sa_v_layout: str = 'HEAD_DIM_MINOR'


@dataclasses.dataclass
class TorchaxJobConfig(torchtitan.config.JobConfig):
  torchax_config: TorchaxConfig = dataclasses.field(
      default_factory=TorchaxConfig
  )
