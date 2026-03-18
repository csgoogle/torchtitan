"""TPU job config."""

import dataclasses
import torchtitan.config


@dataclasses.dataclass
class TPUConfig:
  use_fairscale: bool = False
  apply_rope_complex_workaround: bool = False
  use_loss_kernel: bool = False
  use_splash_attention_kernel: bool = False


@dataclasses.dataclass
class TPUJobConfig(torchtitan.config.JobConfig):
  tpu_config: TPUConfig = dataclasses.field(default_factory=TPUConfig)


def use_fairscale(
    job_config: torchtitan.config.JobConfig | TPUJobConfig,
) -> bool:
  """Returns whether fairscale is enabled for the given job config."""
  if isinstance(job_config, TPUJobConfig):
    return job_config.tpu_config.use_fairscale
  return False
