"""TPU job config."""

import dataclasses
import torchtitan.config


@dataclasses.dataclass
class TPUConfig:
  use_fairscale: bool = False
  apply_rope_complex_workaround: bool = False
  use_loss_kernel: bool = False
  # Chunked linear → softmax → CE loss that avoids materialising the full
  # (B, S, vocab_size) logit tensor. Mutually exclusive with use_loss_kernel.
  # Only implemented for redacted (train_minimal.py).
  use_chunked_loss: bool = False
  use_splash_attention_kernel: bool = False
  use_gmm_kernel: bool = False
  use_fill_indices_kernel: bool = False
  use_jax_profiler: bool = False
  # Insert a graph-split synchronization point between the forward+loss pass
  # and the backward pass. This compiles two smaller XLA graphs instead of one
  # monolithic graph, significantly reducing first-step compilation time.
  use_graph_split: bool = False
  # Disable Automatic Mixed Precision (AMP), so all training is done in uniform
  # precision.
  enable_amp: bool = True


@dataclasses.dataclass
class TPUJobConfig(torchtitan.config.JobConfig):
  tpu_config: TPUConfig = dataclasses.field(default_factory=TPUConfig)

  def __post_init__(self):
    # This prevents creating folder in
    # torchtitan.distributed.utils.init_distributed
    # that is chrashing when running on TAP.
    self.comm.trace_buf_size = 0


def use_fairscale(
    job_config: torchtitan.config.JobConfig | TPUJobConfig,
) -> bool:
  """Returns whether fairscale is enabled for the given job config."""
  if isinstance(job_config, TPUJobConfig):
    return job_config.tpu_config.use_fairscale
  return False
