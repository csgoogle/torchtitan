"""TPU job config."""

import dataclasses
from typing import Literal

import torchtitan.config


@dataclasses.dataclass
class TPUConfig:
  use_simple_fsdp: bool = False
  compile_mode: str = 'layer'
  # Whether to enable LoRA.
  use_lora: bool = False
  lora_rank: int = 16
  lora_alpha: float = 16.0
  lora_dtype: str = 'bfloat16'
  # Whether to force LoRA parameters to use DDP (replicated) instead of FSDP
  # (sharded).
  force_lora_parameter_ddp: bool = True
  apply_rope_complex_workaround: bool = False
  use_loss_kernel: bool = False
  # Chunked linear → softmax → CE loss that avoids materialising the full
  # (B, S, vocab_size) logit tensor. Mutually exclusive with use_loss_kernel.
  # Only implemented for redacted (train_minimal.py).
  use_chunked_loss: bool = False
  use_ctc_loss_for_conformer: bool = False
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
  enable_manual_ddp: bool = False
  log_freq: int = 10
  # Splash attention block sizes for performance optimization.
  sa_block_q: int = 512
  sa_block_kv: int = 512
  sa_block_dkv: int = 512
  sa_block_kv_compute: int = 512
  sa_block_q_dkv: int = 512
  sa_block_kv_dkv: int = 512
  sa_block_kv_dkv_compute: int = 512
  sa_block_q_dq: int = 512
  sa_block_kv_dq: int = 512
  sa_use_fused_bwd_kernel: bool = True
  sa_q_layout: str = 'HEAD_DIM_MINOR'
  sa_k_layout: str = 'HEAD_DIM_MINOR'
  sa_v_layout: str = 'HEAD_DIM_MINOR'
  # Linear softmax cross entropy loss block sizes for performance optimization.
  loss_b_block_size: int = 1024
  loss_h_block_size: int = 512
  loss_v_block_size: int = 2048
  enable_pallas_loss_kernel: bool = True
  eager_mode: Literal[
      "",
      "DEFER_AND_FUSE",
      "INTERNAL_DEFER_ALL",
      "DEFER_NEVER",
      "DEFER_NEVER_AND_LAUNCH_BLOCKING"] = ""


@dataclasses.dataclass
class TPUJobConfig(torchtitan.config.JobConfig):
  tpu_config: TPUConfig = dataclasses.field(default_factory=TPUConfig)

  def __post_init__(self):
    # This prevents creating folder in
    # torchtitan.distributed.utils.init_distributed
    # that is chrashing when running on TAP.
    self.comm.trace_buf_size = 0
