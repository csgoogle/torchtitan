"""JAX experiment job config."""

import dataclasses
import torchtitan.config


@dataclasses.dataclass
class JaxConfig:
    use_scan: bool = True
    tpu_megacore: bool = True
    tpu_num_slices: int = 1
    model_layer_override: int | None = None

    # Splash Attention. Enabled by default on TPU; falls back to standard
    # scaled-dot-product attention on other platforms or when set to False.
    use_splash_attention_kernel: bool = True

    # Splash Attention block sizes (same defaults as torchax)
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

    # When True, replace optax.adamw with a custom Adam that stores both mu
    # and nu in bf16 (upcast to fp32 per-leaf inside the update math). Useful
    # on fp32 master weights + memory-tight configs (e.g. Llama3 8B on v6e-4
    # where fp32 nu is 3.5 GiB/chip for the stacked MLP kernel alone). Off
    # by default; stock optax adamw is used when False.
    adamw_bf16_state: bool = False

    # redacted-specific: when True, wrap the compute_loss chunked-CE scan body
    # in ``jax.checkpoint`` so backward recomputes per-chunk logits instead
    # of saving ``[n_chunks, chunk_size, V]`` all at once. Required at
    # larger batch sizes where the materialised logits would OOM
    # (e.g. B=16 S=8192 V=153600 bf16 ≈ 40 GiB). Off by default; opt in
    # only when you hit that allocation.
    redacted_remat_chunks: bool = False


@dataclasses.dataclass
class JaxJobConfig(torchtitan.config.JobConfig):
    jax_config: JaxConfig = dataclasses.field(default_factory=JaxConfig)
