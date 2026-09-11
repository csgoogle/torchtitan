"""TPU recipe configs for FLUX.1-dev.

One config per recipe variant under torchtitan/experiments/tpu/recipes,
carrying everything the recipe used to pass on the launcher command line --
including the libtpu init flags it used to export. Each derives from the shared
``flux_dev`` config in torchtitan/models/flux/config_registry.py, so the model,
tokenizer, encoder and validation settings stay in one place.

Recipes train all 12B parameters. The float32 variant keeps master weights in
fp32 with bf16 mixed-precision compute; the bfloat16 variant trains in bf16
throughout and fits a larger batch.
"""

from torchtitan.experiments.tpu import tpu_flags
from torchtitan.models.flux.config_registry import flux_dev
from torchtitan.models.flux.trainer import FluxTrainer


def _flux_dev_tpu(
    *,
    dtype: str,
    local_batch_size: int,
    libtpu_init_flags: tpu_flags.FlagMap,
    enable_tpu_topo_mesh: bool = False,
) -> FluxTrainer.Config:
  """Shared base for the FLUX.1-dev TPU recipes.

  Args:
    dtype: Master-weight dtype, "float32" or "bfloat16".
    local_batch_size: Per-device batch size, set by the memory ceiling of the
      slice and dtype.
    libtpu_init_flags: libtpu init flags for the slice.
    enable_tpu_topo_mesh: Whether to map the device mesh onto the physical ICI
      topology.
  """
  cfg = flux_dev()
  cfg.optimizer.implementation = "foreach"
  cfg.metrics.log_freq = 1
  cfg.training.steps = 40
  cfg.training.dtype = dtype
  cfg.training.local_batch_size = local_batch_size
  cfg.dataloader.img_size = 1024
  # Benchmark on synthetic images and prompts: throughput here is bounded by the
  # model, not by cc12m decode.
  cfg.flux.random_dataset = True
  cfg.parallelism.use_simple_fsdp = True
  cfg.parallelism.enable_tpu_topo_mesh = enable_tpu_topo_mesh
  cfg.splash_attention_kernel.use_splash_attention_kernel = True
  cfg.compile.enable = True
  cfg.compile.backend = "tpu"
  # Recomputation does not need to reproduce the forward pass's RNG draws: the
  # dropout-free Flux blocks under activation checkpointing consume none.
  cfg.activation_checkpoint.preserve_rng_state = False
  cfg.libtpu_init_flags = libtpu_init_flags
  return cfg


def flux_dev_v6e_8_float32() -> FluxTrainer.Config:
  """FLUX.1-dev on v6e-8, fp32 master weights (recipes/trillium/flux1/v6e-8)."""
  return _flux_dev_tpu(
      dtype="float32",
      local_batch_size=2,
      libtpu_init_flags=tpu_flags.merge_flags(
          xla_tpu_scoped_vmem_limit_kib=65536
      ),
  )


def flux_dev_v6e_8_bfloat16() -> FluxTrainer.Config:
  """FLUX.1-dev on v6e-8, bf16 training (recipes/trillium/flux1/v6e-8)."""
  return _flux_dev_tpu(
      dtype="bfloat16",
      local_batch_size=3,
      libtpu_init_flags=tpu_flags.merge_flags(
          xla_tpu_scoped_vmem_limit_kib=65536
      ),
  )


def flux_dev_v6e_256_float32() -> FluxTrainer.Config:
  """FLUX.1-dev on v6e-256, fp32 master weights (recipes/trillium/flux1/v6e-256)."""
  return _flux_dev_tpu(
      dtype="float32",
      local_batch_size=2,
      libtpu_init_flags=tpu_flags.merge_flags(
          tpu_flags.SPARSECORE_COLLECTIVE_OFFLOAD
      ),
  )


def flux_dev_v6e_256_bfloat16() -> FluxTrainer.Config:
  """FLUX.1-dev on v6e-256, bf16 training (recipes/trillium/flux1/v6e-256)."""
  return _flux_dev_tpu(
      dtype="bfloat16",
      local_batch_size=3,
      libtpu_init_flags=tpu_flags.merge_flags(
          tpu_flags.SPARSECORE_COLLECTIVE_OFFLOAD
      ),
  )


def flux_dev_tpu7x_4x4x4_float32() -> FluxTrainer.Config:
  """FLUX.1-dev on tpu7x-4x4x4, fp32 master weights (recipes/ironwood/flux1/tpu7x-4x4x4)."""
  return _flux_dev_tpu(
      dtype="float32",
      local_batch_size=3,
      enable_tpu_topo_mesh=True,
      libtpu_init_flags=tpu_flags.merge_flags(
          xla_tpu_scoped_vmem_limit_kib=65536
      ),
  )


def flux_dev_tpu7x_4x4x4_bfloat16() -> FluxTrainer.Config:
  """FLUX.1-dev on tpu7x-4x4x4, bf16 training (recipes/ironwood/flux1/tpu7x-4x4x4)."""
  return _flux_dev_tpu(
      dtype="bfloat16",
      local_batch_size=5,
      enable_tpu_topo_mesh=True,
      libtpu_init_flags=tpu_flags.merge_flags(
          xla_tpu_scoped_vmem_limit_kib=65536
      ),
  )
