"""Parallelization utilities for Conformer on TPU."""

from torch import nn
from torch.distributed import fsdp
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.tpu import tpu_job_config
import torchtitan.experiments.tpu.utils as tpu_utils
from torchtitan.tools.logging import logger


def parallelize_conformer(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: torchtitan.config.JobConfig | tpu_job_config.TPUJobConfig,
) -> nn.Module:
  """Apply parallelism and activation checkpointing to Conformer."""

  if parallel_dims.tp_enabled:
    raise RuntimeError("Tensor Parallelism is not supported for Conformer yet")

  # Apply activation checkpointing if enabled
  if job_config.activation_checkpoint.mode != "none":
    apply_ac(model)

  # Resolve DeviceMesh and mode based on active dimensions
  if parallel_dims.fsdp_enabled:
    if parallel_dims.dp_replicate_enabled:
      dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
      reshard = not parallel_dims.pp_enabled
      dp_mode = "hybrid_shard"
    else:
      dp_mesh = parallel_dims.get_mesh(["fsdp"])
      reshard = not parallel_dims.pp_enabled
      dp_mode = "fully_shard"
  elif parallel_dims.dp_replicate_enabled:
    dp_mesh = parallel_dims.get_mesh("dp_replicate")
    reshard = False
    dp_mode = "replicate"
  else:
    dp_mesh = None
    reshard = False
    dp_mode = None

  fsdp_config = {}
  if dp_mesh:
    fsdp_config["mesh"] = dp_mesh

  enable_amp = True
  if isinstance(job_config, tpu_job_config.TPUJobConfig):
    enable_amp = job_config.tpu_config.enable_amp

  if enable_amp:
    fsdp_config["mp_policy"] = fsdp.MixedPrecisionPolicy(
        param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            job_config.training.mixed_precision_param
        ],
        reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            job_config.training.mixed_precision_reduce
        ],
    )

  if hasattr(model, "model"):
    encoder = model.model
  else:
    encoder = model

  # Shard each individual Conformer layer
  if hasattr(encoder, "layers"):
    layers = encoder.layers
  elif hasattr(encoder, "conformer_layers"):
    layers = encoder.conformer_layers
  else:
    layers = None

  if layers is not None:
    logger.info(f"Sharding Conformer layers (FSDP, mode={dp_mode}).")
    for layer in layers.children():
      fsdp.fully_shard(layer, **fsdp_config, reshard_after_forward=reshard)
  else:
    logger.warning("No layers found to shard.")

  # Shard the full model (if it's a separate module)
  if encoder is not model:
    fsdp.fully_shard(encoder, **fsdp_config, reshard_after_forward=reshard)

  # Shard the outer wrapper
  fsdp.fully_shard(model, **fsdp_config)

  logger.info("Applied FSDP to the model")

  return model


def apply_ac(model: nn.Module) -> None:
  """Apply activation checkpointing."""
  logger.info("Applying activation checkpointing to Conformer layers.")

  if hasattr(model, "model"):
    encoder = model.model
  else:
    encoder = model

  if hasattr(encoder, "layers"):
    layers = encoder.layers
  elif hasattr(encoder, "conformer_layers"):
    layers = encoder.conformer_layers
  else:
    layers = None

  if layers is not None:
    for layer_id, layer in layers.named_children():
      wrapped_layer = tpu_utils.ptd_checkpoint_wrapper_with_early_stop(
          layer, preserve_rng_state=False
      )
      layers.register_module(layer_id, wrapped_layer)
  else:
    logger.warning("No layers found to apply AC.")
