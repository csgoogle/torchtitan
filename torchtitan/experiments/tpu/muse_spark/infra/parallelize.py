"""Parallelization for Muse Spark models on TPU.

Mirrors ``experiments/tpu/muse_glimmer/infra/parallelize.py``:
AC -> per-block compile -> SimpleFSDP (or FSDP2). The ported model carries no
TP/EP sharding configs yet, so only data parallelism is supported.
"""

import dataclasses

import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.models.common.moe import MoE
from torchtitan.tools.logging import logger


def configure_muse_spark_simple_fsdp_shard_dims(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
) -> list[str]:
  """Shards modules whose dim 0 does not divide the FSDP degree on dim 1.

  At FSDP degrees above the expert count (e.g. 64 experts over 128 v7x
  devices) the grouped expert banks ``[E, F, C]`` and the router gate
  ``[E, D]`` do not divide on dim 0, and the 202048-token vocabulary never
  divides a power-of-two degree. Their dim 1 (a hidden size) does. The rule is
  per module, because ``_fsdp_shard_dim`` applies to all of a module's direct
  parameters.

  Returns:
    FQNs of the modules switched to dim-1 sharding.

  Raises:
    ValueError: If a module divides on neither dim 0 nor dim 1.
  """
  fsdp_degree = parallel_dims.dp_shard * parallel_dims.cp
  if fsdp_degree <= 1:
    return []
  hidden_axis_modules = []
  for module_fqn, module in model.named_modules():
    params = list(module.named_parameters(recurse=False))
    if all(p.shape[0] % fsdp_degree == 0 for _, p in params):
      continue
    incompatible = [
        f"{name}{tuple(p.shape)}"
        for name, p in params
        if p.ndim < 2 or p.shape[1] % fsdp_degree != 0
    ]
    if incompatible:
      raise ValueError(
          f"Cannot FSDP-shard {module_fqn} over {fsdp_degree} ranks on dim 0"
          f" or dim 1: {incompatible}"
      )
    # apply_simple_fsdp's documented per-submodule shard-axis hook.
    setattr(module, "_fsdp_shard_dim", 1)
    hidden_axis_modules.append(module_fqn)
  if hidden_axis_modules:
    logger.info(
        "Using SimpleFSDP dimension 1 for %d uneven Muse Spark modules: %s",
        len(hidden_axis_modules),
        hidden_axis_modules[:8],
    )
  return hidden_axis_modules


def _validate_supported(
    parallel_dims: torchtitan.distributed.ParallelDims,
) -> None:
  for name, enabled in (
      ("Tensor", parallel_dims.tp_enabled),
      ("Expert", parallel_dims.ep_enabled),
      ("Context", parallel_dims.cp_enabled),
      ("Pipeline", parallel_dims.pp_enabled),
  ):
    if enabled:
      raise NotImplementedError(
          f"{name} parallelism is not supported for Muse Spark on TPU yet."
      )


def parallelize_muse_spark(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config: "torchtitan.distributed.activation_checkpoint.ActivationCheckpointingConfig | None",
    dump_folder: str,
    skip_dp: bool = False,
    tpu_compile_mode: str = "layer",
    **kwargs,
) -> nn.Module:
  """Parallelizes a text-only Muse Spark model for TPU.

  Args:
    model: The ``MuseSparkModel`` to parallelize (may be on meta device).
    parallel_dims: Parallel dimensions for this run.
    training: Training config (mixed-precision dtypes, CPU offload).
    parallelism: Parallelism config; ``use_simple_fsdp`` selects SimpleFSDP
      instead of FSDP2.
    compile_config: ``torch.compile`` config.
    ac_config: Activation checkpointing config, or None.
    dump_folder: Dump folder for AC debugging artifacts.
    skip_dp: Skip data parallelism (e.g. for inference).
    tpu_compile_mode: One of ``layer``, ``fullgraph`` or ``aot_fx_trace``.
    **kwargs: Ignored; accepted for signature compatibility with train.py.

  Returns:
    The parallelized model.
  """
  del kwargs  # Unused.
  _validate_supported(parallel_dims)

  effective_compile_config = compile_config
  if compile_config.enable and hasattr(model, "layers"):
    # As in the Qwen3 TPU lane, MoE blocks are not marked spmd_safe: their
    # routed-token counts are data dependent.
    for layer in model.layers.values():
      if not any(isinstance(m, MoE) for m in layer.modules()):
        apply_spmd_safe(layer)
  if tpu_compile_mode in ("fullgraph", "aot_fx_trace") and compile_config.enable:
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  if ac_config is not None:
    ac_config.build(dump_folder=dump_folder).apply(model)

  if (
      effective_compile_config.enable
      and "model" in effective_compile_config.components
  ):
    apply_compile(
        model,
        compile_config=effective_compile_config,
        parallel_dims=parallel_dims,
    )

  if not skip_dp and (
      parallel_dims.dp_shard_enabled or parallel_dims.dp_replicate_enabled
  ):
    if parallelism.use_simple_fsdp:
      configure_muse_spark_simple_fsdp_shard_dims(model, parallel_dims)
      model = apply_simple_fsdp(
          model, parallel_dims=parallel_dims, training=training
      )
    else:
      dp_mesh_names = (
          ["dp_replicate", "fsdp"]
          if parallel_dims.dp_replicate_enabled
          else ["fsdp"]
      )
      apply_fsdp_to_decoder(
          model,
          parallel_dims.get_mesh(dp_mesh_names),
          param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
              training.mixed_precision_param
          ],
          reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
              training.mixed_precision_reduce
          ],
          pp_enabled=parallel_dims.pp_enabled,
          cpu_offload=training.enable_cpu_offload,
          reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
          dp_mesh_dims=None,
          enable_symm_mem=parallelism.enable_fsdp_symm_mem,
      )

  if tpu_compile_mode == "fullgraph" and compile_config.enable:
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")
  return model
