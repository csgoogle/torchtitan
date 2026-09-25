"""Parallelize an HF backend model for TorchTPU.

Follows the backend's `parallelize_hf_transformers` step for step (un-tie,
MoE swap, Module-protocol conversion, sharding configs, parallelize, AC,
compile) and differs only in data parallelism: SimpleFSDP rather than FSDP2.

That choice is not stylistic. FSDP2 injects its all-gather from a
``forward_pre_hook``, so Dynamo hits a collective inside the hook's own frame,
graph-breaks there, and bails out of the wrapping ``__call__`` entirely --
leaving no compile region and dropping the whole layer body to eager dispatch.
SimpleFSDP registers a parametrization instead, which puts the all-gather on
the ``.weight`` access path, so the break lands *between* the all-gather and
the layer body and the body stays intact. Measured on this lane at 3B:
SimpleFSDP ~30 HLO modules per step, FSDP2 414 with torch.compile disabled
outright, and OOM at the same batch size.

This mirrors `experiments/tpu/llama3/infra/parallelize.py`, which does the
same swap for the torchtitan-native llama3.
"""

import dataclasses

import torch
from torch import nn

import torchtitan.config
import torchtitan.distributed
from torchtitan.distributed.compile import apply_compile
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel as simple_fsdp_data_parallel,
    MixedPrecisionPolicy as SimpleFSDPMixedPrecisionPolicy,
)
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.experiments.transformers_modeling_backend.hf_sharding import (
    set_hf_sharding_configs,
)
from torchtitan.experiments.transformers_modeling_backend.module_conversion import (
    convert_hf_to_module,
)
from torchtitan.experiments.transformers_modeling_backend.parallelize import (
    parallelize_hf_transformers,
)
from torchtitan.tools.logging import logger


def _untie_embeddings(model: nn.Module) -> None:
  """Give lm_head its own copy of a tied embedding weight (backend step 0)."""
  if (
      model.tok_embeddings is not None
      and model.lm_head is not None
      and any(
          p1 is p2
          for p1 in model.tok_embeddings.parameters()
          for p2 in model.lm_head.parameters()
      )
  ):
    model.lm_head.weight = nn.Parameter(
        model.lm_head.weight.clone(),
        requires_grad=model.lm_head.weight.requires_grad,
    )
    logger.info("Un-tied embedding/lm_head weights for FSDP compatibility")


def _raise_recompile_limit(model: nn.Module) -> None:
  """SimpleFSDP + AC give every decoder layer its own wrapper class, so Dynamo
  recompiles the shared checkpoint-wrapper forward once per layer (a type
  guard miss, not a real recompile). The default limit of 8 fails hard under
  fullgraph for any model deeper than 8 layers.
  """
  import torch._dynamo.config as dynamo_config  # pylint: disable=g-import-not-at-top

  wanted = max(64, 2 * len(model.layers) + 16)
  dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, wanted)
  dynamo_config.accumulated_recompile_limit = max(
      dynamo_config.accumulated_recompile_limit, 8 * wanted
  )


def _dp_mesh_names(parallel_dims) -> list[str]:
  """The SimpleFSDP mesh axes.

  The shard axis is "fsdp" under partial_dtensor (what the native llama3 lane
  asks for) but "dp_shard" under spmd_types, which this backend requires. Use
  whichever this ParallelDims actually exposes.
  """
  for shard_axis in ("fsdp", "dp_shard"):
    try:
      parallel_dims.get_mesh([shard_axis])
    except Exception:  # pylint: disable=broad-except
      continue
    if parallel_dims.dp_replicate_enabled:
      return ["dp_replicate", shard_axis]
    return [shard_axis]
  raise ValueError(
      "Neither 'fsdp' nor 'dp_shard' is an activated mesh axis; cannot place"
      " SimpleFSDP."
  )


def parallelize_hf_transformers_tpu(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config,
    dump_folder: str,
    tpu_compile_mode: str = "layer",
    **kwargs,
) -> nn.Module:
  """Apply TP/AC/compile as the backend does, then SimpleFSDP."""
  # TODO b/499075866: re-enable CPU offload for TPU torchtitan run.
  if training.enable_cpu_offload:
    logger.warning(
        "CPU offload is not supported for TPU torchtitan run, setting it to"
        " False."
    )
    training.enable_cpu_offload = False
  if parallel_dims.cp_enabled:
    raise NotImplementedError(
        "Context parallelism in the HF backend relies on flex attention, "
        "which has no TPU lowering."
    )

  compile_mode = tpu_compile_mode
  effective_compile_config = compile_config
  if compile_mode in ("fullgraph", "aot_fx_trace") and compile_config.enable:
    # These modes compile the whole model below, so do not also compile per
    # layer.
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  if not parallelism.use_simple_fsdp:
    # Opt-out path: the backend's own FSDP2 flow, kept so the two
    # data-parallel implementations can be compared on the same config.
    return parallelize_hf_transformers(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=effective_compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
    )

  if parallel_dims.spmd_backend != "spmd_types":
    raise ValueError(
        "The Transformers modeling backend only supports "
        "parallelism.spmd_backend='spmd_types'; "
        f"got '{parallel_dims.spmd_backend}'."
    )

  _untie_embeddings(model)
  if any(getattr(b, "moe_enabled", False) for b in model.layers):
    from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (  # pylint: disable=g-import-not-at-top
        build_and_swap_native_moe,
    )

    build_and_swap_native_moe(model, parallel_dims)
  convert_hf_to_module(model)
  set_hf_sharding_configs(model, enable_sp=parallel_dims.tp_enabled)
  model.parallelize(parallel_dims)

  if ac_config is not None:
    ac_config.build(dump_folder=dump_folder).apply(model)

  if (
      effective_compile_config.enable
      and "model" in effective_compile_config.components
  ):
    for layer in model.layers.values():
      apply_spmd_safe(layer)
    _raise_recompile_limit(model)
    apply_compile(
        model,
        compile_config=effective_compile_config,
        parallel_dims=parallel_dims,
    )

  dp_mesh = parallel_dims.get_mesh(_dp_mesh_names(parallel_dims))
  dp_mode = (
      "hybrid_shard" if parallel_dims.dp_replicate_enabled else "fully_shard"
  )
  mp_policy = SimpleFSDPMixedPrecisionPolicy(
      param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
          training.mixed_precision_param
      ],
      reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
          training.mixed_precision_reduce
      ],
  )
  model = simple_fsdp_data_parallel(
      model, dp_mesh, mode=dp_mode, mp_policy=mp_policy
  )
  logger.info("Applied SimpleFSDP (%s) to the HF backend model", dp_mode)

  if compile_mode == "fullgraph":
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")

  return model
