"""Parallelization utilities for Llama models on TPU.

This module is used in distributed training tests configured with
`model_name="llama3_tpu".
"""

import dataclasses
import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
import torchtitan.distributed.activation_checkpoint
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    MixedPrecisionPolicy as SimpleFSDPMixedPrecisionPolicy,
    data_parallel as simple_fsdp_data_parallel,
)
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.models.llama3 import parallelize as dtensor_parallelize
from torchtitan.tools.logging import logger


def parallelize_llama(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config: "torchtitan.distributed.activation_checkpoint.ActivationCheckpointing.Config",
    dump_folder: str,
    skip_dp: bool = False,
    tpu_compile_mode: str = "layer",
    tpu_use_scan: bool = False,
    tpu_scan_peel_first_layer: bool = False,
    **kwargs,
) -> nn.Module:
  """Parallelizes the Llama model based on the provided configuration and applies TPU-specific workarounds.

  Args:
    model: The nn.Module to be parallelized.
    parallel_dims: Contains the world mesh and other parallelization dimensions.
    training: Training configuration.
    parallelism: Parallelism configuration.
    compile_config: Compile configuration.
    ac_config: Activation checkpointing configuration.
    dump_folder: Folder for dumped artifacts.
    skip_dp: Whether to skip data parallelism wrapping.
    tpu_compile_mode: TPU compile mode ('layer' or 'fullgraph').
    tpu_use_scan: Replace the decoder's per-layer Python loop with a single
      `scan` HOP, so the N identical blocks compile to one `stablehlo.while`.
    tpu_scan_peel_first_layer: Run layer 0 outside the scan. Required when the
      block calls a JAX-exported kernel (splash attention); see
      `scan_model.ScannedLayerStack.forward`.

  Raises:
    ValueError: if `tpu_use_scan` is set outside the configuration it is known
      to be correct in.
  """
  # TODO b/499075866: re-enable CPU offload for TPU torchtitan run.
  if training.enable_cpu_offload:
    logger.warning(
        "CPU offload is not supported for TPU torchtitan run, setting it to False."
    )
    training.enable_cpu_offload = False

  compile_mode = tpu_compile_mode
  effective_compile_config = compile_config
  scanned_layers = None
  if compile_mode in ("fullgraph", "aot_fx_trace") and compile_config.enable:
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  if tpu_use_scan:
    # Must run before anything else rewrites `model.layers`: AC wrapping and
    # FSDP both iterate the ModuleDict's children, and the scanned stack has a
    # different child structure. The guards below fence off the combinations
    # that are not validated yet rather than letting them fail silently.
    if compile_mode != "fullgraph":
      raise ValueError(
          "tpu_config.use_scan requires tpu_config.compile_mode='fullgraph' on"
          " this path. 'layer' compiles each block separately, which is exactly"
          " what the scan removes. 'aot_fx_trace' is not unsupported, it just"
          " lives elsewhere: scan + aot_fx_trace runs through the graph"
          " trainer's parallelize_scan.py, so pass"
          " --tpu-config.use-graph-trainer with it."
      )
    if ac_config is not None and not isinstance(
        ac_config, torchtitan.distributed.activation_checkpoint.FullAC.Config
    ):
      raise ValueError(
          "tpu_config.use_scan supports activation_checkpoint=none or =full."
          " `torch.utils.checkpoint` inside a scan body is a SILENT no-op --"
          " the HOP re-traces the body and drops the recompute tags, so you"
          " get min-cut's saved set with no warning. Activation memory for the"
          " scanned stack comes from an FX tagging pass on the body's joint"
          " graph instead, and only the 'full' policy has one."
      )
    from torchtitan.experiments.tpu.scan.fsdp import (  # pylint: disable=g-import-not-at-top
        shard_stacked_layers,
    )
    from torchtitan.experiments.tpu.scan.layer_stack import (  # pylint: disable=g-import-not-at-top
        install_scan_over_layers,
    )

    scanned_layers = install_scan_over_layers(
        model,
        parallel_dims=parallel_dims,
        peel_first_layer=tpu_scan_peel_first_layer,
    )

    # The block-wrapping AC policies iterate `layers`' children, which the
    # scanned stack no longer has; the tagging pass below replaces them.
    recompute_scan_body = ac_config is not None
    ac_config = None

  if effective_compile_config.enable and hasattr(model, "layers"):
    for layer in model.layers.values():
      apply_spmd_safe(layer)

  if parallelism.use_simple_fsdp:
    # Apply all other parallelisms but skip data parallelism
    dtensor_parallelize.parallelize_llama(
          model,  # pyrefly: ignore[bad-argument-type]
          ac_config=ac_config,
          compile_config=effective_compile_config,
          dump_folder=dump_folder,
          parallel_dims=parallel_dims,
          parallelism=parallelism,
          training=training,
          skip_dp=True,
      )

    dp_mesh_names = (
          ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
      )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    dp_mode = "hybrid_shard" if parallel_dims.dp_replicate_enabled else "fully_shard"
    mp_policy = SimpleFSDPMixedPrecisionPolicy(
          param_dtype=torchtitan.config.TORCH_DTYPE_MAP[training.mixed_precision_param],
          reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[training.mixed_precision_reduce],
      )
    model = simple_fsdp_data_parallel(
          model,
          dp_mesh,
          mode=dp_mode,
          mp_policy=mp_policy,
          ignored_params=(
              None
              if scanned_layers is None
              else set(scanned_layers.stacked_parameters())
          ),
      )
    if scanned_layers is not None:
      shard_stacked_layers(
          scanned_layers,
          dp_mesh=dp_mesh,
          dp_mode=dp_mode,
          mp_policy=mp_policy,
          recompute_body=recompute_scan_body,
      )
    logger.info(f"Applied SimpleFSDP ({dp_mode}) to the model")
  else:
    dtensor_parallelize.parallelize_llama(
          model,  # pyrefly: ignore[bad-argument-type]
          ac_config=ac_config,
          compile_config=effective_compile_config,
          dump_folder=dump_folder,
          parallel_dims=parallel_dims,
          parallelism=parallelism,
          training=training,
          skip_dp=skip_dp,
      )
  if compile_mode == "fullgraph":
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")
  return model
