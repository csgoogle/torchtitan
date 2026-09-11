# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization for Muse Glimmer on TPU (mirrors the qwen3 TPU wrapper)."""

import dataclasses

import torch
from torch import nn

import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    MixedPrecisionPolicy as SimpleFSDPMixedPrecisionPolicy,
    data_parallel as simple_fsdp_data_parallel,
)
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.models.muse_glimmer import parallelize as upstream_parallelize
from torchtitan.tools.logging import logger


def parallelize_muse_glimmer(
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
    **kwargs,
) -> nn.Module:
  """Delegates to the upstream Muse Glimmer parallelizer (which applies the
  model's spmd_types sharding, AC, per-block compile and FSDP2); with
  ``--parallelism.use_simple_fsdp`` upstream data parallelism is skipped and
  the TPU SimpleFSDP is applied instead.
  """
  compile_mode = tpu_compile_mode
  effective_compile_config = compile_config
  reduce_dtype = torchtitan.config.TORCH_DTYPE_MAP[training.mixed_precision_reduce]

  if effective_compile_config.enable and hasattr(model, "layers"):
    for layer in model.layers.values():
      apply_spmd_safe(layer)

  if compile_mode in ("fullgraph", "aot_fx_trace") and compile_config.enable:
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  upstream_parallelize.parallelize_muse_glimmer(
      model,  # pyrefly: ignore[bad-argument-type]
      parallel_dims=parallel_dims,
      training=training,
      parallelism=parallelism,
      compile_config=effective_compile_config,
      ac_config=ac_config,
      dump_folder=dump_folder,
      skip_dp=skip_dp or parallelism.use_simple_fsdp,
  )

  if parallelism.use_simple_fsdp and not skip_dp:
    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
    dp_mode = "hybrid_shard" if parallel_dims.dp_replicate_enabled else "fully_shard"
    mp_policy = SimpleFSDPMixedPrecisionPolicy(
        param_dtype=torchtitan.config.TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=reduce_dtype,
    )
    model = simple_fsdp_data_parallel(
        model,
        dp_mesh,
        mode=dp_mode,
        mp_policy=mp_policy,
    )
    logger.info(f"Applied SimpleFSDP ({dp_mode}) to the model")

  if compile_mode == "fullgraph":
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")
  return model
