# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder, resolve_fsdp_mesh

from .model import MuseSparkModel


def parallelize_muse_spark(
    model: MuseSparkModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    skip_dp: bool = False,
):
    """AC -> per-block compile -> FSDP2. TP/EP/CP/PP are not ported yet."""
    # TODO: port upstream's TP/SP/EP sharding configs (muse_spark/sharding.py).
    for name, enabled in (
        ("Tensor", parallel_dims.tp_enabled),
        ("Expert", parallel_dims.ep_enabled),
        ("Context", parallel_dims.cp_enabled),
        ("Pipeline", parallel_dims.pp_enabled),
    ):
        if enabled:
            raise NotImplementedError(
                f"{name} parallelism is not supported for Muse Spark in this fork yet."
            )

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if compile_config.enable and "model" in compile_config.components:
        apply_compile(model, compile_config=compile_config, parallel_dims=parallel_dims)

    if skip_dp:
        return model

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        dp_mesh_dims=dp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )
    return model
