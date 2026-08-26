# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scan-over-layers parallelization for llama3 on the graph_trainer lane.

`torchtitan.experiments.graph_trainer.llama3.parallelize.parallelize_llama` has
no scan support, and it is not reachable from the TPU lane's
`patched_parallelize`: `to_graph_trainer_config` replaces the model spec's
`parallelize_fn` with the graph_trainer one on a *copy* of the spec
(`configs.py`, `replace(...)`), built before `train.py` installs its patch on
the original. So `tpu_use_scan` can never arrive there.

This module is the graph_trainer-lane counterpart of the fullgraph lane's
`torchtitan.experiments.tpu.llama3.infra.parallelize`: same four steps, plus the
scan block. `to_tpu_graph_trainer_config` installs it onto the spec as it builds
that copy, which is the only point at which `--tpu-config.use-scan` can still
reach the lane. Keeping it here rather than editing the generic graph_trainer
tree confines scan knowledge to the TPU lane.
"""

from __future__ import annotations

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.experiments.graph_trainer.common_utils import (
    apply_cp_to_attention,
    apply_simple_fsdp,
    resolve_dp_plan,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.llama3.parallelize import annotate_llama
from torchtitan.tools.logging import logger


def parallelize_llama_scan(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    peel_first_layer: bool = False,
):
    """graph_trainer-lane `parallelize_llama` with the layers stacked for scan."""
    from torchtitan.experiments.tpu.scan.fsdp import shard_stacked_layers
    from torchtitan.experiments.tpu.scan.layer_stack import (
        install_scan_over_layers,
    )

    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), (
        f"Sequence length {training.seq_len} must be divisible by the product of"
        f" TP degree ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp})."
    )

    if not parallelism.use_simple_fsdp:
        raise ValueError(
            "tpu_config.use_scan on the graph_trainer lane requires"
            " --parallelism.use-simple-fsdp: the in-body gather is installed as a"
            " simple_fsdp parametrization plan."
        )

    if parallel_dims.cp_enabled:
        apply_cp_to_attention(model, parallel_dims)

    # Before `annotate_llama`, so the FQN annotations describe the stack that
    # actually executes rather than the per-layer modules it replaces.
    scanned_layers = install_scan_over_layers(
        model, parallel_dims=parallel_dims, peel_first_layer=peel_first_layer
    )

    annotate_llama(model)

    model = apply_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=training,
        ignored_params=set(scanned_layers.stacked_parameters()),
    )

    # This lane spells activation checkpointing `memory_policy`, not
    # `ac_config` — the latter is discarded by graph_trainer's
    # `parallelize_llama` and rewritten to SelectiveAC by
    # `to_graph_trainer_config`, so it carries no user intent here.
    del ac_config
    recompute_scan_body = getattr(compile_config, "memory_policy", "default") == "full"

    dp_mesh, dp_mode, mp_policy = resolve_dp_plan(parallel_dims, training)
    shard_stacked_layers(
        scanned_layers,
        dp_mesh=dp_mesh,
        dp_mode=dp_mode,
        mp_policy=mp_policy,
        recompute_body=recompute_scan_body,
    )
    logger.info(
        "Applied scan-over-layers on the graph_trainer lane (dp mode=%s,"
        " recompute_scan_body=%s, peel_first_layer=%s)",
        dp_mode,
        recompute_scan_body,
        peel_first_layer,
    )

    return apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )


__all__ = ["parallelize_llama_scan"]
