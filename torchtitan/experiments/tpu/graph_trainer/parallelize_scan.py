# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scan-over-layers parallelization for llama3 on the graph_trainer lane.

The graph_trainer lane traces forward + loss + backward into one joint graph via
`aot_fx_trace`, and reaches the model through the model spec's `parallelize_fn`.
The generic `graph_trainer.llama3.parallelize.parallelize_llama` has no scan
support, so this module is the scan-aware replacement `config.py` installs in
its place. Same steps as the stock function, with the layers stacked first.

It lives in the TPU lane rather than in the generic graph_trainer tree so scan
knowledge stays in one place; `llama3/infra/parallelize.py` is its fullgraph-lane
counterpart.
"""

from __future__ import annotations

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.experiments.graph_trainer.common_utils import (
    apply_simple_fsdp,
    resolve_dp_plan,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.llama3.parallelize import annotate_llama
from torchtitan.experiments.tpu import scan


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
    """graph_trainer-lane `parallelize_llama` with the layers stacked for scan.

    Args:
      peel_first_layer: run layer 0 outside the scan. See
        `scan.resolve_peel_first_layer`.

    Raises:
      ValueError: TP, CP or PP is enabled; the scan body supports none of them.
    """
    # This lane spells activation checkpointing `memory_policy`, not
    # `ac_config`: graph_trainer's own `parallelize_llama` discards the latter
    # and `to_graph_trainer_config` rewrites it to SelectiveAC, so it carries no
    # user intent here. `full` is the policy that recomputes everything but the
    # layer outputs, which for the stack means the in-body all-gather.
    del ac_config
    recompute_body = getattr(compile_config, "memory_policy", "default") == "full"

    dp_mesh, dp_mode, mp_policy = resolve_dp_plan(parallel_dims, training)
    # Entering stacks the layers, leaving shards the stack and installs the
    # in-body gather; the two calls below have to happen between the halves.
    with scan.over_layers(
        model,
        parallel_dims=parallel_dims,
        dp_mesh=dp_mesh,
        dp_mode=dp_mode,
        mp_policy=mp_policy,
        peel_first_layer=peel_first_layer,
        recompute_body=recompute_body,
    ) as scanned_layers:
        annotate_llama(model)
        model = apply_simple_fsdp(
            model,
            parallel_dims=parallel_dims,
            training=training,
            # The stacked parameters shard on dim 1 and are gathered one layer
            # at a time inside the scan body instead.
            ignored_params=set(scanned_layers.stacked_parameters()),
        )

    return apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )


__all__ = ["parallelize_llama_scan"]
