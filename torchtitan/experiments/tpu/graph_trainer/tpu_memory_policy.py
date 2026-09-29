# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU-only GraphTrainer memory policies."""

from collections.abc import Iterable

import torch

from torchtitan.experiments.graph_trainer.cpu_offload import (
    tag_all_offloadable_activations,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _configured_fsdp_force_save_nodes,
    _full_memory_policy_pass,
    _make_full_memory_policy,
    tag_sac_policy,
)
from torchtitan.experiments.graph_trainer.registry import register_memory_policy


@register_memory_policy("full_and_offload")
def _full_and_offload_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: "GraphTrainer.Config",
) -> torch.fx.GraphModule:
    """Full recompute plus CPU offload of saved layer boundaries."""
    _full_memory_policy_pass(gm, config=config)
    tag_all_offloadable_activations(
        gm,
        cpu_budget_gb=config.compile.cpu_offload_budget_gb,
    )
    return gm


def full_remat_with_saves(
    gm: torch.fx.GraphModule,
    config: "GraphTrainer.Config",
    save_nodes: Iterable[torch.fx.Node],
) -> torch.fx.GraphModule:
    """Recompute everything except ``save_nodes`` and layer boundaries.

    Every TPU selective policy tags the graph through this one function, so
    none of them can drop the FSDP contract: with
    ``reshard_after_forward=False`` (ZeRO-2) the all-gather outputs stay
    ``MUST_SAVE`` whatever else the policy chose to keep.
    """
    force_save_nodes = set(save_nodes)
    force_save_nodes |= _configured_fsdp_force_save_nodes(gm, config) or set()
    return tag_sac_policy(
        gm,
        policy_fn=_make_full_memory_policy(),
        force_save_nodes=force_save_nodes,
    )
