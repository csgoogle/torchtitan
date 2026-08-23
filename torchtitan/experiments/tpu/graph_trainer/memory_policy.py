# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU-specific GraphTrainer activation-memory policies.

The TPU pass-pipeline module imports this module for its registration side
effects. Shared GraphTrainer code therefore owns only the policy registry and
generic policies; TPU policy names, selectors, and configuration stay here.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node
from torchtitan.experiments.graph_trainer.memory_policy import (
    _make_full_memory_policy,
    tag_sac_policy,
)
from torchtitan.experiments.graph_trainer.registry import register_memory_policy
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
        TpuGraphTrainer,
    )


_TPU_SPLASH_FORWARD_TARGET = re.compile(
    r"^pallas\.splash_attn_fwd_[0-9a-f]{16}\.default$"
)


def _is_tpu_splash_attention_forward(node: torch.fx.Node) -> bool:
    output = node.meta.get("val")
    if not (
        _TPU_SPLASH_FORWARD_TARGET.fullmatch(str(node.target))
        and isinstance(output, tuple)
        and len(output) == 2
    ):
        return False
    attention_output, logsumexp = output
    if not (
        isinstance(attention_output, torch.Tensor)
        and isinstance(logsumexp, torch.Tensor)
        and attention_output.ndim == 4
        and logsumexp.ndim == 3
        and attention_output.dtype == torch.bfloat16
        and logsumexp.dtype == torch.float32
    ):
        return False
    kernel_layout = attention_output.shape[:-1] == logsumexp.shape
    model_layout = (
        attention_output.shape[0],
        attention_output.shape[2],
        attention_output.shape[1],
    ) == logsumexp.shape
    return kernel_layout or model_layout


def _select_tpu_splash_attention_save_nodes(
    gm: torch.fx.GraphModule,
    *,
    budget_bytes: int,
) -> tuple[set[torch.fx.Node], int, int]:
    candidates = [
        node
        for node in gm.graph.nodes
        if not _is_backward_node(node)
        and _is_tpu_splash_attention_forward(node)
    ]
    save_nodes: set[torch.fx.Node] = set()
    saved_bytes = 0
    # Later layers are consumed and released earlier in backward, reducing
    # retained lifetime for the same amount of avoided recomputation.
    for node in reversed(candidates):
        node_bytes = sum(
            int(output.numel()) * output.element_size()
            for output in node.meta["val"]
        )
        if saved_bytes + node_bytes > budget_bytes:
            continue
        save_nodes.add(node)
        saved_bytes += node_bytes
    logger.info(
        "Selected %d/%d Splash attention output tuple(s), %.3f GiB",
        len(save_nodes),
        len(candidates),
        saved_bytes / (1024**3),
    )
    return save_nodes, saved_bytes, len(candidates)


@register_memory_policy("tpu_splash")
def _tpu_splash_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: "TpuGraphTrainer.Config",
) -> torch.fx.GraphModule:
    """Apply full recompute except for budgeted Splash output/LSE tuples."""
    budget_gb = config.tpu_config.graph_trainer_memory_policy_save_budget_gb
    splash_save_nodes, _, candidate_count = (
        _select_tpu_splash_attention_save_nodes(
            gm,
            budget_bytes=int(budget_gb * (1024**3)),
        )
    )
    if candidate_count == 0:
        raise ValueError("tpu_splash found no Splash attention forward nodes")
    if not splash_save_nodes:
        raise ValueError(
            f"No Splash output tuple fits within the {budget_gb:.3f} GiB "
            "save budget"
        )
    tag_sac_policy(
        gm,
        policy_fn=_make_full_memory_policy(),
        force_save_nodes=splash_save_nodes,
    )
    return gm
