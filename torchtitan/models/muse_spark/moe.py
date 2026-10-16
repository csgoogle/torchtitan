# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Spark compressed MoE, ported from upstream TorchTitan (95788b05).

Upstream builds this from first-class ``RoutedExperts.output_postprocess``,
``TokenChoiceTopKRouter.route_norm_epsilon`` and a configurable router score
function. This fork lacks those hooks, so they live here:

* ``MuseSparkRoutedExperts`` normalizes each expert output before the
  dispatcher's score-weighted combine.
* ``MuseSparkRouterGate`` folds the router logit multiplier into the gate, so
  the stock ``sigmoid`` score function yields ``sigmoid(logits * multiplier)``.
* ``MuseSparkMoE`` normalizes the top-k scores itself (with the configured
  epsilon) instead of the router.
"""

from dataclasses import dataclass

import torch

from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.models.common.linear import Linear, RouterGateLinear
from torchtitan.models.common.moe import _tokens_per_expert, MoE, RoutedExperts

from .model import MuseSparkRMSNorm, MuseSparkRMSNormWithInputScale

# Shape suffix legend:
# T = num tokens, D = model dimension, C = compressed expert dimension,
# E = num experts, K = top-k, R = routed tokens assigned to local experts.


class MuseSparkRouterGate(RouterGateLinear):
    """FP32 router projection scaled by ``logit_multiplier``."""

    @dataclass(kw_only=True, slots=True)
    class Config(RouterGateLinear.Config):
        logit_multiplier: float = 1.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.logit_multiplier = config.logit_multiplier

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input) * self.logit_multiplier


class MuseSparkRoutedExperts(RoutedExperts):
    """Routed experts whose per-expert outputs are normalized before combine."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        output_norm: MuseSparkRMSNormWithInputScale.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.output_norm = config.output_norm.build()

    def forward(
        self,
        x_TC: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        (
            routed_input_RC,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = self.token_dispatcher.dispatch(
            x_TC,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        with maybe_set_sparse_mesh():
            routed_output_RC = self.inner_experts(
                routed_input_RC, num_global_tokens_per_local_expert_e
            )
            routed_output_RC = self.output_norm(routed_output_RC)
        return self.token_dispatcher.combine(routed_output_RC, metadata, x_TC)


class MuseSparkMoE(MoE):
    """MoE whose experts run in a compressed ``C``-dim space.

    The router reads the ``D``-dim input. ``pooler`` projects the input to
    ``C`` features, ``pre_expert_norm`` normalizes it, the routed experts
    compute and combine in ``C`` features, and ``unpooler`` restores ``D``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        # pyrefly: ignore [bad-override]
        routed_experts: MuseSparkRoutedExperts.Config
        pooler: Linear.Config
        pre_expert_norm: MuseSparkRMSNorm.Config
        unpooler: Linear.Config
        route_norm_epsilon: float
        route_scale: float = 1.0
        expert_bias_init: float = 0.0

        def __post_init__(self) -> None:
            if self.shared_experts is not None:
                raise ValueError("Muse Spark MoE does not support shared experts")
            if self.router.route_norm or self.router.route_scale != 1.0:
                raise ValueError(
                    "Muse Spark MoE normalizes and scales the top-k scores itself; "
                    "set router.route_norm=False and router.route_scale=1.0"
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self.num_experts = config.num_experts
        self.route_norm_epsilon = config.route_norm_epsilon
        self.route_scale = config.route_scale
        self.expert_bias_init = config.expert_bias_init
        self.pooler = config.pooler.build()
        self.pre_expert_norm = config.pre_expert_norm.build()
        self.unpooler = config.unpooler.build()
        # Upstream always keeps an expert bias buffer; the optimizer hook only
        # updates it when load_balance_coeff is set.
        if self.expert_bias_E is None:
            del self.expert_bias_E
            self.register_buffer(
                "expert_bias_E",
                torch.full(
                    (self.num_experts,), self.expert_bias_init, dtype=torch.float32
                ),
                persistent=True,
            )
        else:
            self.expert_bias_E.fill_(self.expert_bias_init)

    def forward(self, x_TD: torch.Tensor, **router_kwargs) -> torch.Tensor:
        topk_scores_TK, topk_expert_ids_TK, routing_map_TE = self.router(
            x_TD, self.expert_bias_E, **router_kwargs
        )
        topk_scores_TK = (
            topk_scores_TK
            / (topk_scores_TK.sum(dim=-1, keepdim=True) + self.route_norm_epsilon)
            * self.route_scale
        )
        num_local_tokens_per_expert_E = (
            routing_map_TE.sum(dim=0)
            if routing_map_TE is not None
            else _tokens_per_expert(topk_expert_ids_TK, self.num_experts)
        )
        if self.training:
            with torch.no_grad():
                self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

        expert_input_TC = self.pre_expert_norm(self.pooler(x_TD))
        out_TC = self.routed_experts(
            expert_input_TC,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        return self.unpooler(out_TC)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        super()._init_self_buffers(buffer_device=buffer_device)
        if buffer_device is None:
            buffer_device = self.tokens_per_expert_E.device
        with torch.device(buffer_device):
            self.expert_bias_E = torch.full(
                (self.num_experts,), self.expert_bias_init, dtype=torch.float32
            )
