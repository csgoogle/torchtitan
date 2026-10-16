# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Spark Open text decoder, ported from upstream TorchTitan.

Ported from upstream ``torchtitan/models/muse_spark/model.py`` (95788b05) onto
this fork's model APIs. Text-only: the upstream vision and audio encoders, the
renderer and the HF state-dict adapter are not ported yet. Numerics follow
upstream, with two simplifications that only affect backward precision or
performance:

* The LM head and router gate use this fork's ``SoftCappedLinear`` and
  ``RouterGateLinear`` instead of upstream's ``HiMidLoLinear``.
* Upstream's ``local_compile`` regions and ``torch_remat`` annotations are
  dropped; activation checkpointing comes from the trainer config.
"""

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.attention import (
    AttentionMasksType,
    GQAttention,
    segment_ids_from_positions,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module

# Shape suffix legend:
# T = tokens, D = model dimension, N = query heads, K = KV heads,
# H = head dimension, O = flattened attention output dimension, V = vocab.


class MuseSparkAttention(GQAttention):
    """GQA with q/k RMSNorm, per-head output RMSNorm and a sigmoid output gate.

    Global layers optionally scale the query by a position-dependent
    temperature. ``window_size`` (None for global layers) is consumed by the
    inner attention; it is kept here for FLOP counting and backend swaps.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        scale_query_by: float
        window_size: int | None = None
        position_temperature_floor_scale: float | None = None
        position_temperature_query_scale: float | None = None
        output_norm: RMSNorm.Config | None = None
        output_gate: Linear.Config | None = None

        @property
        def sliding_window_size(self) -> int | None:
            return self.window_size

    def __init__(self, config: Config):
        super().__init__(config)
        self.window_size = config.window_size
        self.scale_query_by = config.scale_query_by
        self.position_temperature_floor_scale = config.position_temperature_floor_scale
        self.position_temperature_query_scale = config.position_temperature_query_scale
        self.output_norm = (
            config.output_norm.build() if config.output_norm is not None else None
        )
        self.output_gate = (
            config.output_gate.build() if config.output_gate is not None else None
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_TNH, key_TKH, value_TKH = self.qkv_linear(x_TD)
        output_gate_TO = (
            self.output_gate(x_TD) if self.output_gate is not None else None
        )

        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            query_TNH = self.q_norm(query_TNH) * self.scale_query_by
            key_TKH = self.k_norm(key_TKH)

        floor_scale = self.position_temperature_floor_scale
        query_scale = self.position_temperature_query_scale
        if floor_scale is not None and query_scale is not None:
            if positions is None:
                raise ValueError(
                    "positions are required for attention temperature tuning"
                )
            temperature_T = (
                torch.log((positions.float() + 1.0) / floor_scale) * query_scale + 1.0
            ).clamp_min(1.0)
            query_TNH = (query_TNH * temperature_T.reshape(-1, 1, 1)).to(
                query_TNH.dtype
            )

        if self.rope is not None:
            query_TNH, key_TKH = self.rope(query_TNH, key_TKH, positions)

        output_TNH = self.inner_attention(
            query_TNH,
            key_TKH,
            value_TKH,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
            positions=positions,
            segment_ids=segment_ids,
        ).contiguous()
        if self.output_norm is not None:
            output_TNH = self.output_norm(output_TNH)
        output_TO = output_TNH.reshape(output_TNH.shape[0], -1)
        if output_gate_TO is not None:
            output_TO = output_TO * torch.sigmoid(output_gate_TO)
        return self.wo(output_TO)


class MuseSparkRMSNorm(RMSNorm):
    """RMSNorm with effective scale ``weight + gain_center``, output in input dtype."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        gain_center: float = 1.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.gain_center = config.gain_center

    def _get_effective_weight(self) -> torch.Tensor | None:
        if self.weight is None:
            return None
        return self.weight + self.gain_center

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.rms_norm(
            input, self.normalized_shape, self._get_effective_weight(), self.eps
        )
        return output.to(input.dtype)


class MuseSparkRMSNormWithInputScale(MuseSparkRMSNorm):
    """Gain-centered RMSNorm that scales its input before normalizing."""

    @dataclass(kw_only=True, slots=True)
    class Config(MuseSparkRMSNorm.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype
        effective_weight_D = self._get_effective_weight()
        if effective_weight_D is not None:
            input = effective_weight_D * input
        return F.rms_norm(input, self.normalized_shape, None, self.eps).to(
            input_dtype
        )


class MuseSparkNormPreservingResidualConnection(Module):
    """``alpha * stream + beta * branch`` with learned per-channel gates.

    ``beta = sigmoid(g)`` and ``alpha = sqrt(sigmoid(-g) * (1 + beta))``, so a
    unit-variance stream and branch keep unit variance.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        temperature: float = 0.3
        eps: float = 1e-3

    def __init__(self, config: Config):
        super().__init__()
        self.temperature = config.temperature
        self.eps = config.eps
        self.gate = nn.Parameter(torch.empty(config.dim))

    def forward(self, stream_TD: torch.Tensor, branch_TD: torch.Tensor) -> torch.Tensor:
        scaled_gate_1D = self.gate.reshape(1, -1).float() / self.temperature
        beta_1D = torch.sigmoid(scaled_gate_1D)
        alpha_squared_1D = torch.sigmoid(-scaled_gate_1D) * (1.0 + beta_1D)
        alpha_1D = torch.sqrt(torch.clamp(alpha_squared_1D, min=self.eps))
        output_TD = alpha_1D * stream_TD + beta_1D * branch_TD.to(stream_TD.dtype)
        return output_TD.to(stream_TD.dtype)


class MuseSparkEmbeddingWithNorm(Module):
    """Token embedding followed by an optional scaleless RMSNorm and a multiplier."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        embedding: Embedding.Config
        norm: MuseSparkRMSNorm.Config | None = None
        multiplier: float = 1.0

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.embedding = config.embedding.build()
        self.norm = config.norm.build() if config.norm is not None else None
        self.multiplier = config.multiplier

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        output_TD = self.embedding(tokens)
        if self.norm is not None:
            output_TD = self.norm(output_TD)
        return output_TD * self.multiplier


class MuseSparkTransformerBlock(TransformerBlock):
    """Pre- and post-normed attention and MoE branches with gated residuals.

    The residual stream stays in ``residual_dtype`` (FP32); branch inputs are
    cast to ``branch_dtype`` (BF16).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # pyrefly: ignore [bad-override]
        attention: MuseSparkAttention.Config
        post_attention_norm: MuseSparkRMSNorm.Config
        attention_residual: MuseSparkNormPreservingResidualConnection.Config
        post_ffn_norm: MuseSparkRMSNormWithInputScale.Config
        ffn_residual: MuseSparkNormPreservingResidualConnection.Config
        branch_dtype: torch.dtype = torch.bfloat16

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.post_attention_norm = config.post_attention_norm.build()
        self.attention_residual = config.attention_residual.build()
        self.ffn_norm = config.ffn_norm.build()
        self.post_ffn_norm = config.post_ffn_norm.build()
        self.ffn_residual = config.ffn_residual.build()
        self.branch_dtype = config.branch_dtype
        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_input_TD = self.attention_norm(x).to(self.branch_dtype)
        attention_output_TD = self.attention(
            attention_input_TD, attention_masks, positions, segment_ids
        )
        attention_branch_TD = self.post_attention_norm(attention_output_TD)
        stream_TD = self.attention_residual(x, attention_branch_TD)

        ffn_input_TD = self.ffn_norm(stream_TD).to(self.branch_dtype)
        if self.moe_enabled:
            ffn_output_TD = self.moe(ffn_input_TD)
        else:
            ffn_output_TD = self.feed_forward(ffn_input_TD)
        ffn_branch_TD = self.post_ffn_norm(ffn_output_TD)
        return self.ffn_residual(stream_TD, ffn_branch_TD)


class MuseSparkModel(Decoder):
    """Muse Spark Open text decoder."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        # pyrefly: ignore [bad-override]
        tok_embeddings: MuseSparkEmbeddingWithNorm.Config
        residual_dtype: torch.dtype = torch.float32
        final_norm_input_dtype: torch.dtype = torch.bfloat16

        def __post_init__(self) -> None:
            if self.enable_weight_tying:
                raise ValueError(
                    "Muse Spark does not support token embedding weight tying"
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            attention_op_flops = 0
            for layer in self.layers:
                attention = layer.attention
                head_dim = (
                    attention.head_dim
                    if attention.head_dim is not None
                    else attention.dim // attention.n_heads
                )
                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=head_dim,
                    v_head_dim=head_dim,
                    seq_len=seq_len,
                    sliding_window_size=attention.window_size,
                )
            return nparams, 6 * active_nparams + attention_op_flops

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.residual_dtype = config.residual_dtype
        self.final_norm_input_dtype = config.final_norm_input_dtype

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        hidden_TD = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )
        hidden_TD = hidden_TD.to(self.residual_dtype)

        segment_ids = (
            segment_ids_from_positions(positions) if positions is not None else None
        )

        for layer in self.layers.values():
            hidden_TD = layer(hidden_TD, attention_masks, positions, segment_ids)

        if self.norm is not None:
            hidden_TD = self.norm(hidden_TD.to(self.final_norm_input_dtype)).to(
                torch.bfloat16
            )

        if self._skip_lm_head or self.lm_head is None:
            return hidden_TD
        return cast(torch.Tensor, self.lm_head(hidden_TD))
