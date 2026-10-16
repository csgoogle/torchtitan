# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Spark Open text decoder (see ``model.py`` for the port scope).

Flavors follow upstream ``torchtitan/models/muse_spark/flavors.py``
(95788b05). ``6B-A1.4B`` and ``220B-A44B`` are not upstream: they are
scaled-down proxies with the 816B-A42B architecture (layer pattern, head and
expert shapes, vocabulary, norm/residual/router constants). ``6B-A1.4B`` trains
on a single v6e-8 host; ``220B-A44B`` keeps the full per-token compute and
targets a v7x-64 (tpu7x-4x4x4) slice.
"""

import dataclasses
import math
from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.models.common import ComplexRoPE, Embedding, Linear
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    QKVLinear,
    VarlenInnerAttention,
)
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.models.common.moe import GroupedExperts, TokenChoiceTopKRouter
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.muse_glimmer.model import SoftCappedLinear
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import (
    MuseSparkAttention,
    MuseSparkEmbeddingWithNorm,
    MuseSparkModel,
    MuseSparkNormPreservingResidualConnection,
    MuseSparkRMSNorm,
    MuseSparkRMSNormWithInputScale,
    MuseSparkTransformerBlock,
)
from .moe import MuseSparkMoE, MuseSparkRoutedExperts, MuseSparkRouterGate
from .parallelize import parallelize_muse_spark

__all__ = [
    "MuseSparkModel",
    "model_registry",
    "muse_spark_configs",
    "parallelize_muse_spark",
]

# The query is scaled by this tuned constant divided by sqrt(head_dim) after
# q-norm.
_SCALE_QUERY_NUMERATOR = 43.7840518911

_LINEAR_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
}
_EMBEDDING_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.normal_, std=1.0),
}
_GAIN_NORM_INIT: dict[str, Callable] = {"weight": nn.init.zeros_}

_NORM_EPS = 1e-5
_POST_NORM_EPS = 1e-8
_ROPE_THETA = 500_000.0
_RESIDUAL_GATE_TEMPERATURE = 0.3
_RESIDUAL_GATE_EPS = 1e-3
_MUSE_SPARK_VOCAB_SIZE = 202_048


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id))
    }


def _output_linear_init(dim: int) -> dict[str, Callable]:
    std = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
    }


def _grouped_experts_init(layer_id: int) -> dict[str, Callable]:
    depth_std = depth_scaled_std(0.02, layer_id)
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=depth_std),
        "w3_EFD": partial(nn.init.trunc_normal_, std=depth_std),
    }


def _gain_norm(
    dim: int,
    *,
    elementwise_affine: bool = True,
    gain_center: float = 1.0,
    eps: float = _NORM_EPS,
) -> MuseSparkRMSNorm.Config:
    return MuseSparkRMSNorm.Config(
        normalized_shape=dim,
        eps=eps,
        elementwise_affine=elementwise_affine,
        gain_center=gain_center,
        param_init=_GAIN_NORM_INIT if elementwise_affine else None,
    )


def _input_scaled_gain_norm(
    dim: int, *, gain_center: float, eps: float
) -> MuseSparkRMSNormWithInputScale.Config:
    return MuseSparkRMSNormWithInputScale.Config(
        normalized_shape=dim,
        eps=eps,
        elementwise_affine=True,
        gain_center=gain_center,
        param_init=_GAIN_NORM_INIT,
    )


def _scaleless_norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_NORM_EPS, elementwise_affine=False)


def _residual_gate(
    dim: int, *, layer_id: int
) -> MuseSparkNormPreservingResidualConnection.Config:
    target_beta = 1.0 / (2.0 * (layer_id + 1))
    target_beta = max(_RESIDUAL_GATE_EPS, min(1.0 - _RESIDUAL_GATE_EPS, target_beta))
    gate_init = _RESIDUAL_GATE_TEMPERATURE * math.log(target_beta / (1.0 - target_beta))
    return MuseSparkNormPreservingResidualConnection.Config(
        dim=dim,
        temperature=_RESIDUAL_GATE_TEMPERATURE,
        eps=_RESIDUAL_GATE_EPS,
        param_init={"gate": partial(nn.init.constant_, val=gate_init)},
    )


def _attention_config(
    *,
    layer_id: int,
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_context_length: int,
    window_size: int | None,
    query_scale: float,
    position_temperature_floor_scale: float | None,
    position_temperature_query_scale: float | None,
    attn_backend: str,
) -> MuseSparkAttention.Config:
    inner_attention = get_attention_config(attn_backend)
    if isinstance(inner_attention, FlexInnerAttention.Config):
        # TODO: port upstream's SlidingWindowFlexInnerAttention.
        raise NotImplementedError("Muse Spark does not support flex attention yet")
    if window_size is not None and isinstance(
        inner_attention, VarlenInnerAttention.Config
    ):
        inner_attention = dataclasses.replace(
            inner_attention, window_size=(window_size - 1, 0)
        )
    # "sdpa" ignores window_size; it is only valid once a backend swap (e.g.
    # torchtitan.experiments.tpu.muse_spark.use_tpu_attention) installs a
    # window-aware inner attention.
    is_global = window_size is None
    return MuseSparkAttention.Config(
        n_heads=num_heads,
        n_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dim=dim,
        qkv_linear=QKVLinear.Config(
            head_dim=head_dim,
            wq=Linear.Config(
                in_features=dim,
                out_features=num_heads * head_dim,
                param_init=_LINEAR_INIT,
            ),
            wkv=Linear.Config(
                in_features=dim,
                out_features=num_kv_heads * head_dim,
                param_init=_LINEAR_INIT,
            ),
        ),
        wo=Linear.Config(
            in_features=num_heads * head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        qk_norm=_scaleless_norm(head_dim),
        inner_attention=inner_attention,
        # Global layers use no RoPE.
        rope=(
            None
            if is_global
            else ComplexRoPE.Config(
                dim=head_dim,
                max_context_length=max_context_length,
                theta=_ROPE_THETA,
            )
        ),
        scale_query_by=query_scale,
        window_size=window_size,
        position_temperature_floor_scale=(
            position_temperature_floor_scale if is_global else None
        ),
        position_temperature_query_scale=(
            position_temperature_query_scale if is_global else None
        ),
        output_norm=_scaleless_norm(head_dim),
        output_gate=Linear.Config(
            in_features=dim,
            out_features=num_heads * head_dim,
            param_init=_LINEAR_INIT,
        ),
    )


def _moe_config(
    *,
    layer_id: int,
    dim: int,
    compressed_dim: int,
    expert_hidden_dim: int,
    num_experts: int,
    top_k: int,
    post_norm_gain_center: float,
    route_scale: float,
    router_logit_multiplier: float,
    route_norm_epsilon: float,
    load_balance_coeff: float | None,
    expert_bias_init: float,
) -> MuseSparkMoE.Config:
    return MuseSparkMoE.Config(
        num_experts=num_experts,
        router=TokenChoiceTopKRouter.Config(
            num_experts=num_experts,
            gate=MuseSparkRouterGate.Config(
                in_features=dim,
                out_features=num_experts,
                param_init=_depth_init(layer_id),
                logit_multiplier=router_logit_multiplier,
            ),
            top_k=top_k,
            score_func="sigmoid",
            route_norm=False,
            route_scale=1.0,
        ),
        routed_experts=MuseSparkRoutedExperts.Config(
            inner_experts=GroupedExperts.Config(
                dim=compressed_dim,
                hidden_dim=expert_hidden_dim,
                num_experts=num_experts,
                param_init=_grouped_experts_init(layer_id),
            ),
            token_dispatcher=AllToAllTokenDispatcher.Config(
                num_experts=num_experts, top_k=top_k
            ),
            output_norm=_input_scaled_gain_norm(
                compressed_dim, gain_center=post_norm_gain_center, eps=_POST_NORM_EPS
            ),
        ),
        load_balance_coeff=load_balance_coeff,
        pooler=Linear.Config(
            in_features=dim, out_features=compressed_dim, param_init=_LINEAR_INIT
        ),
        pre_expert_norm=_gain_norm(compressed_dim),
        unpooler=Linear.Config(
            in_features=compressed_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        route_norm_epsilon=route_norm_epsilon,
        route_scale=route_scale,
        expert_bias_init=expert_bias_init,
    )


def _muse_spark_config(
    *,
    dim: int,
    num_layers: int,
    vocab_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_context_length: int,
    global_attention_layers: set[int],
    sliding_window_size: int,
    compressed_dim: int,
    expert_hidden_dim: int,
    num_experts: int,
    top_k: int,
    position_temperature_floor_scale: float | None,
    position_temperature_query_scale: float | None,
    post_norm_gain_centers: tuple[float, ...],
    post_attention_norm_has_weight: bool,
    final_norm_gain_center: float,
    embedding_multiplier: float,
    output_multiplier: float,
    output_soft_cap_temp: float,
    final_norm_input_dtype: torch.dtype,
    route_scale: float,
    router_logit_multiplier: float,
    route_norm_epsilon: float,
    load_balance_coeff: float | None,
    expert_bias_init: float,
    attn_backend: str,
) -> MuseSparkModel.Config:
    assert len(post_norm_gain_centers) == num_layers
    layers = []
    for layer_id, post_norm_gain_center in enumerate(post_norm_gain_centers):
        is_global_attention = layer_id in global_attention_layers
        layers.append(
            MuseSparkTransformerBlock.Config(
                attention=_attention_config(
                    layer_id=layer_id,
                    dim=dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    max_context_length=max_context_length,
                    window_size=None if is_global_attention else sliding_window_size,
                    query_scale=_SCALE_QUERY_NUMERATOR / math.sqrt(head_dim),
                    position_temperature_floor_scale=position_temperature_floor_scale,
                    position_temperature_query_scale=position_temperature_query_scale,
                    attn_backend=attn_backend,
                ),
                moe=_moe_config(
                    layer_id=layer_id,
                    dim=dim,
                    compressed_dim=compressed_dim,
                    expert_hidden_dim=expert_hidden_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    post_norm_gain_center=post_norm_gain_center,
                    route_scale=route_scale,
                    router_logit_multiplier=router_logit_multiplier,
                    route_norm_epsilon=route_norm_epsilon,
                    load_balance_coeff=load_balance_coeff,
                    expert_bias_init=expert_bias_init,
                ),
                attention_norm=_gain_norm(dim),
                post_attention_norm=_gain_norm(
                    dim,
                    elementwise_affine=post_attention_norm_has_weight,
                    gain_center=post_norm_gain_center,
                    eps=_POST_NORM_EPS,
                ),
                attention_residual=_residual_gate(dim, layer_id=layer_id),
                ffn_norm=_gain_norm(dim),
                post_ffn_norm=_input_scaled_gain_norm(
                    dim, gain_center=post_norm_gain_center, eps=_POST_NORM_EPS
                ),
                ffn_residual=_residual_gate(dim, layer_id=layer_id),
                branch_dtype=torch.bfloat16,
            )
        )

    return MuseSparkModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=MuseSparkEmbeddingWithNorm.Config(
            embedding=Embedding.Config(
                num_embeddings=vocab_size,
                embedding_dim=dim,
                param_init=_EMBEDDING_INIT,
            ),
            norm=_gain_norm(dim, elementwise_affine=False),
            multiplier=embedding_multiplier,
        ),
        layers=layers,
        # Keep the zero-centered checkpoint parameterization while starting
        # every final norm with unit effective scale.
        norm=dataclasses.replace(
            _gain_norm(dim, gain_center=final_norm_gain_center),
            param_init=(
                {"weight": nn.init.ones_}
                if final_norm_gain_center == 0.0
                else _GAIN_NORM_INIT
            ),
        ),
        lm_head=SoftCappedLinear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            output_multiplier=output_multiplier,
            output_soft_cap_temp=output_soft_cap_temp,
        ),
        residual_dtype=torch.float32,
        final_norm_input_dtype=final_norm_input_dtype,
    )


def _debugmodel(attn_backend: str, *, seq_len: int) -> MuseSparkModel.Config:
    num_layers = 8
    return _muse_spark_config(
        dim=256,
        num_layers=num_layers,
        vocab_size=2048,
        num_heads=8,
        num_kv_heads=4,
        head_dim=32,
        max_context_length=seq_len,
        global_attention_layers={3, 7},
        sliding_window_size=128,
        compressed_dim=128,
        expert_hidden_dim=256,
        num_experts=8,
        top_k=2,
        position_temperature_floor_scale=32.0,
        position_temperature_query_scale=0.125,
        post_norm_gain_centers=tuple(
            1.0 / math.sqrt(2.0 * (layer_id + 1)) for layer_id in range(num_layers)
        ),
        post_attention_norm_has_weight=True,
        final_norm_gain_center=1.0,
        embedding_multiplier=1.25,
        output_multiplier=0.5,
        output_soft_cap_temp=30.0,
        final_norm_input_dtype=torch.float32,
        route_scale=1.5,
        router_logit_multiplier=0.75,
        route_norm_epsilon=1e-6,
        load_balance_coeff=None,
        expert_bias_init=0.125,
        attn_backend=attn_backend,
    )


def _production_constants(num_layers: int) -> dict:
    """816B-A42B settings that do not depend on model size."""
    return dict(
        vocab_size=_MUSE_SPARK_VOCAB_SIZE,
        head_dim=64,
        global_attention_layers=set(range(1, num_layers, 4)),
        sliding_window_size=2048,
        position_temperature_floor_scale=None,
        position_temperature_query_scale=None,
        post_norm_gain_centers=(1.0,) * num_layers,
        post_attention_norm_has_weight=False,
        final_norm_gain_center=0.0,
        embedding_multiplier=1.0,
        output_multiplier=1.0 / math.sqrt(32.0),
        output_soft_cap_temp=20.0,
        final_norm_input_dtype=torch.bfloat16,
        route_scale=1.0,
        router_logit_multiplier=1.0 / math.sqrt(32.0),
        route_norm_epsilon=1e-15,
        load_balance_coeff=1e-3,
        expert_bias_init=0.0,
    )


def _muse_spark_816b_a42b(attn_backend: str, *, seq_len: int) -> MuseSparkModel.Config:
    num_layers = 62
    return _muse_spark_config(
        dim=8192,
        num_layers=num_layers,
        num_heads=128,
        num_kv_heads=16,
        max_context_length=seq_len,
        compressed_dim=4096,
        expert_hidden_dim=4096,
        num_experts=256,
        top_k=8,
        attn_backend=attn_backend,
        **_production_constants(num_layers),
    )


def _muse_spark_6b_a1_4b(attn_backend: str, *, seq_len: int) -> MuseSparkModel.Config:
    """816B-A42B scaled down 4x in width, 2.6x in depth and 4x in experts.

    Keeps head_dim 64, the 8:1 query:KV head ratio, compressed = expert hidden
    = dim / 2, top-8 routing, the 2048-token window, every fourth layer global
    (starting at layer 1) and the production vocabulary and constants.
    """
    num_layers = 24
    return _muse_spark_config(
        dim=2048,
        num_layers=num_layers,
        num_heads=32,
        num_kv_heads=4,
        max_context_length=seq_len,
        compressed_dim=1024,
        expert_hidden_dim=1024,
        num_experts=64,
        top_k=8,
        attn_backend=attn_backend,
        **_production_constants(num_layers),
    )


def _muse_spark_220b_a44b(attn_backend: str, *, seq_len: int) -> MuseSparkModel.Config:
    """816B-A42B with 64 instead of 256 experts.

    Full production width and depth, top-8 routing and expert shapes, so the
    per-token compute (active parameters) matches 816B-A42B; only the expert
    count, and with it the total parameter count, is reduced.
    """
    num_layers = 62
    return _muse_spark_config(
        dim=8192,
        num_layers=num_layers,
        num_heads=128,
        num_kv_heads=16,
        max_context_length=seq_len,
        compressed_dim=4096,
        expert_hidden_dim=4096,
        num_experts=64,
        top_k=8,
        attn_backend=attn_backend,
        **_production_constants(num_layers),
    )


# Each tuple contains the model factory and its maximum context length.
muse_spark_configs = {
    "debugmodel": (_debugmodel, 32_768),
    "6B-A1.4B": (_muse_spark_6b_a1_4b, 131_072),
    "220B-A44B": (_muse_spark_220b_a44b, 1_048_576),
    "816B-A42B": (_muse_spark_816b_a42b, 1_048_576),
}


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "varlen",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    if flavor not in muse_spark_configs:
        raise ValueError(
            f"Unknown Muse Spark flavor {flavor!r}; available: "
            f"{sorted(muse_spark_configs)}"
        )
    get_config, max_context_len = muse_spark_configs[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(attn_backend=attn_backend, seq_len=context_len)
    if converters is not None:
        validate_converter_order(converters)
        for converter in converters:
            converter.build().convert(config)
    return ModelSpec(
        name="muse_spark",
        flavor=flavor,
        model=config,
        max_context_length=context_len,
        parallelize_fn=parallelize_muse_spark,
        pipelining_fn=None,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        # TODO: port upstream's MuseSparkStateDictAdapter for HF checkpoints.
        state_dict_adapter=None,
    )
