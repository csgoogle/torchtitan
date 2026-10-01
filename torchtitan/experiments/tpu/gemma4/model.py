# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexAttention,
    get_causal_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
    get_sliding_window_mask_mod,
    GQAttention,
    QKVLinear,
    VarlenAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import (
    GroupedExperts,
    MoE,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.utils import (
    get_dense_model_nparams_and_flops,
    get_moe_model_nparams_and_flops,
)
from torchtitan.tools.logging import logger


class Gemma4Embedding(Embedding):
    """Token embedding scaled by ``sqrt(embedding_dim)`` as in Gemma models."""

    @dataclass(kw_only=True, slots=True)
    class Config(Embedding.Config):
        embed_scale: float | None = None

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.embed_scale: float = (
            config.embed_scale
            if config.embed_scale is not None
            else config.embedding_dim**0.5
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input) * self.embed_scale


class Gemma4RoPE(CosSinRoPE):
    """RoPE with optional proportional rotary factor for Gemma4 global layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        partial_rotary_factor: float = 1.0

    def __init__(self, config: Config) -> None:
        self.partial_rotary_factor = config.partial_rotary_factor
        super().__init__(config)

    def _precompute_cache(self) -> torch.Tensor:
        cfg = self.config
        dim = cfg.dim
        max_seq_len = cfg.max_seq_len
        base = cfg.theta
        assert cfg.scaling == "none", f"RoPE scaling {cfg.scaling} is not supported"
        rope_angles = int(self.partial_rotary_factor * (dim // 2))
        inv_freq_rot = 1.0 / (
            base
            ** (
                torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32)
                / float(dim)
            )
        )
        nop_angles = (dim // 2) - rope_angles
        if nop_angles > 0:
            inv_freq = torch.cat(
                [
                    inv_freq_rot,
                    torch.zeros(nop_angles, dtype=torch.float32),
                ],
                dim=0,
            )
        else:
            inv_freq = inv_freq_rot
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype, device=inv_freq.device)
        freqs = torch.outer(t, inv_freq).float()
        theta = torch.cat([freqs, freqs], dim=-1)
        return torch.cat([theta.cos(), theta.sin()], dim=-1)


class QKVLinearKeqV(QKVLinear):
    """Two separate linear projections for Q and K, reusing K as V (use_k_eq_v=True)."""

    @dataclass(kw_only=True, slots=True)
    class Config(QKVLinear.Config):
        pass

    def __init__(self, config: Config) -> None:
        # Call BaseQKVLinear.__init__ directly so we only build wq and wk (no wv).
        super(QKVLinear, self).__init__(config)
        self.wq = config.wq.build()
        self.wk = config.wkv.build()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, seqlen, _ = x.shape
        xq, xk = self.wq(x), self.wk(x)

        def local_qkv_head_split(tensor: torch.Tensor) -> torch.Tensor:
            with spmd.local():
                tensor_ = tensor.view(bs, seqlen, -1, self.head_dim)
                if get_spmd_backend() == "spmd_types":
                    spmd.assert_type(
                        tensor_, spmd.V, spmd.PartitionSpec("dp", "cp", "tp", None)
                    )
            return tensor_

        xq = local_qkv_head_split(xq)
        xk = local_qkv_head_split(xk)
        return xq, xk, xk


class Gemma4Attention(GQAttention):
    """Gemma4 attention supporting hybrid sliding/global layers, K=V, and Q/K/V norms."""

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        v_norm: RMSNorm.Config
        is_sliding: bool = True
        sliding_window_size: int | None = None
        use_k_eq_v: bool = False

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.v_norm = config.v_norm.build()
        self.is_sliding = config.is_sliding
        self.sliding_window_size = config.sliding_window_size
        self.use_k_eq_v = config.use_k_eq_v
        # Gemma4 normalizes Q, K, and V with RMSNorm and uses a softmax scale of 1.0.
        self.scaling = 1.0
        if self.is_sliding and self.sliding_window_size is not None:
            # HF's sliding_window includes the current token (0 <= q - kv < W),
            # while the TPU splash kernels read inner_attention.sliding_window_size
            # as the number of *previous* keys (LocalMask(window_size=(left, 0))).
            self.inner_attention.sliding_window_size = self.sliding_window_size - 1

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x_BLD.shape
        xq_BLNH, xk_BLNH, xv_BLNH = self.qkv_linear(x_BLD)

        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            xq_BLNH = self.q_norm(xq_BLNH)
            xk_BLNH = self.k_norm(xk_BLNH)
        xv_BLNH = self.v_norm(xv_BLNH)

        xq_BLNH, xk_BLNH = self.rope(xq_BLNH, xk_BLNH, positions)

        mask: AttentionMasksType | None
        if isinstance(attention_masks, dict):
            mask = (
                attention_masks["sliding_window_mask"]
                if self.is_sliding
                else attention_masks["basic_mask"]
            )
        else:
            mask = attention_masks

        out_BLNH = self.inner_attention(
            xq_BLNH,
            xk_BLNH,
            xv_BLNH,
            attention_masks=mask,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
            positions=positions,
            segment_ids=segment_ids,
        ).contiguous()
        out_BLD = out_BLNH.view(B, L, -1)
        return self.wo(out_BLD)


class Gemma4FeedForward(FeedForward):
    """GeGLU feed-forward network (GELU with tanh approximation)."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x), approximate="tanh") * self.w3(x))


class Gemma4GroupedExperts(GroupedExperts):
    """Grouped routed experts using GeGLU activation (`gelu(approximate='tanh')`)."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        activation: str = "gelu_tanh"

    def _experts_forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        from torch.distributed.tensor import DTensor
        from torchtitan.distributed.spmd_types import spmd_mesh_size

        if isinstance(self.w1_EFD, DTensor):
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
            if spmd_mesh_size("ep") == 1:
                for axis in ("dp", "cp"):
                    spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)
            h_RF = F.gelu(
                torch._grouped_mm(
                    x_RD.bfloat16(),
                    w1_EFD.bfloat16().transpose(-2, -1),
                    offs=offsets_E,
                ),
                approximate="tanh",
            )
            h_RF = h_RF * torch._grouped_mm(
                x_RD.bfloat16(),
                w3_EFD.bfloat16().transpose(-2, -1),
                offs=offsets_E,
            )
            return torch._grouped_mm(
                h_RF, w2_EDF.bfloat16().transpose(-2, -1), offs=offsets_E
            ).type_as(x_RD)

        # Pad 1 dummy zero token at the end of each expert's segment so every
        # expert has >= 1 token (K_e >= 1).
        R, D = x_RD.shape
        E = offsets_E.size(0)
        r_idx = torch.arange(R, device=x_RD.device, dtype=offsets_E.dtype)
        padded_idx = (
            r_idx + torch.searchsorted(offsets_E, r_idx, right=True)
        ).long()
        padded_offsets_E = offsets_E + torch.arange(
            1, E + 1, device=offsets_E.device, dtype=offsets_E.dtype
        )
        x_padded_RD = x_RD.new_zeros((R + E, D)).index_copy(0, padded_idx, x_RD)

        h_RF = F.gelu(
            torch._grouped_mm(
                x_padded_RD.bfloat16(),
                w1_EFD.bfloat16().transpose(-2, -1),
                offs=padded_offsets_E,
            ),
            approximate="tanh",
        )
        h_RF = h_RF * torch._grouped_mm(
            x_padded_RD.bfloat16(),
            w3_EFD.bfloat16().transpose(-2, -1),
            offs=padded_offsets_E,
        )
        out_padded_RD = torch._grouped_mm(
            h_RF, w2_EDF.bfloat16().transpose(-2, -1), offs=padded_offsets_E
        ).type_as(x_RD)
        return out_padded_RD.index_select(0, padded_idx)


class Gemma4Router(TokenChoiceTopKRouter):
    """Gemma4 top-k router with input RMSNorm, learned input scale, and per-expert scale."""

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        dim: int
        norm: RMSNorm.Config

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.dim = config.dim
        self.root_size: float = config.dim**-0.5
        self.norm = config.norm.build()
        self.scale = nn.Parameter(torch.ones(config.dim))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.scale)
        nn.init.ones_(self.per_expert_scale)

    def forward(
        self, x_BLD: torch.Tensor, expert_bias_E: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_normed = self.norm(x_BLD) * self.root_size * self.scale.to(x_BLD.dtype)
        topk_scores_BLK, topk_expert_ids_BLK, scores_BLE = super().forward(
            x_normed, expert_bias_E
        )
        expert_scales_BLK = (
            scores_BLE.new_zeros(scores_BLE.shape)
            + self.per_expert_scale.to(topk_scores_BLK.dtype)
        ).gather(dim=-1, index=topk_expert_ids_BLK)
        topk_scores_BLK = topk_scores_BLK * expert_scales_BLK
        return topk_scores_BLK, topk_expert_ids_BLK, scores_BLE


class Gemma4MoE(MoE):
    """Gemma4 MoE module applying ``pre_feedforward_layernorm_2`` to expert inputs while routing on residual ``x_BLD``."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pre_feedforward_layernorm_2: RMSNorm.Config | None = None

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.pre_feedforward_layernorm_2 = (
            config.pre_feedforward_layernorm_2.build()
            if config.pre_feedforward_layernorm_2 is not None
            else None
        )

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        from torch.distributed.tensor import DTensor
        from torchtitan.models.common.moe import _tokens_per_expert
        from torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher

        B, L, _ = x_BLD.shape
        sp_size = getattr(self.experts.token_dispatcher, "sp_size", 1)
        if not isinstance(x_BLD, DTensor) and self.seq_dim_tp_sharded:
            seq_pad = 0
            seq_dim_pad_tokens = 0
            num_local_tokens_after_seq_dim_padding = B * L
        else:
            seq_pad = sp_size - L if L < sp_size else 0
            if seq_pad:
                x_BLD = F.pad(x_BLD, (0, 0, 0, seq_pad))
                L = L + seq_pad
            seq_dim_pad_tokens = (-L) % sp_size
            local_batch_size = (
                x_BLD._local_tensor.shape[0] if isinstance(x_BLD, DTensor) else B
            )
            num_local_tokens_after_seq_dim_padding = (
                local_batch_size * (L + seq_dim_pad_tokens) // sp_size
            )

        # Router receives unnormalized residual x_BLD (applies its own scale-free RMSNorm).
        (
            topk_scores_BLK,
            topk_expert_ids_BLK,
            scores_BLE,
        ) = self.router(x_BLD, self.expert_bias_E)

        if self.training and self.aux_loss is not None:
            topk_scores_BLK = self.aux_loss(
                topk_scores_BLK,
                scores_BLE,
                topk_expert_ids_BLK,
            )

        num_local_tokens_per_expert_E = _tokens_per_expert(
            topk_expert_ids_BLK,
            scores_BLE.size(-1),
        )
        if (
            self.training
            and self.aux_loss is not None
            and self.aux_loss.collect_routing_metrics
        ):
            self.aux_loss._last_local_tokens_per_expert_E = (
                num_local_tokens_per_expert_E.detach().float()
            )

        with torch.no_grad():
            self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

        # Routed experts receive pre_feedforward_layernorm_2(x_BLD).
        x_experts_BLD = (
            self.pre_feedforward_layernorm_2(x_BLD)
            if self.pre_feedforward_layernorm_2 is not None
            else x_BLD
        )
        out_BLD = self.experts(
            x_experts_BLD,
            topk_scores_BLK,
            topk_expert_ids_BLK,
            num_local_tokens_per_expert_E,
            num_local_tokens_after_seq_dim_padding=(
                num_local_tokens_after_seq_dim_padding
            ),
        )

        shared_out_BLD = (
            self.shared_experts(x_BLD) if self.shared_experts is not None else None
        )

        if (
            isinstance(self.experts.token_dispatcher, DeepEPTokenDispatcher)
            and self.experts.token_dispatcher.sp_size == 1
        ):
            from torchtitan.distributed.deepep.deepep import sync_combine

            sync_combine()

        if seq_dim_pad_tokens:
            out_BLD = out_BLD[:, :L, :]

        if shared_out_BLD is not None:
            out_BLD = out_BLD + shared_out_BLD

        if seq_pad:
            out_BLD = out_BLD[:, : L - seq_pad, :]
        return out_BLD


class Gemma4TransformerBlock(TransformerBlock):
    """Gemma4 Transformer block with pre/post norms, layer_scalar, and parallel dense + MoE FFN."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        post_attention_layernorm: RMSNorm.Config
        post_feedforward_layernorm: RMSNorm.Config
        moe: Gemma4MoE.Config | None = None
        post_feedforward_layernorm_1: RMSNorm.Config | None = None
        post_feedforward_layernorm_2: RMSNorm.Config | None = None

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.post_attention_layernorm = config.post_attention_layernorm.build()

        self.ffn_norm = config.ffn_norm.build()
        self.post_feedforward_layernorm = config.post_feedforward_layernorm.build()
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build()

        self.moe_enabled = config.moe is not None
        self.moe: Gemma4MoE | None = None
        self.post_feedforward_layernorm_1: RMSNorm | None = None
        self.post_feedforward_layernorm_2: RMSNorm | None = None
        if self.moe_enabled:
            assert config.moe is not None
            assert config.post_feedforward_layernorm_1 is not None
            assert config.post_feedforward_layernorm_2 is not None
            self.moe = config.moe.build()
            self.post_feedforward_layernorm_1 = (
                config.post_feedforward_layernorm_1.build()
            )
            self.post_feedforward_layernorm_2 = (
                config.post_feedforward_layernorm_2.build()
            )

        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)

    def _init_self_buffers(
        self, *, buffer_device: torch.device | None = None
    ) -> None:
        device = buffer_device or self.layer_scalar.device
        self.layer_scalar = torch.ones(
            1, dtype=self.layer_scalar.dtype, device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_attn = self.attention(
            self.attention_norm(x),
            attention_masks,
            positions,
            segment_ids=segment_ids,
        )
        h = x + self.post_attention_layernorm(h_attn)

        h_mlp = self.feed_forward(self.ffn_norm(h))
        if self.moe_enabled:
            assert self.moe is not None
            assert self.post_feedforward_layernorm_1 is not None
            assert self.post_feedforward_layernorm_2 is not None
            h_mlp = self.post_feedforward_layernorm_1(h_mlp)
            h_moe = self.post_feedforward_layernorm_2(self.moe(h))
            h_ffn = h_mlp + h_moe
        else:
            h_ffn = h_mlp

        out = h + self.post_feedforward_layernorm(h_ffn)
        return out * self.layer_scalar.to(out.dtype)


class Gemma4Model(Decoder):
    """Gemma4 text decoder model supporting hybrid sliding/global attention and dense/MoE blocks."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 5376
        vocab_size: int = 262144
        enable_weight_tying: bool = True
        sliding_window_size: int = 1024
        final_logit_softcapping: float | None = 30.0

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            from torchtitan.experiments.tpu.gemma4.sharding import (
                set_gemma4_sharding_config,
            )
            from torchtitan.models.common.config_utils import (
                update_moe_aux_loss_configs,
            )

            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism
            seq_len = config.training.seq_len
            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                for idx, layer in enumerate(self.layers):
                    attn = layer.attention
                    assert isinstance(attn, Gemma4Attention.Config)
                    if attn.n_heads % tp != 0 or (
                        attn.n_kv_heads is not None and attn.n_kv_heads % tp != 0
                    ):
                        raise ValueError(
                            f"Tensor parallel degree (tp={tp}) must divide "
                            f"layer {idx} n_heads ({attn.n_heads}) and "
                            f"n_kv_heads ({attn.n_kv_heads})."
                        )
            if parallelism.context_parallel_degree > 1 and isinstance(
                self.layers[0].attention.inner_attention, VarlenAttention.Config
            ):
                raise NotImplementedError(
                    "Context Parallelism is not supported with VarlenAttention "
                    "in Gemma4."
                )

            local_global_batch_size = kwargs.get(
                "local_global_batch_size", config.training.local_batch_size
            )
            global_batch_size = kwargs.get(
                "global_batch_size", config.training.global_batch_size
            )
            if global_batch_size < 0:
                global_batch_size = local_global_batch_size

            update_moe_aux_loss_configs(
                self.layers,
                pp_enabled=parallelism.pipeline_parallel_degree > 1,
                local_global_batch_size=local_global_batch_size,
                global_batch_size=global_batch_size,
            )

            for layer in self.layers:
                assert isinstance(layer, Gemma4TransformerBlock.Config)
                attn = layer.attention
                assert isinstance(attn, Gemma4Attention.Config)
                if (
                    isinstance(attn.inner_attention, VarlenAttention.Config)
                    and attn.is_sliding
                    and attn.sliding_window_size is not None
                ):
                    attn.inner_attention = VarlenAttention.Config(
                        window_size=(attn.sliding_window_size - 1, 0)
                    )
                if (
                    layer.moe is not None
                    and hasattr(config, "debug")
                    and hasattr(config.debug, "moe_force_load_balance")
                ):
                    layer.moe.router._debug_force_load_balance = (
                        config.debug.moe_force_load_balance
                    )

            if (
                self.enable_weight_tying
                and parallelism.pipeline_parallel_degree > 1
            ):
                raise NotImplementedError(
                    "Weight tying is not supported with Pipeline Parallelism."
                )

            if isinstance(
                self.layers[0].attention.inner_attention, FlexAttention.Config
            ):
                block_size = self.layers[0].attention.inner_attention.block_size
                if isinstance(block_size, int) and seq_len % block_size != 0:
                    logger.warning(
                        f"training.seq_len ({seq_len}) is not divisible by "
                        f"FlexAttention block_size ({block_size})."
                    )

            set_gemma4_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            has_moe = any(
                isinstance(layer, Gemma4TransformerBlock.Config)
                and layer.moe is not None
                for layer in self.layers
            )
            assert isinstance(self.first_attention, GQAttention.Config)
            assert self.first_attention.head_dim is not None
            if has_moe:
                return get_moe_model_nparams_and_flops(
                    self,
                    model,
                    n_heads=self.first_attention.n_heads,
                    head_dims=2 * self.first_attention.head_dim,
                    seq_len=seq_len,
                )
            return get_dense_model_nparams_and_flops(
                model,
                n_layers=len(self.layers),
                n_heads=self.first_attention.n_heads,
                head_dims=2 * self.first_attention.head_dim,
                seq_len=seq_len,
                enable_weight_tying=self.enable_weight_tying,
            )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.sliding_window_size = config.sliding_window_size
        self.final_logit_softcapping = config.final_logit_softcapping

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> AttentionMasksType | None:
        attn_config = self.config.first_attention
        if attn_config is None:
            return None
        inner_attn = attn_config.inner_attention
        if isinstance(inner_attn, FlexAttention.Config):
            base_mods = [
                get_causal_mask_mod(),
                get_efficient_causal_mask_mod_for_packed_document(positions),
            ]
            basic_mask = self._create_flex_attention_mask(
                positions,
                attn_config,
                base_mods,
            )
            sliding_window_mask = self._create_flex_attention_mask(
                positions,
                attn_config,
                [
                    *base_mods,
                    get_sliding_window_mask_mod(self.sliding_window_size),
                ],
            )
            return {
                "basic_mask": basic_mask,
                "sliding_window_mask": sliding_window_mask,
            }
        return super().get_attention_masks(positions)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        output = super().forward(
            tokens, positions=positions, attention_masks=attention_masks
        )
        if (
            self.lm_head is not None
            and not self._skip_lm_head
            and self.final_logit_softcapping is not None
        ):
            output = torch.tanh(output / self.final_logit_softcapping) * (
                self.final_logit_softcapping
            )
        return output


@dataclass(kw_only=True, slots=True)
class MaxTextGemma4Config(Gemma4Model.Config):
    """Gemma4 config that builds Gemma4Model and uses MaxText-style FLOPs."""

    @classmethod
    def from_gemma4(cls, config: Gemma4Model.Config) -> "MaxTextGemma4Config":
        if isinstance(config, cls):
            return config
        return cls(
            **{
                field.name: getattr(config, field.name)
                for field in dataclasses.fields(Gemma4Model.Config)
                if field.init
            }
        )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
        if seq_len <= 0:
            raise ValueError("Sequence length must be positive")
        if not self.layers:
            raise ValueError("Model must have at least one decoder layer")

        nparams = sum(parameter.numel() for parameter in model.parameters())

        dim = self.dim
        # MaxText counts the LM-head projection but not the embedding lookup.
        forward_weight_flops = 2 * dim * self.vocab_size
        forward_attention_flops = 0

        for layer in self.layers:
            if not isinstance(layer, Gemma4TransformerBlock.Config):
                raise ValueError("Expected Gemma4TransformerBlock.Config")
            attention = layer.attention
            if not isinstance(attention, Gemma4Attention.Config):
                raise ValueError("Expected Gemma4Attention.Config")

            n_heads = attention.n_heads
            n_kv_heads = attention.n_kv_heads or n_heads
            head_dim = attention.head_dim or dim // n_heads

            if attention.use_k_eq_v:
                # Global K=V layers have only Q, K, and O projections (no V projection).
                qkv_out = (n_heads + n_kv_heads) * head_dim
            else:
                qkv_out = (n_heads + 2 * n_kv_heads) * head_dim
            forward_weight_flops += 2 * dim * (qkv_out + n_heads * head_dim)

            eff_seq_len = (
                min(seq_len, attention.sliding_window_size)
                if attention.is_sliding and attention.sliding_window_size
                else seq_len
            )
            forward_attention_flops += 2 * eff_seq_len * n_heads * head_dim

            feed_forward = layer.feed_forward
            if feed_forward is not None:
                if not isinstance(feed_forward, FeedForward.Config):
                    raise ValueError("Expected FeedForward.Config")
                forward_weight_flops += 6 * dim * feed_forward.w1.out_features

            moe = layer.moe
            if moe is not None:
                if not isinstance(moe, MoE.Config):
                    raise ValueError("Expected MoE.Config")
                forward_weight_flops += (
                    2 * dim * moe.num_experts
                    + 6 * dim * moe.experts.hidden_dim * moe.router.top_k
                )

        return nparams, 3 * (forward_weight_flops + forward_attention_flops)


__all__ = [
    "Gemma4Attention",
    "Gemma4Embedding",
    "Gemma4FeedForward",
    "Gemma4GroupedExperts",
    "Gemma4MoE",
    "Gemma4Model",
    "Gemma4RoPE",
    "Gemma4Router",
    "Gemma4TransformerBlock",
    "MaxTextGemma4Config",
    "QKVLinearKeqV",
]
