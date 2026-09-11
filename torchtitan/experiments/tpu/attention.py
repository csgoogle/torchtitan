# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Splash attention as a first-class inner-attention backend for the TPU lane.

Upstream language models select their inner attention through
``get_attention_config(backend)``, which only knows ``flex`` / ``flex_flash``
/ ``varlen`` -- all of which need masks the model builds in
``Decoder.preprocess_inputs`` and none of which have a torch_tpu lowering.
``SplashAttention`` below is the TPU-lane replacement: a ``Module`` with its
own ``Config`` that ``GQAttention`` builds like any other inner attention, and
``SplashAttentionConverter`` swaps it in for every flex / varlen inner
attention of a model config *before* the model is built. Because the decoder
only builds ``BlockMask`` / varlen metadata when its first inner attention is a
``FlexAttention.Config`` / ``VarlenAttention.Config``, a converted model never
constructs a flex mask -- the document mask is derived from ``positions``
inside this module (``segment_ids_from_positions``).

Layout. Inner attentions receive token-major ``q/k: [T, H, K]``, ``v: [T, H,
V]``; the local batch is folded into ``T``. Two layouts are supported:

* ``batch_seq_len=None`` (packed): the whole ``T`` is one splash row of batch
  1 and the packed-document segment ids carry every document boundary. This
  is exactly upstream's flex ``causal AND same-document`` mask over ``T``.
  ``use_document_aware_block_skipping`` keeps the cost per-document rather
  than quadratic in ``T``.
* ``batch_seq_len=S``: ``T`` is reshaped to ``[T // S, S]`` rows that attend
  independently (the MaxText / pre-merge fork convention the block-size
  recipes were tuned on). Segment ids are still derived over the full ``T``,
  so a document straddling two rows keeps one id but the rows never see each
  other.

Below ``min_kernel_seq_len`` tokens per row, or off-TPU, a plain-torch
reference with an explicit boolean mask runs instead of the Pallas kernel, so
CPU tests and tiny debug shapes work unchanged.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.models.common.attention import (
    BaseAttention,
    FlexAttention,
    ScaledDotProductAttention,
    segment_ids_from_positions,
    VarlenAttention,
)
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger, warn_once


# Kernel-tuning knobs, in the names ``splash_sdpa`` takes. ``workarounds`` maps
# the recipe-level ``SplashAttentionKernelConfig`` (``sa_*``) onto these.
KERNEL_KWARG_NAMES = (
    "block_q",
    "block_kv",
    "block_dkv",
    "block_kv_compute",
    "block_q_dkv",
    "block_kv_dkv",
    "block_kv_dkv_compute",
    "block_q_dq",
    "block_kv_dq",
    "use_fused_bwd_kernel",
    "use_vmap_bwd",
    "use_document_aware_block_skipping",
    "q_layout",
    "k_layout",
    "v_layout",
)


class SplashAttention(Module):
    """Inner attention backed by the TPU splash-attention Pallas kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        is_causal: bool = True
        # Number of tokens before each query that it may attend to, on top of
        # itself (JAX ``LocalMask(window_size=(N, 0))``). ``None`` is full
        # causal. Flex's ``sliding_window_size=W`` (window including self)
        # maps to ``W - 1`` here; the converter does that translation.
        sliding_window_size: int | None = None
        # Row length for the batched layout; ``None`` runs the packed layout.
        batch_seq_len: int | None = None
        # Rows shorter than this run the reference path instead of the kernel.
        min_kernel_seq_len: int = 128
        # Tokamax variant of the kernel (no segment-id support).
        use_tokamax: bool = False
        block_q: int = 512
        block_kv: int = 512
        block_dkv: int = 512
        block_kv_compute: int = 512
        block_q_dkv: int = 512
        block_kv_dkv: int = 512
        block_kv_dkv_compute: int = 512
        block_q_dq: int | None = None
        block_kv_dq: int | None = None
        use_fused_bwd_kernel: bool = True
        use_vmap_bwd: bool = False
        use_document_aware_block_skipping: bool = True
        q_layout: str = "HEAD_DIM_MINOR"
        k_layout: str = "HEAD_DIM_MINOR"
        v_layout: str = "HEAD_DIM_MINOR"

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.is_causal = config.is_causal
        self.sliding_window_size = config.sliding_window_size
        self.batch_seq_len = config.batch_seq_len
        self.min_kernel_seq_len = config.min_kernel_seq_len
        self.use_tokamax = config.use_tokamax
        self.kernel_kwargs: dict[str, Any] = {
            name: getattr(config, name) for name in KERNEL_KWARG_NAMES
        }
        # Set by the legacy module-replacement patch in ``workarounds``: the
        # inner attention this one replaced, used as the off-kernel fallback.
        self.original_module: torch.nn.Module | None = None

    def configure_kernel(self, **kwargs: Any) -> None:
        """Apply recipe / CLI kernel tuning after construction.

        The converter runs at config-registry time, before the CLI is parsed,
        so ``--splash_attention_kernel.sa_*`` overrides reach the modules
        through here (``workarounds.apply_patches``).
        """
        for key, value in kwargs.items():
            if value is None:
                continue
            if key == "use_tokamax":
                self.use_tokamax = bool(value)
            elif key == "batch_seq_len":
                self.batch_seq_len = value
            elif key in self.kernel_kwargs:
                self.kernel_kwargs[key] = value
            else:
                raise ValueError(f"Unknown splash attention kernel option: {key}")

    # ------------------------------------------------------------------
    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_masks: Any = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool | None = None,
        positions: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError(
                "SplashAttention builds its mask from `positions`; it does not"
                " take runtime attention_masks (a FlexAttention BlockMask or"
                " varlen metadata reached the TPU splash backend)."
            )
        causal = self.is_causal if is_causal is None else is_causal
        # The positional names match the decoder's ``LocalMapConfig`` keys
        # (``set_gqa_inner_attention_local_map``); the legacy 4-D callers pass
        # ``[B, S, H, K]`` through the same slots.
        q, k, v = q_THK, k_THK, v_THV

        q_is_dt = isinstance(q, DTensor)
        q_mesh = q_placements = None
        if q_is_dt:
            q_mesh, q_placements = q.device_mesh, q.placements
            q = q.to_local()
        k = k.to_local() if isinstance(k, DTensor) else k
        v = v.to_local() if isinstance(v, DTensor) else v
        if isinstance(positions, DTensor):
            positions = positions.to_local()
        if isinstance(segment_ids, DTensor):
            segment_ids = segment_ids.to_local()

        # Token-major ``[T, H, K]`` (the language models) or the legacy
        # ``[B, S, H, K]`` (modules replaced by ``workarounds``, e.g. Flux).
        token_major = q.ndim == 3
        if token_major:
            num_tokens = q.shape[0]
            row_len = self.batch_seq_len or num_tokens
            if num_tokens % row_len != 0:
                raise ValueError(
                    f"SplashAttention batch_seq_len={row_len} does not divide"
                    f" the {num_tokens} tokens of the local batch"
                )
            batch = num_tokens // row_len
            q_BSHK = q.reshape(batch, row_len, *q.shape[1:])
            k_BSHK = k.reshape(batch, row_len, *k.shape[1:])
            v_BSHV = v.reshape(batch, row_len, *v.shape[1:])
        else:
            batch, row_len = q.shape[0], q.shape[1]
            q_BSHK, k_BSHK, v_BSHV = q, k, v

        if segment_ids is None and positions is not None:
            segment_ids = segment_ids_from_positions(positions)
        if segment_ids is not None:
            segment_ids = segment_ids.reshape(batch, row_len).to(torch.int32)

        if self._use_kernel(q, row_len):
            out_BSHV = self._kernel(
                q_BSHK, k_BSHK, v_BSHV, segment_ids, causal, scale, enable_gqa
            )
        elif self.original_module is not None and not token_major:
            out_BSHV = self.original_module(
                q_BSHK,
                k_BSHK,
                v_BSHV,
                scale=scale,
                enable_gqa=enable_gqa,
                is_causal=causal,
                **kwargs,
            )
        else:
            out_BSHV = reference_attention(
                q_BSHK,
                k_BSHK,
                v_BSHV,
                segment_ids,
                is_causal=causal,
                sliding_window_size=self.sliding_window_size,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        out = out_BSHV.reshape(num_tokens, *out_BSHV.shape[2:]) if token_major else out_BSHV
        if q_is_dt:
            out = DTensor.from_local(out, q_mesh, q_placements)
        return out

    def _use_kernel(self, q: torch.Tensor, row_len: int) -> bool:
        """Whether the Pallas kernel runs (else the reference path)."""
        return q.device.type == "tpu" and row_len >= self.min_kernel_seq_len

    def _kernel(
        self,
        q_BSHK: torch.Tensor,
        k_BSHK: torch.Tensor,
        v_BSHV: torch.Tensor,
        segment_ids_BS: torch.Tensor | None,
        causal: bool,
        scale: float | None,
        enable_gqa: bool,
    ) -> torch.Tensor:
        if self.use_tokamax:
            # The tokamax path transposes on the JAX side and takes [B, S, H, D].
            from torchtitan.experiments.tpu.kernels.tokamax_splash_attention import (  # pylint: disable=g-import-not-at-top
                splash_sdpa as tokamax_splash_sdpa,
            )

            if segment_ids_BS is not None:
                warn_once(
                    logger,
                    "tokamax splash attention ignores packed-document segment"
                    " ids; documents in a row attend to each other",
                )
            return tokamax_splash_sdpa(
                q_BSHK.contiguous(),
                k_BSHK.contiguous(),
                v_BSHV.contiguous(),
                scale=scale,
                is_causal=causal,
                local_window_size=self.sliding_window_size,
                enable_gqa=enable_gqa,
                **self.kernel_kwargs,
            )

        from torchtitan.experiments.tpu.kernels.splash_attention import (  # pylint: disable=g-import-not-at-top
            splash_sdpa,
        )

        # The Pallas kernel wants (B, H, S, D) and contiguous operands.
        out_BHSV = splash_sdpa(
            q_BSHK.transpose(1, 2).contiguous(),
            k_BSHK.transpose(1, 2).contiguous(),
            v_BSHV.transpose(1, 2).contiguous(),
            segment_ids=segment_ids_BS,
            scale=scale,
            is_causal=causal,
            local_window_size=self.sliding_window_size,
            enable_gqa=enable_gqa,
            **self.kernel_kwargs,
        )
        return out_BHSV.transpose(1, 2)


def reference_attention(
    q_BSHK: torch.Tensor,
    k_BSHK: torch.Tensor,
    v_BSHV: torch.Tensor,
    segment_ids_BS: torch.Tensor | None,
    *,
    is_causal: bool,
    sliding_window_size: int | None,
    scale: float | None,
    enable_gqa: bool,
) -> torch.Tensor:
    """Plain-torch attention with the same mask the splash kernel applies.

    ``causal AND same-document AND within-window``, materialized as a boolean
    ``[B, 1, S, S]`` mask -- only for rows too short for the kernel or for
    CPU reference runs.
    """
    row_len = q_BSHK.shape[1]
    idx = torch.arange(row_len, device=q_BSHK.device)
    mask = torch.ones(row_len, row_len, dtype=torch.bool, device=q_BSHK.device)
    if is_causal:
        mask &= idx[:, None] >= idx[None, :]
    if sliding_window_size is not None:
        mask &= (idx[:, None] - idx[None, :]) <= sliding_window_size
    mask = mask[None, None]
    if segment_ids_BS is not None:
        same_doc = segment_ids_BS[:, None, :, None] == segment_ids_BS[:, None, None, :]
        mask = mask & same_doc
    q_BHSK = q_BSHK.transpose(1, 2)
    k_BHSK = k_BSHK.transpose(1, 2)
    v_BHSV = v_BSHV.transpose(1, 2)
    if enable_gqa and k_BHSK.shape[1] != q_BHSK.shape[1]:
        n_rep = q_BHSK.shape[1] // k_BHSK.shape[1]
        k_BHSK = k_BHSK.repeat_interleave(n_rep, dim=1)
        v_BHSV = v_BHSV.repeat_interleave(n_rep, dim=1)
    with sdpa_kernel([SDPBackend.MATH]):
        out_BHSV = F.scaled_dot_product_attention(
            q_BHSK, k_BHSK, v_BHSV, attn_mask=mask, scale=scale
        )
    return out_BHSV.transpose(1, 2)


def _sliding_window_from_attention(
    attn_config: BaseAttention.Config, inner: Module.Config
) -> int | None:
    """Translate a model's sliding-window declaration to splash's convention.

    Returns the number of *previous* tokens a query may attend to (``None`` =
    unbounded). Flex-style ``sliding_window_size=W`` counts the query itself,
    varlen's ``window_size=(left, 0)`` does not.
    """
    if isinstance(inner, VarlenAttention.Config):
        left, right = inner.window_size
        if right != 0:
            raise NotImplementedError(
                "SplashAttention only supports causal windows (right == 0)"
            )
        return None if left in (-1, None) else int(left)
    window = getattr(attn_config, "sliding_window_size", None)
    if window is None:
        return None
    return int(window) - 1


class SplashAttentionConverter(ModelConfigConverter):
    """Swap every flex / varlen / sdpa inner attention for ``SplashAttention``.

    Runs on the model config tree before build, so the decoder never sees a
    ``FlexAttention.Config`` and never builds a flex ``BlockMask``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        # Template for every installed inner attention; the converter fills in
        # the per-layer ``sliding_window_size``.
        attention: SplashAttention.Config = dataclasses.field(
            default_factory=SplashAttention.Config
        )

    def __init__(self, config: Config) -> None:
        self.template = config.attention

    def convert(self, model_config: Module.Config) -> Module.Config:
        replaced = 0
        for _, attn_config, _, _ in list(
            model_config.traverse(BaseAttention.Config, recurse=True)
        ):
            inner = attn_config.inner_attention
            if isinstance(inner, SplashAttention.Config):
                continue
            if not isinstance(
                inner,
                (FlexAttention.Config, VarlenAttention.Config, ScaledDotProductAttention.Config),
            ):
                continue
            attn_config.inner_attention = dataclasses.replace(
                self.template,
                sliding_window_size=_sliding_window_from_attention(attn_config, inner),
            )
            replaced += 1
        logger.info("Replaced %d inner attention config(s) with SplashAttention", replaced)
        return model_config


__all__ = [
    "KERNEL_KWARG_NAMES",
    "reference_attention",
    "SplashAttention",
    "SplashAttentionConverter",
]
