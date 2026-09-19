"""Register TPU attention kernels with HuggingFace's attention interface.

The TPU lane's splash-attention patch works by walking the model for modules
whose class name contains "ScaledDotProductAttention" -- torchtitan's own
attention modules. HF models have no such module: every attention layer calls
`ALL_ATTENTION_FUNCTIONS[config._attn_implementation]` inline, so the patch
finds nothing and raises.

That interface is the generic hook. Registering a kernel here reaches every HF
architecture that uses the standard attention path, without naming any of them.

The shape contract already matches: HF hands attention (B, H, S, D), which is
what `splash_sdpa` takes, and expects (B, S, H, D) back, which is one transpose.

Splash cannot serve every call -- it builds its own causal mask and wants
sequence lengths that are whole blocks. When it cannot, this falls back to
SDPA and *says so*, once per distinct reason. A silent fallback is how a
kernel that never fired gets reported as a speedup.
"""

import logging

import os
import torch
from torchtitan.models.common.attention import segment_ids_from_positions
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import AttentionInterface

from torchtitan.tools.logging import logger

_BLOCK = 512
# Block size for every splash stage. See the call site for the measurement.
# TODO(alekseyv): plumb this to splash_attention config
_SPLASH_BLOCK = int(os.environ.get("TPU_HF_SPLASH_BLOCK", "1024"))
# Block sizes for the two backward stages, overridable independently: the dkv
# stage sweeps the (q, kv) grid a second time and its optimum need not match
# the forward's.
_SPLASH_BLOCK_DKV = int(os.environ.get("TPU_HF_SPLASH_BLOCK_DKV", str(_SPLASH_BLOCK)))
_reported: set[str] = set()


def _tracing() -> bool:
    """True while Dynamo is tracing.

    Diagnostics cannot run inside a compiled region: Dynamo refuses to trace
    the logging call, and marking the helper `disable()`d turns the refusal
    into a hard "Skip calling torch.compiler.disable()'d function" error
    rather than a graph break. Guarding the call sites lets the branch fold
    away during tracing while staying live in eager, where the fallback
    decision is the one worth hearing about.
    """
    try:
        return torch.compiler.is_compiling()
    except Exception:
        return False


def _report(reason: str) -> None:
    """Report once. Call sites guard with `_tracing()`; see that helper."""
    if reason not in _reported:
        _reported.add(reason)
        logger.warning("splash_tpu: falling back to SDPA (%s)", reason)


def _log_once(key: str, message: str, *args) -> None:
    if key not in _reported:
        _reported.add(key)
        logger.info(message, *args)


def _unsupported(query, key, attention_mask, dropout, is_causal) -> str | None:
    """Why splash cannot take this call, or None if it can."""
    if attention_mask is not None:
        return "an attention_mask was supplied; splash builds its own causal mask"
    if dropout:
        return f"dropout={dropout}; splash has no dropout path"
    if is_causal is False:
        return "non-causal attention"
    q_len, kv_len = query.shape[-2], key.shape[-2]
    if q_len % _BLOCK or kv_len % _BLOCK:
        return f"sequence {q_len}x{kv_len} is not a multiple of {_BLOCK}"
    if query.device.type != "tpu":
        return f"tensors are on {query.device.type}, not tpu"
    return None



def _document_masked(module) -> bool:
    """True when the flavor asked for document-causal attention.

    Mirrors ``HFTransformerModel.get_attention_masks``: with
    ``attn_mask_type == "block_causal"`` the flex path builds a document
    BlockMask so packed samples do not attend across boundaries. Splash cannot
    take that BlockMask, so this adapter expresses the same mask as
    ``segment_ids`` derived from the per-document ``position_ids`` HF threads
    through ``**kwargs``.
    """
    cfg = getattr(module, "config", None)
    return getattr(cfg, "attn_mask_type", "causal") == "block_causal"

def splash_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask=None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
):
    """HF attention implementation backed by the TPU splash-attention kernel."""
    reason = _unsupported(query, key, attention_mask, dropout, is_causal)
    if reason is not None:
        if not _tracing():
            _report(reason)
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
        )

    from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa

    # splash_sdpa keeps enable_gqa only for signature compatibility, so the
    # kv heads have to be expanded here when the model is grouped-query.
    heads, kv_heads = query.shape[1], key.shape[1]
    if kv_heads != heads:
        repeat = heads // kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)

    # Only forward a genuine int window. Reading whatever `sliding_window`
    # happens to be off a wrapped module and handing it to the kernel is how
    # you get a bogus mask extent and a segfault rather than an exception.
    window = getattr(module, "sliding_window", None)
    if not isinstance(window, int) or window <= 0:
        window = None

    # The trainer hands the model one token-major stream per rank (batch
    # folded into the sequence: 8 x 2048 arrives as [1, 16384]). Without
    # segment ids the kernel runs ONE causal problem over the whole stream:
    # every sample attends to every earlier sample (not the semantics the
    # flex/GPU path has under block_causal) and the attention work grows
    # quadratically -- measured 2026-09-19 on Llama-3.2-1B, 4x v6e: splash
    # went from 151 ms to 618 ms per step and the step from 0.59 s to 1.15 s.
    # ``position_ids`` restart at 0 at every document boundary; turning them
    # into segment ids restores per-document masking AND lets the kernel skip
    # every block whose q and kv segments do not overlap.
    segment_ids = None
    if _document_masked(module):
        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            segment_ids = segment_ids_from_positions(position_ids)

    if not _tracing():
        _log_once(
            "splash_tpu args",
        "splash_tpu: first call q=%s k=%s v=%s dtype=%s scaling=%s window=%s",
            tuple(query.shape), tuple(key.shape), tuple(value.shape),
            query.dtype, scaling, window,
        )

    # splash_sdpa defaults every block to 512, which is the library's default
    # rather than a tuned one. Measured on Llama-3.2-1B at batch 8, seq 2048,
    # 4x v6e: 512 gives 17.08%, 1024 gives 18.12%, 2048 gives 17.78% and 256
    # gives 11.88% -- an interior optimum, +6.0% over the default, against a
    # repeat-measured run-to-run spread of 0.05pp. The forward blocks carry
    # most of it, which matches the profile: the two splash forward kernels
    # are 20.8% of device time against 12.5% for the backward dkv.
    out = splash_sdpa(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        segment_ids=segment_ids,
        scale=scaling,
        is_causal=True,
        local_window_size=window,
        block_q=_SPLASH_BLOCK,
        block_kv=_SPLASH_BLOCK,
        block_kv_compute=_SPLASH_BLOCK,
        block_q_dkv=_SPLASH_BLOCK_DKV,
        block_kv_dkv=_SPLASH_BLOCK_DKV,
        block_kv_dkv_compute=_SPLASH_BLOCK_DKV,
    )
    if not _tracing():
        _log_once(
            "splash_tpu fired",
            "splash_tpu: kernel active (heads=%d kv_heads=%d seq=%d head_dim=%d "
            "segment_ids=%s)",
            heads, kv_heads, query.shape[-2], query.shape[-1],
            "document" if segment_ids is not None else "none",
        )
    return out.transpose(1, 2).contiguous(), None


def register() -> None:
    """Make `attn_implementation="splash_tpu"` selectable on any HF model."""
    AttentionInterface._global_mapping["splash_tpu"] = splash_attention_forward


register()
