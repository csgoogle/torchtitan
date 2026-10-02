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

Splash takes boolean structure -- causal / bidirectional / sliding window,
plus per-token segment ids -- and nothing else. This adapter translates what
HF hands it into that vocabulary: ``is_causal`` is passed through (encoders
run bidirectional), document boundaries and a 2D padding mask both become
segment ids. What it cannot express (an additive float bias, dropout, a
sequence that is not a whole number of blocks) falls back to SDPA and *says
so*, once per distinct reason. A silent fallback is how a kernel that never
fired gets reported as a speedup.

HF builds a mask only for implementations registered with its
``AttentionMaskInterface``; ``splash_tpu`` registers FA2's mask function, so
a 2D padding mask arrives here as-is (or as None when nothing is padded) and
the module's ``is_causal`` plus the packed ``position_ids`` carry the rest.
"""

import logging
from typing import Any

import torch
from torchtitan.experiments.tpu.attention import KERNEL_KWARG_NAMES
from torchtitan.models.common.attention import segment_ids_from_positions
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask
from transformers.modeling_utils import AttentionInterface

from torchtitan.tools.logging import logger

# Kernel tuning (block sizes, backward variant, layouts) comes from the job
# config's ``splash_attention_kernel`` section -- the same ``sa_*`` fields the
# native lane's ``SplashAttention`` modules take -- pushed in through
# ``configure_kernel`` by ``workarounds.apply_patches``. The HF adapter is a
# function rather than a module, so the tuning lives here at module level.
# Empty until configured, in which case ``splash_sdpa``'s own defaults apply.
_KERNEL_KWARGS: dict[str, Any] = {}
# The kernel needs both sequence lengths to be whole blocks, so every
# configured block is stepped down to the largest size that divides them
# (Mosaic's floor is 128) instead of gating on one number and launching with
# another -- a 512-gate / 1024-launch pair once let a 512-multiple that was
# not a 1024-multiple through the gate and into a kernel shape error.
_MIN_BLOCK = 128
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


def configure_kernel(**kwargs: Any) -> None:
    """Apply recipe / CLI kernel tuning (``--splash_attention_kernel.sa_*``).

    Same contract as ``SplashAttention.configure_kernel`` on the native lane:
    keys are the ``KERNEL_KWARG_NAMES``; ``None`` leaves a key untouched.
    """
    for key, value in kwargs.items():
        if value is None:
            continue
        if key not in KERNEL_KWARG_NAMES:
            raise ValueError(f"Unknown splash attention kernel option: {key}")
        _KERNEL_KWARGS[key] = value


def _pick_block(q_len: int, kv_len: int, preferred: int) -> int | None:
    """Largest block <= ``preferred`` that tiles both lengths, or None."""
    block = preferred
    while block >= _MIN_BLOCK:
        if q_len % block == 0 and kv_len % block == 0:
            return block
        block //= 2
    return None


def _kernel_kwargs(q_len: int, kv_len: int) -> dict[str, Any]:
    """The configured tuning with every block clamped to the sequence."""
    out = dict(_KERNEL_KWARGS)
    for key, value in out.items():
        if key.startswith("block_") and isinstance(value, int):
            out[key] = _pick_block(q_len, kv_len, value)
    return out


def _padding_mask(attention_mask) -> torch.Tensor | None:
    """The 2D per-token keep-mask behind ``attention_mask``, or None.

    HF's 2D mask is ``(B, S)`` int/bool with 1/True for real tokens and
    0/False for padding; that is exactly one segment-id bit. Anything else --
    a 4D additive float mask, a BlockMask -- carries structure this adapter
    does not decode (a float bias is the one thing the kernel truly cannot
    take; see ``splash_bias_dropout`` for that), so the caller falls back.
    The dtype test is deliberately shape/dtype-only: inspecting values would
    be a device sync in eager and a graph break under compile.
    """
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        return None
    if attention_mask.dtype.is_floating_point:
        return None
    return attention_mask != 0


def _unsupported(query, key, attention_mask, dropout) -> str | None:
    """Why splash cannot take this call, or None if it can."""
    if attention_mask is not None and _padding_mask(attention_mask) is None:
        return (
            f"attention_mask of shape {tuple(attention_mask.shape)} "
            f"{attention_mask.dtype} is not a 2D padding mask; splash takes "
            "boolean structure only (causal / window / segment ids)"
        )
    if dropout:
        return f"dropout={dropout}; splash has no dropout path"
    q_len, kv_len = query.shape[-2], key.shape[-2]
    if _pick_block(q_len, kv_len, _MIN_BLOCK) is None:
        return f"sequence {q_len}x{kv_len} is not a multiple of {_MIN_BLOCK}"
    if query.device.type != "tpu":
        return f"tensors are on {query.device.type}, not tpu"
    return None


def _segment_ids(module, attention_mask, position_ids) -> torch.Tensor | None:
    """Fold document boundaries and padding into one segment-id tensor.

    Documents come from ``position_ids`` (restart at 0 per packed sample) and
    are numbered from 1; padding tokens get segment 0, the MaxText convention
    (``tfds_data_processing_c4_mlperf`` / ``tunix_adapter``). Real tokens then
    never share a segment with padding, padding attends only to padding (so
    no all-masked row goes NaN), and the loss ignores those rows anyway.
    Returns None when neither source applies, which lets the kernel run its
    plain causal / full mask with no segment pass at all.
    """
    seg = None
    if _document_masked(module) and position_ids is not None:
        seg = segment_ids_from_positions(position_ids) + 1
    keep = _padding_mask(attention_mask)
    if keep is not None:
        if seg is None:
            seg = torch.ones_like(keep, dtype=torch.int32)
        seg = torch.where(keep, seg, torch.zeros_like(seg))
    return seg


def _fallback_mask(module, query, key, attention_mask, position_ids, is_causal):
    """The dense equivalent of the kernel's mask, for the SDPA fallback.

    Returns ``(mask, is_causal)`` for ``sdpa_attention_forward``. Documents and
    padding come from the same ``_segment_ids`` the kernel uses; causality and
    the sliding window are folded in, so the fallback computes what the kernel
    would have. An incoming 4D mask (the reason for falling back, e.g. an
    additive float mask) is kept and narrowed, never replaced. With no
    document, padding or window structure the call is passed through
    unchanged. Cost: one (B, S, S) boolean per call -- 256 MiB at a folded
    16k stream -- acceptable on a path that is already the slow one.
    """
    q_len, kv_len = query.shape[-2], key.shape[-2]
    two_d = attention_mask if (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2) else None
    seg = _segment_ids(module, two_d, position_ids) if q_len == kv_len else None
    window = getattr(module, "sliding_window", None)
    if not isinstance(window, int) or window <= 0:
        window = None
    if seg is None and window is None:
        return (None if two_d is not None else attention_mask), is_causal
    q_pos = torch.arange(q_len, device=query.device)[:, None] + (kv_len - q_len)
    k_pos = torch.arange(kv_len, device=query.device)[None, :]
    allowed = torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device)
    if is_causal:
        allowed = allowed & (k_pos <= q_pos)
    if window is not None:
        allowed = allowed & (q_pos - k_pos <= window)
    allowed = allowed[None]
    if seg is not None:
        allowed = allowed & (seg[:, :, None] == seg[:, None, :])
    allowed = allowed[:, None]  # (B, 1, S, S): broadcast over heads
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
        if attention_mask.dtype == torch.bool:
            return attention_mask & allowed, False
        return attention_mask.masked_fill(~allowed, torch.finfo(attention_mask.dtype).min), False
    return allowed, False


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
    # Same default as HF's own sdpa_attention_forward: a module that does not
    # say is the decoder-style causal one.
    if is_causal is None:
        is_causal = getattr(module, "is_causal", True)
    reason = _unsupported(query, key, attention_mask, dropout)
    if reason is not None:
        if not _tracing():
            _report(reason)
        # Fall back to the SAME attention, not just the same kernel interface:
        # HF hands this adapter no document mask (packing lives in
        # position_ids), so a bare SDPA call would let packed documents attend
        # across boundaries and ignore the sliding window.
        fb_mask, fb_causal = _fallback_mask(
            module, query, key, attention_mask, kwargs.get("position_ids"), is_causal
        )
        return sdpa_attention_forward(
            module, query, key, value, fb_mask,
            dropout=dropout, scaling=scaling, is_causal=fb_causal, **kwargs,
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
    segment_ids = _segment_ids(module, attention_mask, kwargs.get("position_ids"))
    kernel_kwargs = _kernel_kwargs(query.shape[-2], key.shape[-2])

    if not _tracing():
        _log_once(
            "splash_tpu args",
            "splash_tpu: first call q=%s k=%s v=%s dtype=%s scaling=%s window=%s "
            "is_causal=%s kernel=%s",
            tuple(query.shape), tuple(key.shape), tuple(value.shape),
            query.dtype, scaling, window, is_causal, kernel_kwargs,
        )

    # splash_sdpa defaults every block to 512, which is the library's default
    # rather than a tuned one; the HF registry's ``splash_attention_kernel``
    # section sets 1024. Measured on Llama-3.2-1B at batch 8, seq 2048, 4x
    # v6e: 512 gives 17.08%, 1024 gives 18.12%, 2048 gives 17.78% and 256
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
        is_causal=is_causal,
        local_window_size=window,
        **kernel_kwargs,
    )
    if not _tracing():
        _log_once(
            "splash_tpu fired",
            "splash_tpu: kernel active (heads=%d kv_heads=%d seq=%d head_dim=%d "
            "causal=%s segment_ids=%s)",
            heads, kv_heads, query.shape[-2], query.shape[-1], is_causal,
            "none" if segment_ids is None
            else "document+padding" if attention_mask is not None
            else "document",
        )
    return out.transpose(1, 2).contiguous(), None


def register() -> None:
    """Make `attn_implementation="splash_tpu"` selectable on any HF model."""
    AttentionInterface._global_mapping["splash_tpu"] = splash_attention_forward
    # HF's masking code early-exits -- and drops the caller's mask -- for any
    # implementation missing from its mask interface. FA2's mask function is
    # the right contract here: it returns the 2D (B, S) padding mask as given,
    # or None when nothing is padded, which is exactly what ``_padding_mask``
    # turns into segment ids. Causal / window / document structure stays this
    # adapter's job (from ``is_causal``, ``sliding_window`` and positions).
    AttentionMaskInterface._global_mapping["splash_tpu"] = flash_attention_mask


register()
