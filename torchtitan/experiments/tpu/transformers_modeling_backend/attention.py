"""Register TPU attention kernels with HuggingFace's attention interface.

The TPU lane's splash-attention patch works by walking the model for modules
whose class name contains "ScaledDotProductAttention" -- torchtitan's own
attention modules. HF models have no such module: every attention layer calls
`ALL_ATTENTION_FUNCTIONS[config._attn_implementation]` inline, so the patch
finds nothing and raises.

That interface is the generic hook. Registering a kernel here reaches every HF
architecture that uses the standard attention path, without naming any of them.

Two kernels serve the call, chosen per call by whether HF asks for dropout:

- ``splash_sdpa`` (the JAX-experimental splash) when ``dropout == 0``: it
  skips every (q, kv) block whose documents do not overlap, which on the
  lane's packed stream (8 x 2048 tokens folded into one 16k sequence, block
  1024) visits ~24 of 136 causal blocks per layer. Measured 2026-10-01 on
  Llama-3.2-1B, 4x v6e: the exact-but-unskipped alternative below costs 29 %
  of step throughput there (14,307 vs ~20,000 tok/s/device, +2.4 GiB).
- ``sbd_splash_attention`` (a tokamax splash fork) when ``dropout > 0``: the
  only kernel with an in-kernel dropout path, so these calls no longer fall
  back to SDPA. It masks packed documents exactly at token granularity but
  does not skip blocks on them (tokamax's MaskInfo is a static compacted
  block list); at the encoder op-point (512-token documents) that is a wash
  with the skipping kernel, so dropout models pay only the dropout itself.

Both take the same translation of HF's contract: ``is_causal`` passed through
(encoders run bidirectional), document boundaries and a 2D padding mask as
segment ids, grouped-KV / MQA heads, a sliding window. What neither can
express (an additive float mask, a sequence that is not a whole number of
blocks) falls back to SDPA and *says so*, once per distinct reason. A silent
fallback is how a kernel that never fired gets reported as a speedup.

Note that HF only builds a mask for implementations registered with its
``AttentionMaskInterface``; ``splash_tpu`` is not, so for every model on the
``masking_utils`` path (BERT, RoBERTa, Llama, ...) ``attention_mask`` arrives
here as ``None`` and the module's ``is_causal`` is the whole mask contract.
"""

import logging
from typing import Any

import torch
from torchtitan.experiments.tpu.attention import KERNEL_KWARG_NAMES
from torchtitan.models.common.attention import segment_ids_from_positions
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import AttentionInterface

from torchtitan.tools.logging import logger

# Kernel tuning (block sizes, backward variant, layouts) comes from the job
# config's ``splash_attention_kernel`` section -- the same ``sa_*`` fields the
# native lane's ``SplashAttention`` modules take -- pushed in through
# ``configure_kernel`` by ``workarounds.apply_patches``. The HF adapter is a
# function rather than a module, so the tuning lives here at module level.
# Empty until configured, in which case ``SbdConfig``'s own defaults apply.
_KERNEL_KWARGS: dict[str, Any] = {}
# The sbd kernel takes the six block sizes; the stock kernel's other knobs
# (dkv/dq blocks, fused/vmap backward, document-aware skipping, layouts) have
# no counterpart there and are reported once as ignored on the dropout path.
_SBD_BLOCK_KEYS = ("block_q", "block_kv", "block_kv_compute",
                   "block_q_dkv", "block_kv_dkv", "block_kv_dkv_compute")
# Keys in ``_reported`` that are informational (the kernel ran); everything
# else in that set is a fallback reason. Tests read this.
INFO_REPORTS = frozenset({"splash_tpu fired", "splash_tpu fired (sbd)", "splash_tpu args", "splash_tpu ignored"})
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
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
        )

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
    # into segment ids restores per-document masking AND (stock kernel) lets
    # it skip every block whose q and kv segments do not overlap.
    segment_ids = _segment_ids(module, attention_mask, kwargs.get("position_ids"))
    kernel_kwargs = _kernel_kwargs(query.shape[-2], key.shape[-2])

    if not _tracing():
        _log_once(
            "splash_tpu args",
            "splash_tpu: first call q=%s k=%s v=%s dtype=%s scaling=%s window=%s "
            "is_causal=%s dropout=%s kernel=%s",
            tuple(query.shape), tuple(key.shape), tuple(value.shape),
            query.dtype, scaling, window, is_causal, dropout, kernel_kwargs,
        )
    seg_desc = (
        "none" if segment_ids is None
        else "document+padding" if attention_mask is not None
        else "document"
    )

    if dropout:
        out = _sbd_forward(query, key, value, segment_ids, is_causal, scaling, window,
                           dropout, kernel_kwargs, seg_desc)
    else:
        out = _splash_forward(query, key, value, segment_ids, is_causal, scaling, window,
                              kernel_kwargs, seg_desc)
    return out, None


def _splash_forward(query, key, value, segment_ids, is_causal, scaling, window,
                    kernel_kwargs, seg_desc):
    """The stock splash kernel: document-aware block skipping, no dropout."""
    from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa

    # splash_sdpa keeps enable_gqa only for signature compatibility, so the
    # kv heads have to be expanded here when the model is grouped-query.
    heads, kv_heads = query.shape[1], key.shape[1]
    if kv_heads != heads:
        repeat = heads // kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
    # splash_sdpa defaults every block to 512, which is the library's default
    # rather than a tuned one; the HF registry's ``splash_attention_kernel``
    # section sets 1024. Measured on Llama-3.2-1B at batch 8, seq 2048, 4x
    # v6e: 512 gives 17.08%, 1024 gives 18.12%, 2048 gives 17.78% and 256
    # gives 11.88% -- an interior optimum, +6.0% over the default, against a
    # repeat-measured run-to-run spread of 0.05pp.
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
            "splash_tpu: kernel active (splash; heads=%d kv_heads=%d seq=%d head_dim=%d "
            "causal=%s segment_ids=%s)",
            heads, kv_heads, query.shape[-2], query.shape[-1], is_causal, seg_desc,
        )
    return out.transpose(1, 2).contiguous()


def _sbd_forward(query, key, value, segment_ids, is_causal, scaling, window,
                 dropout, kernel_kwargs, seg_desc):
    """The sbd kernel: in-kernel PRNG dropout, exact segment masking."""
    from torchtitan.experiments.tpu.kernels.splash_bias_dropout import (
        SbdConfig,
        sbd_splash_attention,
    )

    ignored = sorted(k for k in kernel_kwargs if k not in _SBD_BLOCK_KEYS)
    if ignored and not _tracing():
        _log_once("splash_tpu ignored", "splash_tpu: sbd kernel has no %s; ignored", ignored)
    blocks = {k: v for k, v in kernel_kwargs.items() if k in _SBD_BLOCK_KEYS}
    # The seed is a (B, 2) int32 tensor drawn from torch's RNG on the device:
    # fresh per call and per layer, reproducible under torch.manual_seed, and
    # a tensor operand so a new seed never retraces the op. Every SbdConfig
    # value is a static op argument hashed into the trace cache key, so
    # setting them per call is what the kernel expects.
    dropout_seed = torch.randint(
        -(2**31), 2**31 - 1, (query.shape[0], 2), dtype=torch.int32, device=query.device
    )
    SbdConfig.update(dropout_rate=float(dropout), **blocks)
    # HF hands (B, H, S, D); the kernel takes and returns torch-standard
    # (B, S, H, D), which is what HF expects back. Grouped-KV and MQA go to
    # the kernel as-is (no K/V expansion).
    out = sbd_splash_attention(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        dropout_seed=dropout_seed,
        segment_ids=segment_ids,
        is_causal=is_causal,
        scale=scaling,
        local_window_size=window,
    )
    if not _tracing():
        _log_once(
            "splash_tpu fired (sbd)",
            "splash_tpu: kernel active (sbd; heads=%d kv_heads=%d seq=%d head_dim=%d "
            "causal=%s dropout=%s segment_ids=%s)",
            query.shape[1], key.shape[1], query.shape[-2], query.shape[-1], is_causal,
            dropout, seg_desc,
        )
    return out


def register() -> None:
    """Make `attn_implementation="splash_tpu"` selectable on any HF model."""
    AttentionInterface._global_mapping["splash_tpu"] = splash_attention_forward


register()
