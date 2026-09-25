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

Splash cannot serve every call -- it builds its own (causal or full) mask and wants
sequence lengths that are whole blocks. A call it cannot serve raises rather
than falling back: the checks are on shapes and arguments, so they fire at
trace time under torch.compile too, where a logged fallback would be silent.
"""

import torch
from transformers.modeling_utils import AttentionInterface

from torchtitan.tools.logging import logger

_reported: set[str] = set()


def _tracing() -> bool:
    """True while Dynamo is tracing.

    Diagnostics cannot run inside a compiled region: Dynamo refuses to trace
    the logging call, and marking the helper `disable()`d turns the refusal
    into a hard "Skip calling torch.compiler.disable()'d function" error
    rather than a graph break. Guarding the call sites lets the branch fold
    away during tracing while staying live in eager.
    """
    try:
        return torch.compiler.is_compiling()
    except Exception:
        return False


def _log_once(key: str, message: str, *args) -> None:
    if key not in _reported:
        _reported.add(key)
        logger.info(message, *args)


def _unsupported(
    query, key, attention_mask, dropout, causal, window, block_q, block_kv
) -> str | None:
    """Why splash cannot take this call, or None if it can."""
    if attention_mask is not None:
        return "an attention_mask was supplied; splash builds its own mask"
    if dropout:
        return f"dropout={dropout}; splash has no dropout path"
    if window is not None and not causal:
        return "non-causal sliding-window attention; splash's LocalMask is causal"
    q_len, kv_len = query.shape[-2], key.shape[-2]
    # The kernel clamps each block to the sequence length.
    block_q, block_kv = min(block_q, q_len), min(block_kv, kv_len)
    if q_len % block_q or kv_len % block_kv:
        return (
            f"sequence {q_len}x{kv_len} is not a multiple of the "
            f"{block_q}x{block_kv} splash blocks"
        )
    if query.device.type != "tpu":
        return f"tensors are on {query.device.type}, not tpu"
    return None


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
    """HF attention implementation backed by the TPU splash-attention kernel.

    Block sizes and the other splash options come from the job's
    `splash_attention_kernel` config, which `TPUHFTransformerModel.Config`
    records on the HF config every attention module holds.
    """
    sa_kwargs = module.config._tpu_splash_kwargs
    # Resolved as HF's own sdpa_attention_forward does: an explicit argument
    # wins, else the module's flag (False on encoders, True on decoders).
    causal = is_causal if is_causal is not None else getattr(
        module, "is_causal", True
    )
    # Only forward a genuine int window. Reading whatever `sliding_window`
    # happens to be off a wrapped module and handing it to the kernel is how
    # you get a bogus mask extent and a segfault rather than an exception.
    window = getattr(module, "sliding_window", None)
    if not isinstance(window, int) or window <= 0:
        window = None

    reason = _unsupported(
        query, key, attention_mask, dropout, causal, window,
        sa_kwargs["block_q"], sa_kwargs["block_kv"],
    )
    if reason is not None:
        raise ValueError(
            f"splash_tpu cannot serve this attention call: {reason}. "
            "Use attn='sdpa' for this model or shape."
        )

    from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa

    # Grouped-query attention is handled inside the kernel: splash maps each
    # query head to its kv head, so k/v go in unexpanded.
    heads, kv_heads = query.shape[1], key.shape[1]

    if not _tracing():
        _log_once(
            "splash_tpu args",
        "splash_tpu: first call q=%s k=%s v=%s dtype=%s scaling=%s "
        "causal=%s window=%s",
            tuple(query.shape), tuple(key.shape), tuple(value.shape),
            query.dtype, scaling, causal, window,
        )

    out = splash_sdpa(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        scale=scaling,
        is_causal=causal,
        local_window_size=window,
        enable_gqa=kv_heads != heads,
        **sa_kwargs,
    )
    if not _tracing():
        _log_once(
            "splash_tpu fired",
            "splash_tpu: kernel active (heads=%d kv_heads=%d seq=%d head_dim=%d)",
            heads, kv_heads, query.shape[-2], query.shape[-1],
        )
    return out.transpose(1, 2).contiguous(), None


def register() -> None:
    """Make `attn_implementation="splash_tpu"` selectable on any HF model."""
    AttentionInterface._global_mapping["splash_tpu"] = splash_attention_forward


register()
