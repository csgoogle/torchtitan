# Implementation to circumvents view_as_complex op during backward pass.
# Workaround will only be applied if TPUJobConfig.apply_rope_complex_workaround
# is set to True.

import logging
import torch
from torchtitan.models.llama3.model import model as llama3_model

_ORIGINAL_APPLY_ROTARY_EMB = llama3_model.apply_rotary_emb


def apply_rotary_emb_tpu_safe(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    TPU-safe version of rotary embedding that avoids torch.view_as_complex.
    This replaces complex number rotation with explicit sin/cos math
    to prevent SIGSEGV on TPU backward passes.
    """
    # Reshape to isolate the real/imaginary component pairs.
    # Shape becomes [bs, seqlen, n_heads, head_dim/2, 2]
    xq_reshaped = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_reshaped = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_r, xq_i = xq_reshaped.unbind(-1)
    xk_r, xk_i = xk_reshaped.unbind(-1)

    # Reshape frequencies for broadcasting
    # (using helper already defined in the original model module)
    freqs_cis = llama3_model.reshape_for_broadcast(freqs_cis, xq_r, positions)
    freqs_cos = freqs_cis.real
    freqs_sin = freqs_cis.imag

    # Apply rotation using standard formula
    # i.e. (a + ib)(cos + isin) = (acos - bsin) + i(asin + bcos)
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # stack back together + flatten to original shape
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_patch():
    """Monkey-patches the llama3 model module to use the TPU-safe rotary embedding."""
    logging.info("Applying TPU workaround: Patching apply_rotary_emb...")
    llama3_model.apply_rotary_emb = apply_rotary_emb_tpu_safe


def revert_patch():
    """Reverts the monkey-patch by restoring the original apply_rotary_emb function."""
    logging.info("Reverting TPU workaround: Restoring apply_rotary_emb...")
    if llama3_model.apply_rotary_emb == apply_rotary_emb_tpu_safe:
      llama3_model.apply_rotary_emb = _ORIGINAL_APPLY_ROTARY_EMB
