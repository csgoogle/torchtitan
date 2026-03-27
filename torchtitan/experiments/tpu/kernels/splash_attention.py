"""Splash Attention integration for torch_tpu via Pallas.

Wraps JAX's splash_attention_kernel for use with PyTorch tensors on TPU.
Uses torch_tpu._internal.pallas.custom_jax_kernel() to bridge JAX -> PyTorch.
"""

import math
import typing
from typing import Optional

# JAX imports
import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
import jax.numpy as jnp
import torch
# torch_tpu Pallas bridge
from torch_tpu._internal import pallas

custom_jax_kernel = pallas.custom_jax_kernel

_DEFAULT_MASK_VALUE = splash_attention_kernel.DEFAULT_MASK_VALUE


def _make_splash_attention_fn(
    seq_len: int,
    n_heads: int,
    n_kv_heads: int,
    is_causal: bool = True,
    block_q: int = 512,
    block_kv: int = 512,
    block_kv_compute: int = 512,
    block_q_dkv: int = 512,
    block_kv_dkv: int = 512,
    block_kv_dkv_compute: int = 512,
    use_fused_bwd_kernel: bool = True,
):
  """Create JAX splash attention forward and backward functions.

  Returns:
    (splash_fn, splash_bwd_fn): forward function taking (q, k, v) returning
      (out, logsumexp), and backward function taking (q, k, v, out, logsumexp,
      grad_out) returning (grad_q, grad_k, grad_v) WITHOUT re-running the
      forward (uses saved residuals to call the backward kernel directly).
  """
  block_sizes = splash_attention_kernel.BlockSizes(
      block_q=min(block_q, seq_len),
      block_kv=min(block_kv, seq_len),
      block_kv_compute=min(block_kv_compute, seq_len),
      block_q_dkv=min(block_q_dkv, seq_len),
      block_kv_dkv=min(block_kv_dkv, seq_len),
      block_kv_dkv_compute=min(block_kv_dkv_compute, seq_len),
      use_fused_bwd_kernel=use_fused_bwd_kernel,
  )

  mask_shape = (seq_len, seq_len)
  if is_causal:
    single_mask = splash_attention_mask.CausalMask(shape=mask_shape)
  else:
    single_mask = splash_attention_mask.FullMask(mask_shape)

  multi_head_mask = splash_attention_mask.MultiHeadMask(
      masks=(single_mask,) * n_heads
  )

  is_mqa = n_kv_heads == 1

  if is_mqa:
    splash_kernel = splash_attention_kernel.make_splash_mqa(
        mask=multi_head_mask,
        head_shards=1,
        q_seq_shards=1,
        block_sizes=block_sizes,
    )
  else:
    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        head_shards=1,
        q_seq_shards=1,
        block_sizes=block_sizes,
    )

  # Static mask infos and kernel kwargs from creation time.
  # These are the same for every forward/backward call with the same config.
  fwd_mask_info = splash_kernel.fwd_mask_info
  dq_mask_info = splash_kernel.dq_mask_info
  dkv_mask_info = splash_kernel.dkv_mask_info
  _mask_value = splash_kernel.kwargs.get("mask_value", _DEFAULT_MASK_VALUE)
  _mask_function = splash_kernel.kwargs.get("mask_function", None)
  _attn_logits_soft_cap = splash_kernel.kwargs.get("attn_logits_soft_cap", None)
  _residual_checkpoint_name = splash_kernel.kwargs.get(
      "residual_checkpoint_name", None
  )
  _interpret = splash_kernel.kwargs.get("interpret", False)

  def _single_fwd_with_lse(q_b, k_b, v_b):
    """Forward for one batch element; returns (out_b, logsumexp_b)."""
    # q_b is already scaled (q * 1/sqrt(head_dim)) from the caller.
    out_b, (logsumexp_b,) = splash_attention_kernel._splash_attention_forward(
        fwd_mask_info,
        q_b,
        k_b,
        v_b,
        segment_ids=None,
        sinks=None,
        mask_value=_mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        residual_checkpoint_name=_residual_checkpoint_name,
        save_residuals=True,
        mask_function=_mask_function,
        attn_logits_soft_cap=_attn_logits_soft_cap,
        interpret=_interpret,
    )
    return out_b, logsumexp_b

  @jax.jit
  def splash_fn(q, k, v):
    """Forward: returns (out, logsumexp) for the whole batch."""
    head_dim = q.shape[-1]
    scale = jnp.float32(1.0 / math.sqrt(head_dim))
    q_scaled = q * scale
    out, logsumexp = jax.vmap(_single_fwd_with_lse)(q_scaled, k, v)
    return out, logsumexp

  def _single_bwd(q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b):
    """Backward for one batch element using precomputed residuals.

    Calls _splash_attention_bwd directly — no extra forward pass.

    Args:
      q_scaled_b: Scaled query tensor for a single batch element.
      k_b: Key tensor for a single batch element.
      v_b: Value tensor for a single batch element.
      out_b: Output of the forward pass for a single batch element.
      logsumexp_b: Logsumexp from the forward pass for a single batch element.
      g_b: Gradient of the loss with respect to the output (`out_b`).
    """
    res = (
        q_scaled_b,
        k_b,
        v_b,
        None,  # segment_ids
        None,  # sinks
        out_b,
        logsumexp_b,
        dq_mask_info,
        dkv_mask_info,
    )
    all_grads = splash_attention_kernel._splash_attention_bwd(
        False,  # save_residuals (nondiff)
        _mask_value,  # mask_value (nondiff)
        is_mqa,  # is_mqa (nondiff)
        block_sizes,  # block_sizes (nondiff)
        _residual_checkpoint_name,  # residual_checkpoint_name (nondiff)
        _mask_function,  # mask_function (nondiff)
        _attn_logits_soft_cap,  # attn_logits_soft_cap (nondiff)
        _interpret,  # interpret (nondiff)
        res,  # residuals
        g_b,  # do (output gradient)
    )
    # all_grads = (dfwd_mask_info=None, ddq_mask_info=None, ddkv_mask_info=None,
    #              dq_scaled, dk, dv, dsegment_ids=None, dsinks)
    return all_grads[3], all_grads[4], all_grads[5]  # dq_scaled, dk, dv

  def _scan_bwd_body(carry, args):
    """Single-element backward body for jax.lax.scan (sequential over batch)."""
    q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b = args
    dq_scaled_b, dk_b, dv_b = _single_bwd(
        q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b
    )
    return carry, (dq_scaled_b, dk_b, dv_b)

  @jax.jit
  def splash_bwd_fn(q, k, v, out, logsumexp, g):
    """Backward: uses saved (out, logsumexp) — no extra forward pass.

    Uses jax.lax.scan over the batch dimension so that the XLA backward program
    is sized for one element at a time (small static buffers).  This avoids the
    O(batch) growth in XLA program size that jax.vmap produces, allowing
    larger batch sizes to fit in HBM.

    Args:
      q: Query tensor.
      k: Key tensor.
      v: Value tensor.
      out: Output of the forward pass.
      logsumexp: Logsumexp values from the forward pass.
      g: Gradient of the loss with respect to the output (`out`).
    """
    head_dim = q.shape[-1]
    scale = jnp.float32(1.0 / math.sqrt(head_dim))
    q_scaled = q * scale

    _, (dq_scaled, dk, dv) = jax.lax.scan(
        _scan_bwd_body,
        None,
        (q_scaled, k, v, out, logsumexp, g),
    )
    # Chain rule: q_scaled = q * scale  =>  dq = dq_scaled * scale
    dq = dq_scaled * scale
    return dq, dk, dv

  return splash_fn, splash_bwd_fn


class _SplashAttentionFn(torch.autograd.Function):
  """torch.autograd.Function bridging splash forward and direct backward."""

  @staticmethod
  def forward(ctx, q, k, v, torch_fwd_fn, torch_bwd_fn):
    # torch_fwd_fn returns (out, logsumexp); both saved for the backward.
    out, logsumexp = torch_fwd_fn(q, k, v)
    ctx.save_for_backward(q, k, v, out, logsumexp)
    ctx.torch_bwd_fn = torch_bwd_fn
    return out

  @staticmethod
  def backward(ctx, grad_output):
    q, k, v, out, logsumexp = ctx.saved_tensors
    # Direct backward — no extra forward inside JAX.
    grad_q, grad_k, grad_v = ctx.torch_bwd_fn(
        q, k, v, out, logsumexp, grad_output
    )
    return (
        grad_q,
        grad_k,
        grad_v,
        None,
        None,
    )  # None for torch_fwd_fn, torch_bwd_fn


_splash_cache = {}


def splash_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    is_causal: bool = True,
    enable_gqa: bool = False,  # Kept for API compatibility
    block_q: int = 512,
    block_kv: int = 512,
    block_dkv: int = 512,
) -> torch.Tensor:
  """Replacement for F.scaled_dot_product_attention using splash attention.

  Supports backward pass: gradients flow through q, k, v via the JAX splash
  backward kernel. The forward saves (out, logsumexp) so the backward can call
  the pallas backward kernel directly — no extra forward pass at backward time.

  Args:
    block_dkv: Block size for the dk/dv backward kernel tiles. Smaller values
      reduce the XLA program size (enabling larger batch sizes) at a small
      efficiency cost. Default 512 matches block_kv; use 256 for large batches.
  """

  batch, n_heads, seq_len, head_dim = q.shape
  n_kv_heads = k.shape[1]

  if enable_gqa != (n_heads != n_kv_heads):
    raise ValueError(
        f"enable_gqa ({enable_gqa}) does not match inputs: "
        f"n_heads={n_heads}, n_kv_heads={n_kv_heads}"
    )

  cache_key = (
      batch,
      n_heads,
      n_kv_heads,
      seq_len,
      head_dim,
      is_causal,
      block_q,
      block_kv,
      block_dkv,
  )

  if cache_key not in _splash_cache:
    splash_fn, splash_bwd_fn = _make_splash_attention_fn(
        seq_len=seq_len,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        is_causal=is_causal,
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv,
        block_q_dkv=block_q,
        block_kv_dkv=block_kv,
        block_kv_dkv_compute=block_kv,
    )
    torch_fwd_fn = custom_jax_kernel(splash_fn, name="splash_attention")
    torch_bwd_fn = custom_jax_kernel(splash_bwd_fn, name="splash_attention_bwd")
    _splash_cache[cache_key] = (torch_fwd_fn, torch_bwd_fn)

  torch_fwd_fn, torch_bwd_fn = _splash_cache[cache_key]

  if scale is not None:
    default_scale = 1.0 / math.sqrt(head_dim)
    if abs(scale - default_scale) > 1e-6:
      scale_ratio = scale / default_scale
      q = q * scale_ratio  # plain PyTorch op — autograd handles gradient here

  input_dtype = q.dtype
  result = _SplashAttentionFn.apply(q, k, v, torch_fwd_fn, torch_bwd_fn)
  result = typing.cast(torch.Tensor, result)

  if result.dtype != input_dtype:
    result = result.to(input_dtype)
  return result
