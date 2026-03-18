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
  """Create a JAX splash attention function for given dimensions."""
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

  @jax.jit
  def splash_fn(q, k, v):
    head_dim = q.shape[-1]
    scale = jnp.float32(1.0 / math.sqrt(head_dim))
    q_scaled = q * scale

    def single_batch_attn(q_b, k_b, v_b):
      return splash_kernel(q_b, k_b, v_b)

    return jax.vmap(single_batch_attn)(q_scaled, k, v)

  return splash_fn


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
) -> torch.Tensor:
  """Drop-in replacement for F.scaled_dot_product_attention using splash attention."""

  batch, n_heads, seq_len, head_dim = q.shape
  n_kv_heads = k.shape[1]

  if enable_gqa != (n_heads != n_kv_heads):
    raise ValueError(
        f"enable_gqa ({enable_gqa}) does not match inputs: "
        f"n_heads={n_heads}, n_kv_heads={n_kv_heads}"
    )

  # Dynamically generates unique cache keys for MHA vs MQA workloads
  cache_key = (
      batch,
      n_heads,
      n_kv_heads,
      seq_len,
      head_dim,
      is_causal,
      block_q,
      block_kv,
  )

  if cache_key not in _splash_cache:
    splash_fn = _make_splash_attention_fn(
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

    torch_splash_fn = custom_jax_kernel(splash_fn, name="splash_attention")
    _splash_cache[cache_key] = torch_splash_fn

  torch_splash_fn = _splash_cache[cache_key]

  if scale is not None:
    default_scale = 1.0 / math.sqrt(head_dim)
    if abs(scale - default_scale) > 1e-6:
      scale_ratio = scale / default_scale
      q = q * scale_ratio

  input_dtype = q.dtype
  result = torch_splash_fn(q, k, v)
  result = typing.cast(torch.Tensor, result)

  if result.dtype != input_dtype:
    result = result.to(input_dtype)
  return result
