# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused Qwen3 QK RMSNorm + cos/sin RoPE forward as one Pallas pass.

XLA lowers ``q_norm -> k_norm -> CosSinRoPE`` into a chain of FP32 passes over
the query: a widening copy, an FP32 relayout, the normalized tensor, the two
rotate-half halves and the final rotation, each materialized in HBM because
the normalized FP32 tensor has several consumers. This kernel reads the BF16
input once and writes the BF16 result once.

The arithmetic is the one XLA runs for the stock modules, in the same order:

  xf  = f32(x)
  rs  = rsqrt(sum(xf * xf) * (1 / head_dim) + eps)
  n   = f32(bf16((xf * rs) * f32(weight)))       # RMSNorm, then .float()
  out = bf16(n * cos + rotate_half(n) * sin)

The forward is replaced; the backward is not. It differentiates the original
``F.rms_norm`` + ``CosSinRoPE.apply_rotary_emb`` ops with ``torch.func.vjp``,
so the gradients are computed by exactly the ops the unfused model uses.
"""

import torch
import torch.nn.functional as F

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from torch_tpu._internal import pallas

from torchtitan.models.common.rope import CosSinRoPE

# One grid step covers this many tokens; the query block is then
# 256 x 64 heads x 128 = 4 MiB of BF16.
_TOKENS_PER_BLOCK_BYTES = 4 * 1024 * 1024


def _kernel(eps, x_ref, w_ref, cos_ref, sin_ref, o_ref):
  head_dim = x_ref.shape[-1]
  x = x_ref[...].astype(jnp.float32)
  ss = jnp.sum(x * x, axis=-1, keepdims=True)
  rs = jax.lax.rsqrt(ss * (1.0 / head_dim) + eps)
  w = w_ref[...].astype(jnp.float32).reshape(1, 1, head_dim)
  n = ((x * rs) * w).astype(jnp.bfloat16).astype(jnp.float32)
  # rotate_half(n) = cat(-n[half:], n[:half]): a half-width lane roll, then
  # negate the first half. Multiplying by -1 is exact.
  lane = jax.lax.broadcasted_iota(jnp.int32, (1, 1, head_dim), 2)
  sign = jnp.where(lane < head_dim // 2, -1.0, 1.0).astype(jnp.float32)
  rot = pltpu.roll(n, head_dim // 2, 2) * sign
  cos = cos_ref[...][:, None, :]
  sin = sin_ref[...][:, None, :]
  o_ref[...] = (n * cos + rot * sin).astype(o_ref.dtype)


def _norm_rope_2d(x, w, cos, sin, eps):
  """x [T, N, H] bf16, w [H], cos/sin [T, H] f32 -> [T, N, H] bf16."""
  T, N, H = x.shape
  tokens = max(8, _TOKENS_PER_BLOCK_BYTES // (N * H * x.dtype.itemsize))
  while T % tokens:
    tokens //= 2
  return pl.pallas_call(
      lambda *refs: _kernel(eps, *refs),
      grid=(T // tokens,),
      in_specs=[
          pl.BlockSpec((tokens, N, H), lambda i: (i, 0, 0)),
          pl.BlockSpec((1, H), lambda i: (0, 0)),
          pl.BlockSpec((tokens, H), lambda i: (i, 0)),
          pl.BlockSpec((tokens, H), lambda i: (i, 0)),
      ],
      out_specs=pl.BlockSpec((tokens, N, H), lambda i: (i, 0, 0)),
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          # The FP32 temporaries of a 4 MiB block exceed the 32 MiB default.
          vmem_limit_bytes=64 * 1024 * 1024,
      ),
  )(x, w.reshape(1, H), cos, sin)


def jax_fused_qk_norm_rope(
    xq: jax.Array,
    xk: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    rope_cache: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
  """xq [B, L, Nq, H], xk [B, L, Nk, H], rope_cache [B, L, 2H] f32."""
  B, L, _, H = xq.shape
  cos = rope_cache[..., :H].reshape(B * L, H)
  sin = rope_cache[..., H:].reshape(B * L, H)
  q = _norm_rope_2d(xq.reshape(B * L, -1, H), wq, cos, sin, eps)
  k = _norm_rope_2d(xk.reshape(B * L, -1, H), wk, cos, sin, eps)
  return q.reshape(xq.shape), k.reshape(xk.shape)


fused_qk_norm_rope_op: torch._library.custom_ops.CustomOpDef = pallas.jax_op(
    "pallas::fused_qk_norm_rope_fwd", jax_fused_qk_norm_rope
)


def reference_qk_norm_rope(xq, xk, wq, wk, rope_cache, eps):
  """The unfused ops, as ``GQAttention`` runs them with RMSNorm + CosSinRoPE."""
  H = xq.shape[-1]
  q = F.rms_norm(xq, (H,), wq, eps)
  k = F.rms_norm(xk, (H,), wk, eps)
  return CosSinRoPE.apply_rotary_emb(q, k, rope_cache.unsqueeze(2))


def _setup_context(ctx, inputs, output):
  xq, xk, wq, wk, rope_cache, eps = inputs
  ctx.save_for_backward(xq, xk, wq, wk, rope_cache)
  ctx.eps = eps


def _backward(ctx, grad_q, grad_k):
  xq, xk, wq, wk, rope_cache = ctx.saved_tensors
  eps = ctx.eps
  _, vjp = torch.func.vjp(
      lambda a, b, c, d: reference_qk_norm_rope(a, b, c, d, rope_cache, eps),
      xq, xk, wq, wk,
  )
  dxq, dxk, dwq, dwk = vjp((grad_q, grad_k))
  return dxq, dxk, dwq, dwk, None, None


fused_qk_norm_rope_op.register_autograd(_backward, setup_context=_setup_context)


def fused_qk_norm_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    rope_cache: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Qwen3 q_norm/k_norm + cos/sin RoPE.

  Args:
    xq, xk: [B, L, N, H] contiguous activations.
    wq, wk: [H] norm weights.
    rope_cache: [B or 1, L, 1, 2H] cos/sin cache as ``CosSinRoPE`` shapes it.
    eps: RMSNorm epsilon.
  """
  B, L = xq.shape[:2]
  cache = rope_cache.reshape(rope_cache.shape[0], L, -1)
  if cache.shape[0] != B:
    cache = cache.expand(B, L, cache.shape[-1])
  return fused_qk_norm_rope_op(
      xq.contiguous(), xk.contiguous(), wq, wk, cache.contiguous(), eps
  )
