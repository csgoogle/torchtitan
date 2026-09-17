# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chunkwise-parallel gated delta rule for TPU.

Qwen3.5 spends 75% of its decoder layers in GatedDeltaNet linear attention.
Upstream offers two implementations, neither usable here:

  * `fla.ops.gated_delta_rule`, which is Triton/CUDA and is not vendored in
    internal.
  * `_torch_native_gated_delta`, a reference that steps the recurrence one
    token at a time in Python. At the sequence lengths used for training this
    unrolls into tens of thousands of graph nodes per layer, which is
    intractable to trace, compile, and schedule.

This module implements the same recurrence in chunkwise-parallel form: the
sequence is split into fixed-size chunks, everything inside a chunk is a dense
matmul over all chunks at once, and only the carried state is sequential. That
turns an `O(seq_len)`-deep graph into an `O(seq_len / chunk_size)`-deep one
built from MXU-friendly matmuls.

Derivation
----------
The recurrence, with state `S` of shape `(D_k, D_v)`, decay `a_t = exp(g_t)`
and update gate `beta_t`, is::

    u_t = beta_t * (v_t - a_t * S_{t-1}^T k_t)
    S_t = a_t * S_{t-1} + k_t u_t^T
    o_t = S_t^T q_t

Unrolling within a chunk, with `c_i` the inclusive cumulative sum of `g` from
the chunk start, gives::

    S_i = exp(c_i) S_in + sum_{j<=i} exp(c_i - c_j) k_j u_j^T

from which the three chunk-level identities below follow. Writing `D_ij =
exp(c_i - c_j)` for `j <= i` and `0` elsewhere::

    U = T @ (beta * V) - T @ (beta * exp(c) * K) @ S_in
    O = (exp(c) * Q) @ S_in + tril(D * (Q K^T)) @ U
    S_next = exp(c_last) * S_in + (exp(c_last - c) * K)^T @ U

`T = (I + diag(beta) @ strict_tril(D * (K K^T)))^{-1}` is the WY / UT
transform. `T` is never formed explicitly: both right-hand sides are stacked
and solved together by forward substitution. See
`_solve_unit_lower_triangular` for why its inner product is a multiply-and-
reduce rather than a matmul.

Numerical stability. Every exponent that survives into a product is
non-positive, so nothing overflows: the intermediate `k_j / exp(c_j)`, which
appears in the naive algebra and grows without bound as the decay accumulates,
is never materialized. The causal mask on `D` must be applied to the
*exponent*, not to `exp` of it -- above the diagonal `c_i - c_j` is positive
and large (at `g = -8` and `chunk_size = 64` it reaches `+504`), so masking
afterwards would compute `inf * 0 = NaN`.
"""

import torch


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
  """Matches `models.qwen3_5.model._l2norm` so the two backends agree."""
  return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _solve_unit_lower_triangular(
    strict_lower: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
  """Solves `(I + strict_lower) @ x = rhs` by forward substitution in fp32.

  Uses elementwise multiply + sum reduction rather than `torch.matmul` (`@`) so
  that XLA lowers the inner product to the TPU VPU in true IEEE float32. The
  MXU truncates float32 inputs to bfloat16; an earlier matmul-based inverse
  (Neumann series by repeated squaring) overflowed and cancelled to NaN for
  correlated keys because of that truncation.

  Args:
    strict_lower: `(..., C, C)` strictly lower triangular matrix.
    rhs: `(..., C, D)` right-hand side tensor.

  Returns:
    `(..., C, D)` solution `x` to `(I + strict_lower) @ x = rhs`.
  """
  size = strict_lower.shape[-1]
  cols: list[torch.Tensor] = []
  for i in range(size):
    rhs_i = rhs[..., i, :]
    if i == 0:
      cols.append(rhs_i)
    else:
      prev_x = torch.stack(cols, dim=-2)  # (..., i, D)
      row_l = strict_lower[..., i, :i].unsqueeze(-1)  # (..., i, 1)
      cols.append(rhs_i - (row_l * prev_x).sum(dim=-2))
  return torch.stack(cols, dim=-2)


def chunked_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
  """Runs the gated delta rule in chunkwise-parallel form.

  Numerically equivalent to `models.qwen3_5.model._torch_native_gated_delta`,
  including the L2 normalization and `D_k ** -0.5` query scaling it applies
  internally, but with a graph depth of `seq_len / chunk_size` rather than
  `seq_len`.

  Args:
    q: `(batch, seq_len, n_heads, key_head_dim)` queries.
    k: `(batch, seq_len, n_heads, key_head_dim)` keys.
    v: `(batch, seq_len, n_heads, value_head_dim)` values.
    g: `(batch, seq_len, n_heads)` log-space decay, non-positive.
    beta: `(batch, seq_len, n_heads)` update gate in `(0, 1)`.
    chunk_size: Tokens per chunk. Trades graph depth against the size of the
      `chunk_size x chunk_size` intra-chunk matrices.
    use_qk_l2norm: L2-normalize queries and keys inside the kernel, matching the
      FLA kernels' `use_qk_l2norm_in_kernel`.

  Returns:
    `(batch, seq_len, n_heads, value_head_dim)` outputs, in `q`'s dtype.

  Raises:
    ValueError: If `chunk_size` is not positive.
  """
  if chunk_size <= 0:
    raise ValueError(f"chunk_size must be positive, got {chunk_size}")

  out_dtype = q.dtype
  batch, seq_len, n_heads, key_head_dim = q.shape
  value_head_dim = v.shape[-1]

  # The recurrence accumulates over the whole sequence; run it in fp32.
  q = q.float()
  k = k.float()
  v = v.float()
  g = g.float()
  beta = beta.float()

  if use_qk_l2norm:
    q = _l2norm(q, dim=-1)
    k = _l2norm(k, dim=-1)
  q = q * (key_head_dim**-0.5)

  # Pad to a whole number of chunks. Padded positions get g=0 (no decay) and
  # beta=0, so their update `u` is zero and they leave the carried state
  # untouched; their outputs are sliced off at the end.
  pad = (-seq_len) % chunk_size
  if pad:
    q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad))
    k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
    v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
    g = torch.nn.functional.pad(g, (0, 0, 0, pad))
    beta = torch.nn.functional.pad(beta, (0, 0, 0, pad))
  padded_len = seq_len + pad
  num_chunks = padded_len // chunk_size

  def to_chunks(x: torch.Tensor) -> torch.Tensor:
    # (batch, len, heads, dim) -> (batch, heads, chunks, chunk, dim)
    return (
        x.view(batch, num_chunks, chunk_size, n_heads, -1)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )

  q = to_chunks(q)
  k = to_chunks(k)
  v = to_chunks(v)
  # (batch, len, heads) -> (batch, heads, chunks, chunk)
  g = (
      g.view(batch, num_chunks, chunk_size, n_heads)
      .permute(0, 3, 1, 2)
      .contiguous()
  )
  beta = (
      beta.view(batch, num_chunks, chunk_size, n_heads)
      .permute(0, 3, 1, 2)
      .contiguous()
  )

  # Inclusive cumulative decay from the start of each chunk.
  cum = g.cumsum(dim=-1)

  row = torch.arange(chunk_size, device=q.device)
  causal = row.unsqueeze(-1) >= row.unsqueeze(-2)
  strict_causal = row.unsqueeze(-1) > row.unsqueeze(-2)

  # decay[..., i, j] = exp(c_i - c_j) for j <= i, and exactly 0 above the
  # diagonal. The mask has to be applied to the *exponent*: for j > i the
  # exponent is positive and grows with the accumulated decay (at g = -8 and
  # chunk_size 64 it reaches +504), so exponentiating first and multiplying by
  # a 0/1 mask afterwards yields `inf * 0 = NaN`.
  decay = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).masked_fill(
      ~causal, -float("inf")
  )
  decay = decay.exp()

  beta_col = beta.unsqueeze(-1)
  entry_decay = cum.exp().unsqueeze(-1)

  # Solve (I + diag(beta) @ strict_tril(decay * K K^T)) @ [u_free, u_state]
  #       = [beta * V, beta * exp(c) * K]
  attn_kk = (k @ k.transpose(-1, -2)) * decay * strict_causal
  strict_lower = beta_col * attn_kk
  rhs = torch.cat([beta_col * v, beta_col * entry_decay * k], dim=-1)
  u_both = _solve_unit_lower_triangular(strict_lower, rhs)
  u_free, u_state = u_both.split([value_head_dim, key_head_dim], dim=-1)

  # O = exp(c) Q S_in + tril(decay * Q K^T) U
  q_entry = q * entry_decay
  attn_qk = (q @ k.transpose(-1, -2)) * decay

  # S_next = exp(c_last) S_in + (exp(c_last - c) K)^T U
  chunk_decay = cum[..., -1:].exp().unsqueeze(-1)
  k_exit = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)

  state = torch.zeros(
      batch,
      n_heads,
      key_head_dim,
      value_head_dim,
      dtype=q.dtype,
      device=q.device,
  )
  outputs = []
  for c in range(num_chunks):
    update = u_free[:, :, c] - u_state[:, :, c] @ state
    outputs.append(q_entry[:, :, c] @ state + attn_qk[:, :, c] @ update)
    state = (
        chunk_decay[:, :, c] * state
        + k_exit[:, :, c].transpose(-1, -2) @ update
    )

  # (batch, heads, chunks, chunk, dim) -> (batch, len, heads, dim)
  out = torch.stack(outputs, dim=2)
  out = (
      out.permute(0, 2, 3, 1, 4)
      .contiguous()
      .reshape(batch, padded_len, n_heads, value_head_dim)
  )
  return out[:, :seq_len].to(out_dtype)
