# Copyright 2026 Google LLC

"""Splash attention with additive bias + in-kernel PRNG dropout for torch_tpu.

Wraps the vendored `sbd_splash/` kernel (a tokamax splash attention fork
adding additive bias and in-kernel PRNG dropout) as a PyTorch custom op
pair via `torch_tpu._internal.pallas.jax_op`, following the torch_tpu
pallas signature contract: tensor args are traced inputs, absent optional
tensors are passed as None (never dummy sentinels, so no unused operand can
be dead-code-eliminated out of the lowered kernel), and every configuration
knob -- mask choice, block sizes, dropout and bias parameters -- is a
non-tensor static op argument, value-hashed into the trace cache key.
Config changes therefore always take effect on the next call, at any
shape; nothing is captured silently at trace time. Backward-only statics
ride through the forward signature so `setup_context` can hand them to the
paired backward op.

Capabilities over stock tokamax splash (which has neither feature):
  - additive attention bias: full-rank `(H, S, S)` with EXACT dbias, or the
    cheap per-key rank `(H, 16, S)` (rows identical; row 0 is the logical
    bias) with optional rank-reduced trainable dbias;
  - in-kernel PRNG dropout on the attention probabilities (fixed-quantum
    keying; fold-rescale; quarter-area mask v4 + the dkv fused-compare
    frontier config), seeded by a per-batch `(B, 2)` int32 seed-word tensor
    so changing seeds never retraces.

Coverage matches the stock splash wrapper otherwise: causal, full
(non-causal), or sliding-window masks, an explicit softmax scale, and
MHA / grouped-KV / MQA head layouts (MQA routes through the kernel's
dedicated rank-reduced k/v entry).

Features off, the kernel is bit-exact with unmodified tokamax splash (the
family's standing no-regression contract). Dropout costs ~+12-13% and the
bias tiers ~+2% (per-key) / ~+12% (function-style) / ~+43% (full-rank) over
features-off at the validated op-points.

Scheduler note (measured): `use_experimental_scheduler` inverts sign across
op-points -- on a GQA fwd+bwd shape the tokamax default (False) was ~2.4%
faster, while on an MHA b4/S4096/H64 fwd+bwd shape False was an 8.06x
DETERMINISTIC regression. Sweep it per op-point x pass-set before changing
the default.

Provenance: the splash-bias-dropout kernel family (closed at ceiling
2026-08-27); every capability and price above is verifier-gated there.
"""

import math
from typing import Optional

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from torchtitan.experiments.tpu.kernels.sbd_splash import (
    splash_attention_kernel as sbd_kernel,
)
from torchtitan.experiments.tpu.kernels.sbd_splash import (
    splash_attention_mask as sbd_mask,
)


_MASK_VALUE = float(jnp.finfo(jnp.float32).min)
PER_KEY_ROWS = 16  # the padded per-key bias rank (bf16 (16, 128) tile alignment)


class SbdConfig:
  """Kernel configuration, forwarded as static op arguments on every call.

  Each value is part of the op's trace-cache key, so `update(...)` takes
  effect on the next call regardless of what shapes were traced before.
  Defaults pin the family's verified frontier configuration (fold-rescale +
  mask v4 + dkv fused-compare).
  """

  block_q: int = 512
  block_kv: int = 512
  block_kv_compute: int = 512
  block_q_dkv: int = 512
  block_kv_dkv: int = 512
  block_kv_dkv_compute: int = 512
  use_experimental_scheduler: bool = False  # tokamax default; see module note
  # Dropout (the verified frontier config; rate 0.0 disables the paths).
  dropout_rate: float = 0.0
  dropout_fold_rescale: bool = True
  dropout_mask_version: int = 4
  dropout_bwd_fused_compare: bool = True
  # Bias rank: 0 = full-rank (H, S, S) with exact dbias; PER_KEY_ROWS = the
  # per-key rank (H, 16, S). Trainable per-key dbias via the two knobs below
  # (MXU ones-matmul reduction is the verified best realization).
  bias_per_key_rows: int = 0
  bias_per_key_dbias: bool = False
  bias_per_key_dbias_mxu_reduce: bool = True
  attn_logits_soft_cap: Optional[float] = None

  @classmethod
  def update(cls, **kwargs):
    for k, v in kwargs.items():
      if v is not None:
        setattr(cls, k, v)

  @classmethod
  def static_args(cls) -> tuple:
    """The static op arguments, in the order the op signatures declare."""
    return (
        cls.block_q,
        cls.block_kv,
        cls.block_kv_compute,
        cls.block_q_dkv,
        cls.block_kv_dkv,
        cls.block_kv_dkv_compute,
        cls.use_experimental_scheduler,
        cls.attn_logits_soft_cap,
        cls.dropout_rate,
        cls.dropout_fold_rescale,
        cls.dropout_mask_version,
        cls.dropout_bwd_fused_compare,
        cls.bias_per_key_rows,
        cls.bias_per_key_dbias,
        cls.bias_per_key_dbias_mxu_reduce,
    )


def _build_kernel(
    seq_len: int,
    causal: bool,
    scale: float,
    local_window_size: int,
    use_dropout: bool,
    use_bias: bool,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_kv_dkv_compute: int,
    use_experimental_scheduler: bool,
    attn_logits_soft_cap: float | None,
    dropout_rate: float,
    dropout_fold_rescale: bool,
    dropout_mask_version: int,
    dropout_bwd_fused_compare: bool,
    bias_per_key_rows: int,
    bias_per_key_dbias: bool,
    bias_per_key_dbias_mxu_reduce: bool,
):
  """Constructs the fork kernel + static mask infos for one configuration."""
  cfg = sbd_kernel.SplashConfig(
      block_q=min(block_q, seq_len),
      block_kv=min(block_kv, seq_len),
      block_kv_compute=min(block_kv_compute, seq_len),
      block_q_dkv=min(block_q_dkv, seq_len),
      block_kv_dkv=min(block_kv_dkv, seq_len),
      block_kv_dkv_compute=min(block_kv_dkv_compute, seq_len),
      use_experimental_scheduler=use_experimental_scheduler,
      use_base2_exp=True,
      attn_logits_soft_cap=attn_logits_soft_cap,
      dropout_rate=dropout_rate if use_dropout else 0.0,
      dropout_fold_rescale=dropout_fold_rescale,
      dropout_mask_version=dropout_mask_version if use_dropout else 1,
      dropout_bwd_fused_compare=dropout_bwd_fused_compare and use_dropout,
      bias_per_key_rows=bias_per_key_rows if use_bias else 0,
      bias_per_key_dbias=bias_per_key_dbias and use_bias,
      bias_per_key_dbias_mxu_reduce=bias_per_key_dbias_mxu_reduce,
  )
  del scale  # applied to q by the callers; not part of the mask/config
  if local_window_size >= 0:
    # Sliding window takes precedence over `causal` (stock wrapper
    # convention): query i attends keys [i - local_window_size, i].
    mask = sbd_mask.LocalMask(
        shape=(seq_len, seq_len),
        window_size=(local_window_size, 0),
        offset=0,
        shard_count=1,
    )
  elif causal:
    mask = sbd_mask.CausalMask(shape=(seq_len, seq_len), shard_count=1)
  else:
    mask = sbd_mask.FullMask((seq_len, seq_len))
  kernel = sbd_kernel.make_splash_mha(
      mask=mask,
      config=cfg,
      q_seq_shards=1,
      save_residuals=False,
      mask_value=_MASK_VALUE,
      downcast_smem_data=False,
  )
  return cfg, kernel


def _kernel_statics(kernel):
  return (
      kernel.fwd_mask_info,
      kernel.dkv_mask_info,
      kernel.kwargs.get("mask_value", _MASK_VALUE),
      kernel.kwargs.get("mask_function", None),
      kernel.kwargs.get("fwd_mask_sparsity", 1.0),
      kernel.kwargs.get("dkv_mask_sparsity", 1.0),
  )


def sbd_attention_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    bias: jax.Array | None,
    dropout_seed: jax.Array | None,
    causal: bool,
    scale: float,
    local_window_size: int,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_kv_dkv_compute: int,
    use_experimental_scheduler: bool,
    attn_logits_soft_cap: float | None,
    dropout_rate: float,
    dropout_fold_rescale: bool,
    dropout_mask_version: int,
    dropout_bwd_fused_compare: bool,
    bias_per_key_rows: int,
    bias_per_key_dbias: bool,
    bias_per_key_dbias_mxu_reduce: bool,
) -> tuple[jax.Array, jax.Array]:
  """Forward. Layouts: q `(B, S, H, D)` torch-standard; k/v `(B, S, H_kv, D)`
  with `H_kv == H` (MHA), `H % H_kv == 0` (grouped KV), or `H_kv == 1`
  (MQA, rank-reduced kernel entry); `bias` is `(H, S, S)` full-rank or
  `(H, PER_KEY_ROWS, S)` per-key, shared across batch, or None;
  `dropout_seed` is `(B, 2)` int32 seed words, or None when dropout is off.
  The remaining args are trace-time statics; the dkv-only ones ride along
  for `setup_context` to hand to the backward op."""
  seq_len = q.shape[1]
  use_bias = bias is not None
  use_dropout = dropout_seed is not None
  is_mqa = k.shape[2] == 1
  cfg, kernel = _build_kernel(
      seq_len,
      causal,
      scale,
      local_window_size,
      use_dropout,
      use_bias,
      block_q,
      block_kv,
      block_kv_compute,
      block_q_dkv,
      block_kv_dkv,
      block_kv_dkv_compute,
      use_experimental_scheduler,
      attn_logits_soft_cap,
      dropout_rate,
      dropout_fold_rescale,
      dropout_mask_version,
      dropout_bwd_fused_compare,
      bias_per_key_rows,
      bias_per_key_dbias,
      bias_per_key_dbias_mxu_reduce,
  )
  fwd_mi, _, mask_value, mask_fn, fwd_sp, _ = _kernel_statics(kernel)

  scale_arr = jnp.array(scale, dtype=q.dtype)
  q_tr = jnp.transpose(q * scale_arr, (0, 2, 1, 3))  # (B, H, S, D)
  k_tr = jnp.transpose(k, (0, 2, 1, 3))
  v_tr = jnp.transpose(v, (0, 2, 1, 3))
  if is_mqa:
    # The kernel's MQA entry takes rank-reduced k/v: (S, D) per batch.
    k_tr = k_tr[:, 0]
    v_tr = v_tr[:, 0]

  def _fwd_b(q_b, k_b, v_b, seed_b):
    # pylint: disable=protected-access
    out_b, stats_b = sbd_kernel._splash_attention_forward(
        fwd_mi,
        q_b,
        k_b,
        v_b,
        bias,
        None,  # dropout_mask (explicit-mask path unused here)
        seed_b,
        None,  # segment_ids
        None,  # sinks
        mask_value,
        is_mqa,
        cfg,
        True,  # save_residuals
        mask_fn,
        bias_function=None,
        fwd_mask_sparsity=fwd_sp,
    )
    # pylint: enable=protected-access
    return out_b, stats_b["logsumexp"]

  out_tr, lse = jax.vmap(_fwd_b, in_axes=(0, 0, 0, 0 if use_dropout else None))(
      q_tr, k_tr, v_tr, dropout_seed
  )
  return jnp.transpose(out_tr, (0, 2, 1, 3)), lse


def sbd_attention_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    bias: jax.Array | None,
    dropout_seed: jax.Array | None,
    out: jax.Array,
    logsumexp: jax.Array,
    grad_out: jax.Array,
    causal: bool,
    scale: float,
    local_window_size: int,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_kv_dkv_compute: int,
    use_experimental_scheduler: bool,
    attn_logits_soft_cap: float | None,
    dropout_rate: float,
    dropout_fold_rescale: bool,
    dropout_mask_version: int,
    dropout_bwd_fused_compare: bool,
    bias_per_key_rows: int,
    bias_per_key_dbias: bool,
    bias_per_key_dbias_mxu_reduce: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Backward from saved residuals -- no forward re-run. Returns
  `(dq, dk, dv, dbias)`; `dbias` is a `(0,)` placeholder when bias is None
  (the op's output arity is fixed)."""
  seq_len = q.shape[1]
  use_bias = bias is not None
  use_dropout = dropout_seed is not None
  is_mqa = k.shape[2] == 1
  cfg, kernel = _build_kernel(
      seq_len,
      causal,
      scale,
      local_window_size,
      use_dropout,
      use_bias,
      block_q,
      block_kv,
      block_kv_compute,
      block_q_dkv,
      block_kv_dkv,
      block_kv_dkv_compute,
      use_experimental_scheduler,
      attn_logits_soft_cap,
      dropout_rate,
      dropout_fold_rescale,
      dropout_mask_version,
      dropout_bwd_fused_compare,
      bias_per_key_rows,
      bias_per_key_dbias,
      bias_per_key_dbias_mxu_reduce,
  )
  _, dkv_mi, mask_value, mask_fn, fwd_sp, dkv_sp = _kernel_statics(kernel)

  scale_arr = jnp.array(scale, dtype=q.dtype)
  q_tr = jnp.transpose(q * scale_arr, (0, 2, 1, 3))
  k_tr = jnp.transpose(k, (0, 2, 1, 3))
  v_tr = jnp.transpose(v, (0, 2, 1, 3))
  if is_mqa:
    k_tr = k_tr[:, 0]
    v_tr = v_tr[:, 0]
  out_tr = jnp.transpose(out, (0, 2, 1, 3))
  g_tr = jnp.transpose(grad_out, (0, 2, 1, 3))

  def _bwd_b(q_b, k_b, v_b, out_b, lse_b, g_b, seed_b):
    res = (
        q_b,
        k_b,
        v_b,
        bias,
        None,  # dropout_mask
        seed_b,
        None,  # segment_ids
        None,  # sinks
        out_b,
        lse_b,
        dkv_mi,
    )
    # pylint: disable=protected-access
    grads = sbd_kernel._splash_attention_bwd(
        False,  # save_residuals
        mask_value,
        is_mqa,
        cfg,
        mask_fn,
        None,  # bias_function
        fwd_sp,
        dkv_sp,
        res,
        g_b,
    )
    # pylint: enable=protected-access
    dq_b, dk_b, dv_b, dbias_b = grads[2], grads[3], grads[4], grads[5]
    if dbias_b is None:
      dbias_b = jnp.zeros((0,), dtype=q_b.dtype)
    return dq_b, dk_b, dv_b, dbias_b

  dq_tr, dk_tr, dv_tr, dbias_b = jax.vmap(
      _bwd_b, in_axes=(0, 0, 0, 0, 0, 0, 0 if use_dropout else None)
  )(q_tr, k_tr, v_tr, out_tr, logsumexp, g_tr, dropout_seed)

  dq = jnp.transpose(dq_tr, (0, 2, 1, 3)) * scale_arr  # chain rule for q * scale
  if is_mqa:
    dk = dk_tr[:, :, None, :]  # (B, S, D) -> (B, S, 1, D)
    dv = dv_tr[:, :, None, :]
  else:
    dk = jnp.transpose(dk_tr, (0, 2, 1, 3))
    dv = jnp.transpose(dv_tr, (0, 2, 1, 3))
  # bias is shared across the batch: sum the per-batch cotangents.
  if use_bias:
    dbias = jnp.sum(dbias_b, axis=0).astype(bias.dtype)
  else:
    dbias = jnp.zeros((0,), dtype=q.dtype)
  return dq, dk, dv, dbias


_sbd_fwd = pallas.jax_op("pallas::sbd_splash_attn_fwd", sbd_attention_fwd)
_sbd_bwd = pallas.jax_op("pallas::sbd_splash_attn_bwd", sbd_attention_bwd)

_NUM_STATIC_ARGS = 18  # causal, scale, window + the 15 SbdConfig values


def _setup_context(ctx, inputs, output):
  q, k, v, bias, dropout_seed = inputs[: len(inputs) - _NUM_STATIC_ARGS]
  out, logsumexp = output
  ctx.save_for_backward(q, k, v, bias, dropout_seed, out, logsumexp)
  ctx.static_args = tuple(inputs[len(inputs) - _NUM_STATIC_ARGS :])


def _backward(ctx, grad_out, grad_lse):
  del grad_lse
  q, k, v, bias, dropout_seed, out, logsumexp = ctx.saved_tensors
  dq, dk, dv, dbias = _sbd_bwd(
      q, k, v, bias, dropout_seed, out, logsumexp, grad_out, *ctx.static_args
  )
  dbias_ret = dbias if bias is not None else None
  return (dq, dk, dv, dbias_ret, None) + (None,) * _NUM_STATIC_ARGS


_sbd_fwd.register_autograd(_backward, setup_context=_setup_context)


def _seed_words(seed: int, batch: int, device: torch.device) -> torch.Tensor:
  """Deterministic per-batch (B, 2) int32 seed words from one python int.

  splitmix64-style mixing done torch-side; the kernel further mixes per
  (head, quantum) internally, so only per-batch decorrelation matters here.
  Passing seeds as a TENSOR keeps seed changes from retracing the op.
  """
  b = torch.arange(batch, dtype=torch.int64)
  z = (torch.tensor(seed, dtype=torch.int64) + b * 0x9E3779B97F4A7C15) & (
      (1 << 64) - 1
  )
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
  z = (z ^ (z >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
  z = z ^ (z >> 31)
  lo = (z & 0xFFFFFFFF).to(torch.uint32).view(torch.int32)
  hi = ((z >> 32) & 0xFFFFFFFF).to(torch.uint32).view(torch.int32)
  return torch.stack([lo, hi], dim=1).to(device)


def sbd_splash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    dropout_seed: int | torch.Tensor = 0,
    is_causal: bool = True,
    scale: Optional[float] = None,
    local_window_size: Optional[int] = None,
) -> torch.Tensor:
  """Splash attention with optional additive bias and PRNG dropout.

  Args:
    q: `(B, S, H, D)`; k/v: `(B, S, H_kv, D)` -- MHA (`H_kv == H`), grouped
      KV (`H % H_kv == 0`), and MQA (`H_kv == 1`, routed through the
      kernel's rank-reduced entry) are all supported.
    bias: optional, shared across batch -- `(H, S, S)` full-rank (exact
      dbias) or `(H, 16, S)` per-key when `SbdConfig.bias_per_key_rows ==
      PER_KEY_ROWS` (rows identical; trainable dbias via
      `SbdConfig.bias_per_key_dbias`).
    dropout_seed: python int (expanded to per-batch seed words) or a
      precomputed `(B, 2)` int32 tensor. Only used when
      `SbdConfig.dropout_rate > 0`. Seeding is explicit and deterministic
      (no torch RNG state): pass a fresh value every step (e.g. the global
      step), and under data/tensor parallelism fold the rank in as well --
      ranks passing the same value apply identical masks at the same local
      positions. Neither form retraces on seed change (the seed reaches
      the op as a tensor input).
    is_causal: causal (default) or full attention. Like every `SbdConfig`
      knob, a trace-time static: each value compiles its own kernel, so
      mixing configurations across calls at the same shapes is safe.
    scale: softmax scale; defaults to `1/sqrt(D)`.
    local_window_size: sliding-window attention -- query i attends keys
      `[i - local_window_size, i]`. Takes precedence over `is_causal`
      (stock wrapper convention).

  Returns:
    `(B, S, H, D)` attention output. Dropout follows the family convention:
    applied to the post-softmax probabilities with a single `1/(1-rate)`
    rescale; online-softmax state and logsumexp stay undropped.
  """
  # TT-UNFLATTEN-SHIM: must stay BELOW the docstring -- code above the string
  # literal would demote it from a docstring to a discarded expression and
  # leave __doc__ as None.
  from torchtitan.experiments.tpu.unflatten_shim import unflatten_tensor, flatten_tensor
  # `bias` is batch-shared (H, S, S), so it is never a token stream.
  was_flat = q.ndim == 3
  q = unflatten_tensor(q, expected_flat_ndim=3)
  k = unflatten_tensor(k, expected_flat_ndim=3)
  v = unflatten_tensor(v, expected_flat_ndim=3)

  if q.shape[2] % k.shape[2] != 0:
    raise ValueError(
        f"num q heads ({q.shape[2]}) must be a multiple of num kv heads"
        f" ({k.shape[2]})"
    )
  if SbdConfig.dropout_rate > 0.0:
    if isinstance(dropout_seed, torch.Tensor):
      seed_t = dropout_seed.to(device=q.device, dtype=torch.int32)
    else:
      seed_t = _seed_words(int(dropout_seed), q.shape[0], q.device)
  else:
    seed_t = None
  scale_val = 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
  window_val = -1 if local_window_size is None else int(local_window_size)
  out, _ = _sbd_fwd(
      q,
      k,
      v,
      bias,
      seed_t,
      is_causal,
      scale_val,
      window_val,
      *SbdConfig.static_args(),
  )
  # TT-UNFLATTEN-SHIM:
  if not was_flat:
    return out
  return flatten_tensor(out, original_flat_ndim=3)
