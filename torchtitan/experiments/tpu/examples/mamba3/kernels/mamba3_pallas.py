# Copyright 2026 Google LLC

"""TPU-native Pallas kernels for Mamba-3 State Space Duality (SSD) chunked scan.

This module provides handwritten, hardware-fused Pallas TPU kernels directly
implementing the Mamba-3 state-space model formulation, including
continuous-time 2D RoPE rotation, exponential-trapezoidal discretization, and
chunked SSD scan with head group tiling for Google Cloud TPU v7x (Ironwood)
VectorCore VMEM limits. The kernels are integrated with PyTorch autograd via
custom operations.
"""

# pylint: disable=invalid-name,g-import-not-at-top

import logging
from typing import Any, Optional
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import torch
import torch.nn.functional as Fn

try:
  from jax.experimental.pallas import tpu as pltpu
except (ImportError, AttributeError):
  pltpu = None

_logger = logging.getLogger(__name__)

_CHUNK_SIZE = 256

try:
  from torch_tpu._internal import pallas as _pallas
  _TORCH_TPU_PALLAS_AVAILABLE = True
except ImportError:

  class _DummyJaxOp:
    """Fallback placeholder for JAX op wrapper when torch_tpu is unavailable."""

    def __init__(self, name: str, fn: Any) -> None:
      self.name = name
      self.fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
      raise RuntimeError(
          f"Op {self.name} requires Google Cloud TPU environment with"
          " torch_tpu._internal.pallas"
      )

    def register_autograd(
        self, backward_impl: Any, setup_context: Any = None
    ) -> None:
      pass

  class _DummyPallasModule:
    """Fallback placeholder module for torch_tpu._internal.pallas."""

    @staticmethod
    def jax_op(name: str, fn: Any) -> Any:
      return _DummyJaxOp(name, fn)

  _pallas = _DummyPallasModule()
  _TORCH_TPU_PALLAS_AVAILABLE = False


def _cumsum_via_matmul(x: jax.Array, axis_len: int) -> jax.Array:
  """Cumsum along axis 0 via lower-triangular matmul for Pallas TPU."""
  x = x.astype(jnp.float32)
  row = jax.lax.broadcasted_iota(jnp.int32, (axis_len, axis_len), 0)
  col = jax.lax.broadcasted_iota(jnp.int32, (axis_len, axis_len), 1)
  lower_incl = (col <= row).astype(jnp.float32)
  return jnp.matmul(
      lower_incl,
      x,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )


def _segsum_strict(x: jax.Array) -> jax.Array:
  """Intra-chunk strict causal decay mask (s < l only)."""
  t = x.shape[-1]
  row = jax.lax.broadcasted_iota(jnp.int32, (t, t), 0)
  col = jax.lax.broadcasted_iota(jnp.int32, (t, t), 1)
  strict_lower = (col < row).astype(x.dtype)
  x_rep = jnp.broadcast_to(x[..., :, None], x.shape + (t,))
  x_rep = x_rep * strict_lower
  x_segsum = _cumsum_via_matmul(x_rep, t)
  return x_segsum * strict_lower - (1.0 - strict_lower) * 1e30


def _segsum_safe(x: jax.Array) -> jax.Array:
  """Cross-chunk inclusive decay mask (s <= l)."""
  t = x.shape[-1]
  row = jax.lax.broadcasted_iota(jnp.int32, (t, t), 0)
  col = jax.lax.broadcasted_iota(jnp.int32, (t, t), 1)
  strict_lower = (col < row).astype(x.dtype)
  incl_diag = (col <= row).astype(x.dtype)
  x_rep = jnp.broadcast_to(x[..., :, None], x.shape + (t,))
  x_rep = x_rep * strict_lower
  x_segsum = _cumsum_via_matmul(x_rep, t)
  return x_segsum * incl_diag - (1.0 - incl_diag) * 1e30


def _mamba3_fused_chunk_kernel(
    x_ref: Any,
    adt_ref: Any,
    b_ref: Any,
    c_ref: Any,
    y_ref: Any,
    s_ref: Any,
) -> None:
  """Fused intra-chunk attention and inter-chunk recurrent state update Pallas TPU kernel."""
  chunk_idx = pl.program_id(1)

  @pl.when(chunk_idx == 0)
  def _init():
    s_ref[...] = jnp.zeros_like(s_ref)

  x_val = x_ref[0]
  adt_val = adt_ref[0, :, :, 0]
  b_val = b_ref[0]
  c_val = c_ref[0]

  x = jnp.swapaxes(x_val, 0, 1)
  adt = jnp.swapaxes(adt_val, 0, 1).astype(jnp.float32)
  b = jnp.swapaxes(b_val, 0, 1)
  c = jnp.swapaxes(c_val, 0, 1)

  chunk_size = x.shape[1]
  row_idx = jnp.arange(chunk_size)[:, None]
  col_idx = jnp.arange(chunk_size)[None, :]
  tril_strict = row_idx > col_idx
  tril_incl = row_idx >= col_idx

  adt_expanded = adt[:, None, :] * tril_incl[None, :, :]
  da_cumsum = jnp.sum(adt_expanded, axis=-1)

  cb = jnp.matmul(
      c, jnp.swapaxes(b, -1, -2), preferred_element_type=jnp.float32
  )
  da_diff = da_cumsum[:, :, None] - da_cumsum[:, None, :]
  decay_mat = jnp.where(
      tril_strict[None, :, :],
      jnp.exp(jnp.clip(da_diff, -100.0, 0.0)),
      0.0,
  ).astype(x.dtype)
  intra_k = cb.astype(x.dtype) * decay_mat
  y_intra = jnp.matmul(intra_k, x, preferred_element_type=jnp.float32)

  da_end = da_cumsum[:, -1:]
  decay_to_end = jnp.exp(jnp.clip(da_end - da_cumsum, -100.0, 0.0)).astype(
      x.dtype
  )
  b_dec = b * decay_to_end[:, :, None]
  bx_dec = jnp.matmul(
      jnp.swapaxes(b_dec, -1, -2), x, preferred_element_type=jnp.float32
  )

  s = s_ref[...]
  decay_from_start = jnp.exp(jnp.clip(da_cumsum, -100.0, 0.0)).astype(x.dtype)
  c_dec = c * decay_from_start[:, :, None]
  y_inter = jnp.matmul(
      c_dec, s.astype(x.dtype), preferred_element_type=jnp.float32
  )

  y_val = (y_intra + y_inter).astype(x.dtype)
  chunk_total_decay = jnp.exp(jnp.clip(da_end[:, 0], -100.0, 0.0))
  s_new = s * chunk_total_decay[:, None, None] + bx_dec
  s_ref[...] = s_new

  y_ref[0] = jnp.swapaxes(y_val, 0, 1)


def _pallas_mamba3_scan(
    x: jax.Array,
    adt: jax.Array,
    k_scaled: jax.Array,
    q_rot: jax.Array,
) -> jax.Array:
  """Executes the fused Pallas TPU scan with head group tiling for TPU v7x VMEM limits."""
  b, l, h, p = x.shape
  _, _, _, n = k_scaled.shape
  chunk_size = _CHUNK_SIZE
  num_chunks = l // chunk_size
  head_group = 8

  if h <= head_group or h % head_group != 0:
    grid = (b, num_chunks)
    adt_expanded = adt[..., None]

    in_specs = [
        pl.BlockSpec((1, chunk_size, h, p), lambda b, c: (b, c, 0, 0)),
        pl.BlockSpec((1, chunk_size, h, 1), lambda b, c: (b, c, 0, 0)),
        pl.BlockSpec((1, chunk_size, h, n), lambda b, c: (b, c, 0, 0)),
        pl.BlockSpec((1, chunk_size, h, n), lambda b, c: (b, c, 0, 0)),
    ]
    out_specs = pl.BlockSpec((1, chunk_size, h, p), lambda b, c: (b, c, 0, 0))
    scratch_shapes = [pltpu.VMEM((h, n, p), jnp.float32)]

    return pl.pallas_call(
        _mamba3_fused_chunk_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
    )(x, adt_expanded, k_scaled, q_rot)

  # Group heads into head_group slices to fit within TPU VectorCore VMEM (64MB)
  y_groups = []
  for h_idx in range(0, h, head_group):
    h_slice = slice(h_idx, h_idx + head_group)
    x_g = x[:, :, h_slice, :]
    adt_g = adt[:, :, h_slice, None]
    k_g = k_scaled[:, :, h_slice, :]
    q_g = q_rot[:, :, h_slice, :]

    grid = (b, num_chunks)
    in_specs = [
        pl.BlockSpec((1, chunk_size, head_group, p), lambda b, c: (b, c, 0, 0)),
        pl.BlockSpec((1, chunk_size, head_group, 1), lambda b, c: (b, c, 0, 0)),
        pl.BlockSpec((1, chunk_size, head_group, n), lambda b, c: (b, c, 0, 0)),
        pl.BlockSpec((1, chunk_size, head_group, n), lambda b, c: (b, c, 0, 0)),
    ]
    out_specs = pl.BlockSpec(
        (1, chunk_size, head_group, p), lambda b, c: (b, c, 0, 0)
    )
    scratch_shapes = [pltpu.VMEM((head_group, n, p), jnp.float32)]

    y_g = pl.pallas_call(
        _mamba3_fused_chunk_kernel,
        out_shape=jax.ShapeDtypeStruct(x_g.shape, x_g.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
    )(x_g, adt_g, k_g, q_g)
    y_groups.append(y_g)

  return jnp.concatenate(y_groups, axis=2)


def _make_inter_kernel(bl: int) -> Any:
  """Creates a Pallas TPU kernel for inter-chunk recurrent state projection."""
  del bl

  def _kernel(
      c_ref: Any, state_ref: Any, acum_ref: Any, yoff_ref: Any
  ) -> None:
    c_i = c_ref[0, 0, 0].astype(jnp.float32)
    state_in = state_ref[0, 0, 0].astype(jnp.float32)
    a_cumsum = acum_ref[0, 0, 0][:, 0]

    state_decay_out = jnp.exp(a_cumsum)
    cs = jnp.matmul(
        c_i,
        state_in,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    y_off = cs * state_decay_out[:, None]
    yoff_ref[0, 0, 0] = y_off.astype(yoff_ref.dtype)

  return _kernel


def _inter_chunk_pallas(
    c_c: jax.Array, states_in: jax.Array, a_cumsum_c: jax.Array
) -> jax.Array:
  """Applies inter-chunk recurrent state projection via Pallas TPU kernel."""
  b, h, nt, bl, n = c_c.shape
  p = states_in.shape[-1]
  kernel = _make_inter_kernel(bl)

  grid = (b, h, nt)
  bspec_c = pl.BlockSpec((1, 1, 1, bl, n), lambda i, j, k: (i, j, k, 0, 0))
  bspec_state = pl.BlockSpec((1, 1, 1, n, p), lambda i, j, k: (i, j, k, 0, 0))
  bspec_acum = pl.BlockSpec((1, 1, 1, bl, 1), lambda i, j, k: (i, j, k, 0, 0))

  return pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[bspec_c, bspec_state, bspec_acum],
      out_specs=pl.BlockSpec(
          (1, 1, 1, bl, p), lambda i, j, k: (i, j, k, 0, 0)
      ),
      out_shape=jax.ShapeDtypeStruct((b, h, nt, bl, p), c_c.dtype),
  )(c_c, states_in, a_cumsum_c)


def _chunkify(x: jax.Array, b: int, nt: int, bl: int) -> jax.Array:
  """Reshapes and permutes a sequence tensor into chunked representation."""
  rest = x.shape[2:]
  x = x.reshape((b, nt, bl) + rest)
  perm = (0, 3, 1, 2) + tuple(range(4, x.ndim))
  return jnp.transpose(x, perm)


def _ssd3_core(
    x: jax.Array,
    a: jax.Array,
    k_scaled: jax.Array,
    q_rot: jax.Array,
    intra_fn: Any,
    inter_fn: Any,
    s0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Core State Space Duality (SSD-3) chunked execution algorithm."""
  b, t, h, p = x.shape
  bl = min(_CHUNK_SIZE, t)
  nt = t // bl

  x_c = _chunkify(x, b, nt, bl)
  a_c = _chunkify(a, b, nt, bl)
  k_c = _chunkify(k_scaled, b, nt, bl)
  q_c = _chunkify(q_rot, b, nt, bl)

  y_diag, states, a_cumsum = intra_fn(x_c, a_c, k_c, q_c)
  a_cumsum = a_cumsum[..., 0]

  a_last = a_cumsum[..., -1]
  a_last_padded = jnp.pad(a_last, ((0, 0), (0, 0), (1, 0)))
  decay_chunk = jnp.exp(_segsum_safe(a_last_padded))
  initial_states = s0[:, :, None].astype(states.dtype)
  states_cat = jnp.concatenate([initial_states, states], axis=2)
  new_states = jnp.einsum(
      "bhzc,bhcnp->bhznp",
      decay_chunk,
      states_cat,
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  states_out = new_states[:, :, :-1].astype(states.dtype)
  s_final = new_states[:, :, -1].astype(states.dtype)

  y_off = inter_fn(q_c, states_out, a_cumsum[..., None])
  y = y_diag + y_off
  y = jnp.transpose(y, (0, 2, 3, 1, 4)).reshape(b, nt * bl, h, p)
  return y, s_final


def _angle_cumsum(angle_raw: jax.Array, dt: jax.Array) -> jax.Array:
  """Computes associative prefix sum of continuous rotary angles modulo 2*pi."""
  vals = jnp.tanh(angle_raw) * jnp.pi * dt[..., None]
  b, t, h, d = vals.shape
  bl = min(_CHUNK_SIZE, t)
  nt = t // bl
  vals_c = vals.reshape(b, nt, bl, h, d)
  intra_cumsum = jnp.cumsum(vals_c, axis=2)
  chunk_sum = intra_cumsum[:, :, -1]

  def _mod_add(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.mod(x + y, 2 * jnp.pi)

  init = jnp.zeros((b, 1, h, d), dtype=vals.dtype)
  seq = jnp.concatenate([init, chunk_sum], axis=1)
  offsets = jax.lax.associative_scan(_mod_add, seq, axis=1)[:, :nt]
  out = jnp.mod(intra_cumsum + offsets[:, :, None], 2 * jnp.pi)
  return out.reshape(b, t, h, d)


def _rope_trap_preprocess(
    q_raw: jax.Array,
    k_raw: jax.Array,
    q_bias: jax.Array,
    k_bias: jax.Array,
    angle_raw: jax.Array,
    dt: jax.Array,
    trap_raw: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Preprocesses queries and keys with 2D continuous RoPE and trapezoidal scaling."""
  b, t, h, n = q_raw.shape
  trap = jax.nn.sigmoid(trap_raw)
  dt_shifted = jnp.pad(dt[:, 1:], ((0, 0), (0, 1), (0, 0)))
  trap_shifted = jnp.pad(trap[:, 1:], ((0, 0), (0, 1), (0, 0)))
  gamma = dt * trap
  shifted_gamma = dt_shifted * (1.0 - trap_shifted)
  scale = gamma + shifted_gamma

  q_pre = q_raw + q_bias[None, None]
  k_pre = k_raw + k_bias[None, None]

  qk_dot_gamma = jnp.sum(q_pre * k_pre, axis=-1) * gamma

  angle_cumsum = _angle_cumsum(angle_raw, dt)
  cos = jnp.cos(angle_cumsum)
  sin = jnp.sin(angle_cumsum)

  def _rotate(x: jax.Array) -> jax.Array:
    x = x.reshape(b, t, h, n // 2, 2)
    x0, x1 = x[..., 0], x[..., 1]
    xo0 = x0 * cos - x1 * sin
    xo1 = x0 * sin + x1 * cos
    return jnp.stack([xo0, xo1], axis=-1).reshape(b, t, h, n)

  q_rot = _rotate(q_pre)
  k_rot = _rotate(k_pre)
  k_scaled = k_rot * scale[..., None]
  return q_rot, k_scaled, qk_dot_gamma


def _fwd_jax(
    x: jax.Array,
    q_raw: jax.Array,
    k_raw: jax.Array,
    q_bias: jax.Array,
    k_bias: jax.Array,
    adt: jax.Array,
    dt: jax.Array,
    trap_raw: jax.Array,
    angle_raw: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Direct JAX forward pass evaluating RoPE preprocessing and chunked SSD scan."""
  b, _, h, p = x.shape
  n = q_raw.shape[-1]
  q_rot, k_scaled, qk_dot_gamma = _rope_trap_preprocess(
      q_raw, k_raw, q_bias, k_bias, angle_raw, dt, trap_raw
  )
  if pltpu is not None and hasattr(pltpu, "VMEM"):
    y = _pallas_mamba3_scan(x, adt, k_scaled, q_rot)
  else:
    s0 = jnp.zeros((b, h, n, p), dtype=x.dtype)
    y, _ = _ssd3_core(
        x,
        adt,
        k_scaled,
        q_rot,
        _intra_chunk_plain,
        _inter_chunk_plain,
        s0,
    )
  return y, qk_dot_gamma


def _intra_chunk_one_plain(
    x_i: jax.Array, a_i: jax.Array, b_i: jax.Array, c_i: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Vectorized reference for single-chunk intra-chunk attention and state extraction."""
  a_cumsum = _cumsum_via_matmul(a_i[:, None], a_i.shape[0])[:, 0]
  decay_intra = jnp.exp(_segsum_strict(a_i))
  y_diag = jnp.einsum(
      "ln,sn,ls,sp->lp",
      c_i,
      b_i,
      decay_intra,
      x_i,
      precision=jax.lax.Precision.HIGHEST,
  )
  decay_states = jnp.exp(a_cumsum[-1] - a_cumsum)
  state = jnp.einsum(
      "ln,l,lp->np",
      b_i,
      decay_states,
      x_i,
      precision=jax.lax.Precision.HIGHEST,
  )
  return y_diag, state, a_cumsum[:, None]


def _intra_chunk_plain(
    x_c: jax.Array, a_c: jax.Array, b_c: jax.Array, c_c: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Batch-vectorized intra-chunk attention across all chunks and heads."""
  return jax.vmap(jax.vmap(jax.vmap(_intra_chunk_one_plain)))(
      x_c, a_c, b_c, c_c
  )


def _inter_chunk_one_plain(
    c_i: jax.Array, state_in: jax.Array, a_cumsum_i: jax.Array
) -> jax.Array:
  """Vectorized reference for single-chunk inter-chunk state projection."""
  state_decay_out = jnp.exp(a_cumsum_i)
  return jnp.einsum(
      "ln,np,l->lp",
      c_i,
      state_in,
      state_decay_out,
      precision=jax.lax.Precision.HIGHEST,
  )


def _inter_chunk_plain(
    c_c: jax.Array, states_in: jax.Array, a_cumsum_c: jax.Array
) -> jax.Array:
  """Batch-vectorized inter-chunk state projection across all chunks and heads."""
  return jax.vmap(jax.vmap(jax.vmap(_inter_chunk_one_plain)))(
      c_c, states_in, a_cumsum_c[..., 0]
  )


def _naive_ssd3_plain_jax(
    x: jax.Array,
    a: jax.Array,
    k_scaled: jax.Array,
    q_rot: jax.Array,
    initial_state: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Plain JAX reference implementation of SSD-3 scan for CPU and autograd surrogates."""
  return _ssd3_core(
      x,
      a,
      k_scaled,
      q_rot,
      _intra_chunk_plain,
      _inter_chunk_plain,
      initial_state,
  )


def _bwd_jax(
    x: jax.Array,
    q_raw: jax.Array,
    k_raw: jax.Array,
    q_bias: jax.Array,
    k_bias: jax.Array,
    adt: jax.Array,
    dt: jax.Array,
    trap_raw: jax.Array,
    angle_raw: jax.Array,
    dy: jax.Array,
    d_qk_dot_gamma: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
  """Computes VJP backward gradients for all inputs of the SISO scan in JAX."""
  b, _, h, p = x.shape
  n = q_raw.shape[-1]
  s0 = jnp.zeros((b, h, n, p), dtype=x.dtype)

  def surrogate(
      x_,
      q_raw_,
      k_raw_,
      q_bias_,
      k_bias_,
      adt_,
      dt_,
      trap_raw_,
      angle_raw_,
  ):
    q_rot, k_scaled, qk_dot_gamma = _rope_trap_preprocess(
        q_raw_, k_raw_, q_bias_, k_bias_, angle_raw_, dt_, trap_raw_
    )
    y, _ = _naive_ssd3_plain_jax(x_, adt_, k_scaled, q_rot, s0)
    return jnp.sum(y * dy) + jnp.sum(qk_dot_gamma * d_qk_dot_gamma)

  return jax.grad(surrogate, argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8))(
      x,
      q_raw,
      k_raw,
      q_bias,
      k_bias,
      adt,
      dt,
      trap_raw,
      angle_raw,
  )


# MIMO Ops
def _rotate_split_half(
    x: jax.Array, cos: jax.Array, sin: jax.Array, num_rope_angles: int
) -> jax.Array:
  """Rotates the RoPE-designated slice of features while preserving remaining channels."""
  half = x.shape[-1] // 2
  x0 = x[..., :num_rope_angles]
  x_mid = x[..., num_rope_angles:half]
  x1 = x[..., half : half + num_rope_angles]
  x_tail = x[..., half + num_rope_angles :]
  xo0 = x0 * cos - x1 * sin
  xo1 = x0 * sin + x1 * cos
  return jnp.concatenate([xo0, x_mid, xo1, x_tail], axis=-1)


def _segsum_strict_fused(x: jax.Array, r: int) -> jax.Array:
  """Computes strict causal decay mask for MIMO multi-channel tokens."""
  t = x.shape[-1]
  n_tok = t // r
  row = jax.lax.broadcasted_iota(jnp.int32, (t, t), 0)
  col = jax.lax.broadcasted_iota(jnp.int32, (t, t), 1)
  strict_lower_pos = (col < row).astype(x.dtype)
  tok_row = jax.lax.broadcasted_iota(jnp.int32, (n_tok, n_tok), 0)
  tok_col = jax.lax.broadcasted_iota(jnp.int32, (n_tok, n_tok), 1)
  tok_mask = (tok_col < tok_row).astype(x.dtype)
  strict_lower_tok = jnp.broadcast_to(
      tok_mask[:, None, :, None], (n_tok, r, n_tok, r)
  ).reshape(t, t)
  x_rep = jnp.broadcast_to(x[..., :, None], x.shape + (t,))
  x_rep = x_rep * strict_lower_pos
  x_segsum = _cumsum_via_matmul(x_rep, t)
  return x_segsum * strict_lower_tok - (1.0 - strict_lower_tok) * 1e30


def _fuse_tr(x: jax.Array) -> jax.Array:
  """Fuses token and channel rank dimensions for MIMO processing."""
  b, t, h, r = x.shape[:4]
  rest = x.shape[4:]
  x = jnp.moveaxis(x, 3, 2)
  return x.reshape((b, t * r, h) + rest)


def _unfuse_tr(x: jax.Array, r: int) -> jax.Array:
  """Unfuses token and channel rank dimensions back to separate axes."""
  b, tr, h = x.shape[:3]
  rest = x.shape[3:]
  t = tr // r
  x = x.reshape((b, t, r, h) + rest)
  return jnp.moveaxis(x, 2, 3)


def _fuse_adt_mimo(adt: jax.Array, r: int) -> jax.Array:
  """Expands and fuses scalar continuous decay across rank dimensions."""
  b, t, h = adt.shape
  zeros = jnp.zeros((b, t, r - 1, h), dtype=adt.dtype)
  fused = jnp.concatenate([adt[:, :, None, :], zeros], axis=2)
  return fused.reshape(b, t * r, h)


def _rope_trap_preprocess_mimo(
    v_raw: jax.Array,
    q_raw: jax.Array,
    k_raw: jax.Array,
    q_bias: jax.Array,
    k_bias: jax.Array,
    angle_raw: jax.Array,
    dt_bht: jax.Array,
    trap_raw_bht: jax.Array,
    psi: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Preprocesses multi-channel inputs with MIMO RoPE and cross-channel attention."""
  num_rope_angles = angle_raw.shape[-1]
  dt_bth = jnp.transpose(dt_bht, (0, 2, 1))
  trap_bth = jax.nn.sigmoid(jnp.transpose(trap_raw_bht, (0, 2, 1)))
  dt_shifted = jnp.pad(dt_bth[:, 1:], ((0, 0), (0, 1), (0, 0)))
  trap_shifted = jnp.pad(trap_bth[:, 1:], ((0, 0), (0, 1), (0, 0)))
  gamma = dt_bth * trap_bth
  shifted_gamma = dt_shifted * (1.0 - trap_shifted)
  scale = gamma + shifted_gamma

  q_pre = q_raw + q_bias[None, None]
  k_pre = k_raw + k_bias[None, None]

  qk_pre = jnp.einsum(
      "bthrn,bthsn->bthrs", q_pre, k_pre, precision=jax.lax.Precision.HIGHEST
  )
  psi_contract = jnp.einsum(
      "bthrs,hsp->bthrp", qk_pre, psi, precision=jax.lax.Precision.HIGHEST
  )
  diag = gamma[..., None, None] * v_raw[:, :, :, None, :] * psi_contract

  angle_cumsum = _angle_cumsum(angle_raw, dt_bth)
  cos = jnp.cos(angle_cumsum)[:, :, :, None, :]
  sin = jnp.sin(angle_cumsum)[:, :, :, None, :]

  q_rot = _rotate_split_half(q_pre, cos, sin, num_rope_angles)
  k_rot = _rotate_split_half(k_pre, cos, sin, num_rope_angles)
  k_scaled = k_rot * scale[..., None, None]
  psi_v = v_raw[:, :, :, None, :] * psi[None, None]
  return q_rot, k_scaled, psi_v, diag


def _make_intra_kernel_mimo(bl: int, r: int) -> Any:
  """Constructs a Pallas TPU kernel for intra-chunk MIMO multi-channel attention."""

  def _kernel(
      x_ref: Any,
      a_ref: Any,
      b_ref: Any,
      c_ref: Any,
      ydiag_ref: Any,
      state_ref: Any,
      acum_ref: Any,
  ) -> None:
    x_i = x_ref[0, 0, 0].astype(jnp.float32)
    a_i = a_ref[0, 0, 0][:, 0].astype(jnp.float32)
    b_i = b_ref[0, 0, 0].astype(jnp.float32)
    c_i = c_ref[0, 0, 0].astype(jnp.float32)

    a_cumsum = _cumsum_via_matmul(a_i[:, None], bl)[:, 0]
    decay_intra = jnp.exp(_segsum_strict_fused(a_i, r))
    cb = jnp.matmul(
        c_i,
        b_i.T,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    y_diag = jnp.matmul(
        cb * decay_intra,
        x_i,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )

    decay_states = jnp.exp(a_cumsum[-1] - a_cumsum)
    state = jnp.matmul(
        (b_i * decay_states[:, None]).T,
        x_i,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )

    ydiag_ref[0, 0, 0] = y_diag.astype(ydiag_ref.dtype)
    state_ref[0, 0, 0] = state.astype(state_ref.dtype)
    acum_ref[0, 0, 0] = a_cumsum.astype(acum_ref.dtype)[:, None]

  return _kernel


def _intra_chunk_pallas_mimo(
    x_c: jax.Array, a_c: jax.Array, b_c: jax.Array, c_c: jax.Array, r: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Executes fused intra-chunk MIMO attention via Pallas TPU kernel."""
  b, h, nt, bl, p = x_c.shape
  n = b_c.shape[-1]
  kernel = _make_intra_kernel_mimo(bl, r)
  a_c2 = a_c[..., None]

  grid = (b, h, nt)
  bspec_x = pl.BlockSpec((1, 1, 1, bl, p), lambda i, j, k: (i, j, k, 0, 0))
  bspec_a = pl.BlockSpec((1, 1, 1, bl, 1), lambda i, j, k: (i, j, k, 0, 0))
  bspec_bc = pl.BlockSpec((1, 1, 1, bl, n), lambda i, j, k: (i, j, k, 0, 0))
  bspec_state = pl.BlockSpec((1, 1, 1, n, p), lambda i, j, k: (i, j, k, 0, 0))

  return pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[bspec_x, bspec_a, bspec_bc, bspec_bc],
      out_specs=[bspec_x, bspec_state, bspec_a],
      out_shape=[
          jax.ShapeDtypeStruct((b, h, nt, bl, p), x_c.dtype),
          jax.ShapeDtypeStruct((b, h, nt, n, p), x_c.dtype),
          jax.ShapeDtypeStruct((b, h, nt, bl, 1), x_c.dtype),
      ],
  )(x_c, a_c2, b_c, c_c)


def _fwd_mimo_jax(
    v: jax.Array,
    q_raw: jax.Array,
    k_raw: jax.Array,
    q_bias: jax.Array,
    k_bias: jax.Array,
    adt: jax.Array,
    dt: jax.Array,
    trap_raw: jax.Array,
    angle_raw: jax.Array,
    psi: jax.Array,
    initial_state: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Forward pass for MIMO multi-channel Mamba-3 scan in JAX."""
  r = q_raw.shape[3]
  q_rot, k_scaled, psi_v, diag = _rope_trap_preprocess_mimo(
      v, q_raw, k_raw, q_bias, k_bias, angle_raw, dt, trap_raw, psi
  )
  x_fused = _fuse_tr(psi_v)
  q_fused = _fuse_tr(q_rot)
  k_fused = _fuse_tr(k_scaled)
  adt_fused = _fuse_adt_mimo(adt, r)

  def intra_fn(
      x_c: jax.Array, a_c: jax.Array, b_c: jax.Array, c_c: jax.Array
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _intra_chunk_plain_mimo(x_c, a_c, b_c, c_c, r)

  y_fused, s_final = _ssd3_core(
      x_fused,
      adt_fused,
      k_fused,
      q_fused,
      intra_fn,
      _inter_chunk_plain,
      initial_state,
  )
  y_full = _unfuse_tr(y_fused, r) + diag
  return y_full, s_final


def _intra_chunk_one_plain_mimo(
    x_i: jax.Array, a_i: jax.Array, b_i: jax.Array, c_i: jax.Array, r: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Performs intra-chunk attention for a single chunk of MIMO multi-channel inputs."""
  a_cumsum = _cumsum_via_matmul(a_i[:, None], a_i.shape[0])[:, 0]
  decay_intra = jnp.exp(_segsum_strict_fused(a_i, r))
  y_diag = jnp.einsum(
      "ln,sn,ls,sp->lp",
      c_i,
      b_i,
      decay_intra,
      x_i,
      precision=jax.lax.Precision.HIGHEST,
  )
  decay_states = jnp.exp(a_cumsum[-1] - a_cumsum)
  state = jnp.einsum(
      "ln,l,lp->np",
      b_i,
      decay_states,
      x_i,
      precision=jax.lax.Precision.HIGHEST,
  )
  return y_diag, state, a_cumsum[:, None]


def _intra_chunk_plain_mimo(
    x_c: jax.Array, a_c: jax.Array, b_c: jax.Array, c_c: jax.Array, r: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Batch-vectorized intra-chunk attention across all chunks and heads for MIMO."""
  fn = lambda x_i, a_i, b_i, c_i: _intra_chunk_one_plain_mimo(
      x_i, a_i, b_i, c_i, r
  )
  return jax.vmap(jax.vmap(jax.vmap(fn)))(x_c, a_c, b_c, c_c)


def _naive_ssd3_mimo_plain_jax(
    x: jax.Array,
    a: jax.Array,
    k_scaled: jax.Array,
    q_rot: jax.Array,
    initial_state: jax.Array,
    r: int,
) -> tuple[jax.Array, jax.Array]:
  """Reference execution of MIMO SSD-3 scan in plain JAX."""
  def intra_fn(
      x_c: jax.Array, a_c: jax.Array, b_c: jax.Array, c_c: jax.Array
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _intra_chunk_plain_mimo(x_c, a_c, b_c, c_c, r)

  return _ssd3_core(
      x, a, k_scaled, q_rot, intra_fn, _inter_chunk_plain, initial_state
  )


def _bwd_mimo_jax(
    v: jax.Array,
    q_raw: jax.Array,
    k_raw: jax.Array,
    q_bias: jax.Array,
    k_bias: jax.Array,
    adt: jax.Array,
    dt: jax.Array,
    trap_raw: jax.Array,
    angle_raw: jax.Array,
    psi: jax.Array,
    initial_state: jax.Array,
    dy_full: jax.Array,
    d_final_state: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
  """Computes VJP backward gradients for all inputs of the MIMO scan in JAX."""
  r = q_raw.shape[3]

  def surrogate(
      v_,
      q_raw_,
      k_raw_,
      q_bias_,
      k_bias_,
      adt_,
      dt_,
      trap_raw_,
      angle_raw_,
      psi_,
      s0_,
  ):
    q_rot, k_scaled, psi_v, diag = _rope_trap_preprocess_mimo(
        v_,
        q_raw_,
        k_raw_,
        q_bias_,
        k_bias_,
        angle_raw_,
        dt_,
        trap_raw_,
        psi_,
    )
    x_fused = _fuse_tr(psi_v)
    q_fused = _fuse_tr(q_rot)
    k_fused = _fuse_tr(k_scaled)
    adt_fused = _fuse_adt_mimo(adt_, r)
    y_fused, s_final = _naive_ssd3_mimo_plain_jax(
        x_fused, adt_fused, k_fused, q_fused, s0_, r
    )
    y_full = _unfuse_tr(y_fused, r) + diag
    return jnp.sum(y_full * dy_full) + jnp.sum(s_final * d_final_state)

  return jax.grad(surrogate, argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))(
      v,
      q_raw,
      k_raw,
      q_bias,
      k_bias,
      adt,
      dt,
      trap_raw,
      angle_raw,
      psi,
      initial_state,
  )


def _mamba3_bwd_surrogate(
    x: jax.Array,
    adt: jax.Array,
    k_scaled: jax.Array,
    q_rot: jax.Array,
    dy: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """VJP backward surrogate computing exact gradients with respect to SSD inputs."""
  b, _, h, p = x.shape
  _, _, _, n = k_scaled.shape
  s0 = jnp.zeros((b, h, n, p), dtype=x.dtype)

  def surrogate(x_, adt_, k_, q_):
    y, _ = _naive_ssd3_plain_jax(x_, adt_, k_, q_, s0)
    return jnp.sum(y.astype(jnp.float32) * dy.astype(jnp.float32))

  return jax.grad(surrogate, argnums=(0, 1, 2, 3))(x, adt, k_scaled, q_rot)


try:
  _pallas_ssd3_fwd = _pallas.jax_op(
      "mamba3::pallas_ssd3_fwd", _pallas_mamba3_scan
  )
  _pallas_ssd3_bwd = _pallas.jax_op(
      "mamba3::pallas_ssd3_bwd", _mamba3_bwd_surrogate
  )

  def _setup_context(ctx: Any, inputs: Any, output: Any) -> None:
    """Saves forward tensors in autograd context for SISO backward pass."""
    del output
    x, adt, k_scaled, q_rot = inputs[:4]
    ctx.save_for_backward(x, adt, k_scaled, q_rot)

  def _backward_impl(
      ctx: Any, grad_out: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes SISO backward gradients via registered Pallas JAX op."""
    x, adt, k_scaled, q_rot = ctx.saved_tensors
    return _pallas_ssd3_bwd(x, adt, k_scaled, q_rot, grad_out)

  _pallas_ssd3_fwd.register_autograd(
      _backward_impl, setup_context=_setup_context
  )

  _pallas_ssd3_mimo_fwd = _pallas.jax_op(
      "pallas_ssd3_mimo::fwd", _fwd_mimo_jax
  )
  _pallas_ssd3_mimo_bwd = _pallas.jax_op(
      "pallas_ssd3_mimo::bwd", _bwd_mimo_jax
  )

  def _mimo_setup_context(ctx: Any, inputs: Any, output: Any) -> None:
    """Saves forward tensors in autograd context for MIMO backward pass."""
    del output
    (
        v,
        q_raw,
        k_raw,
        q_bias,
        k_bias,
        adt,
        dt,
        trap_raw,
        angle_raw,
        psi,
        initial_state,
    ) = inputs
    ctx.save_for_backward(
        v,
        q_raw,
        k_raw,
        q_bias,
        k_bias,
        adt,
        dt,
        trap_raw,
        angle_raw,
        psi,
        initial_state,
    )

  def _mimo_backward_impl(
      ctx: Any, grad_y_full: torch.Tensor, grad_final_state: torch.Tensor
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
  ]:
    """Computes MIMO backward gradients via registered Pallas JAX op."""
    (
        v,
        q_raw,
        k_raw,
        q_bias,
        k_bias,
        adt,
        dt,
        trap_raw,
        angle_raw,
        psi,
        initial_state,
    ) = ctx.saved_tensors
    return _pallas_ssd3_mimo_bwd(
        v,
        q_raw,
        k_raw,
        q_bias,
        k_bias,
        adt,
        dt,
        trap_raw,
        angle_raw,
        psi,
        initial_state,
        grad_y_full,
        grad_final_state,
    )

  _pallas_ssd3_mimo_fwd.register_autograd(
      _mimo_backward_impl, setup_context=_mimo_setup_context
  )
except (ImportError, AttributeError, RuntimeError) as exc:
  _logger.warning(
      "Failed to register Mamba3 SSD JAX ops with autograd: %s", exc
  )
  _pallas_ssd3_fwd = None
  _pallas_ssd3_bwd = None
  _pallas_ssd3_mimo_fwd = None
  _pallas_ssd3_mimo_bwd = None


def ssd3_chunk_scan_pallas(
    v: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    b_bias: torch.Tensor,
    c_bias: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap_raw: torch.Tensor,
    angle_raw: torch.Tensor,
    d: Optional[torch.Tensor] = None,
    chunk_size: int = _CHUNK_SIZE,
    psi: Optional[torch.Tensor] = None,
    phi: Optional[torch.Tensor] = None,
    zeta: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    is_mimo: bool = False,
) -> torch.Tensor:
  """Executes the fused Pallas TPU SSD-3 scan with PyTorch tensor interface."""
  if is_mimo:
    return _execute_cpu_mimo_scan(
        v,
        b,
        c,
        b_bias,
        c_bias,
        adt,
        dt,
        trap_raw,
        angle_raw,
        d,
        psi,
        phi,
        zeta,
        z,
    )

  if _pallas_ssd3_fwd is not None and v.device.type in ("tpu", "xla"):
    try:
      return _execute_tpu_siso_scan(
          v, b, c, b_bias, c_bias, adt, dt, trap_raw, angle_raw, d, chunk_size
      )
    except (RuntimeError, ValueError) as err:
      _logger.warning(
          "Pallas SSD-3 SISO kernel failed with error: %s. "
          "Falling back to vectorized CPU scan.",
          err,
      )
      return _execute_cpu_siso_scan(
          v, b, c, b_bias, c_bias, adt, dt, trap_raw, angle_raw, d
      )
  return _execute_cpu_siso_scan(
      v, b, c, b_bias, c_bias, adt, dt, trap_raw, angle_raw, d
  )


def _execute_tpu_siso_scan(
    v: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    b_bias: torch.Tensor,
    c_bias: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap_raw: torch.Tensor,
    angle_raw: torch.Tensor,
    d: Optional[torch.Tensor],
    chunk_size: int,
) -> torch.Tensor:
  """Executes the fused Pallas TPU scan for SISO configuration with head grouping."""
  _, seq_len, nheads, _ = v.shape
  v = v.contiguous()
  b = b.contiguous()
  c = c.contiguous()
  adt = adt.contiguous()
  dt = dt.contiguous()
  trap_raw = trap_raw.contiguous()
  angle_raw = angle_raw.contiguous()
  c_bias = c_bias.contiguous()
  b_bias = b_bias.contiguous()

  effective_chunk_size = max(chunk_size, _CHUNK_SIZE)
  pad = (-seq_len) % effective_chunk_size
  if pad:
    v = Fn.pad(v, (0, 0, 0, 0, 0, pad)).contiguous()
    b = Fn.pad(b, (0, 0, 0, 0, 0, pad)).contiguous()
    c = Fn.pad(c, (0, 0, 0, 0, 0, pad)).contiguous()
    angle_raw = Fn.pad(angle_raw, (0, 0, 0, 0, 0, pad)).contiguous()
    dt = Fn.pad(dt, (0, 0, 0, pad)).contiguous()
    adt = Fn.pad(adt, (0, 0, 0, pad)).contiguous()
    trap_raw = Fn.pad(trap_raw, (0, 0, 0, pad)).contiguous()

  # 1. Preprocessing in PyTorch
  trap = torch.sigmoid(trap_raw)
  dt_shifted = Fn.pad(dt[:, 1:], (0, 0, 0, 1))
  trap_shifted = Fn.pad(trap[:, 1:], (0, 0, 0, 1))
  gamma = dt * trap
  scale = gamma + dt_shifted * (1.0 - trap_shifted)

  q_pre = c + c_bias[None, None]
  k_pre = b + b_bias[None, None]
  qk_dot_gamma = torch.sum(q_pre * k_pre, dim=-1) * gamma

  # RoPE rotation
  angle_cumsum = (
      torch.cumsum(
          torch.tanh(angle_raw) * 3.141592653589793 * dt[..., None], dim=1
      )
  ) % (2 * 3.141592653589793)
  cos = torch.cos(angle_cumsum)
  sin = torch.sin(angle_cumsum)

  def _rotate(t_in: torch.Tensor) -> torch.Tensor:
    t_pair = t_in.reshape(t_in.shape[0], t_in.shape[1], t_in.shape[2], -1, 2)
    x0, x1 = t_pair[..., 0], t_pair[..., 1]
    return torch.stack(
        [x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1
    ).reshape_as(t_in)

  q_rot = _rotate(q_pre)
  k_scaled = _rotate(k_pre) * scale[..., None]

  # 2. Hardware-fused Pallas TPU Scan in head groups
  head_group = 8
  if nheads <= head_group or nheads % head_group != 0:
    y = _pallas_ssd3_fwd(
        v.contiguous(),
        adt.contiguous(),
        k_scaled.contiguous(),
        q_rot.contiguous(),
    )
  else:
    y_splits = []
    for h_idx in range(0, nheads, head_group):
      h_slice = slice(h_idx, h_idx + head_group)
      y_group = _pallas_ssd3_fwd(
          v[:, :, h_slice].contiguous(),
          adt[:, :, h_slice].contiguous(),
          k_scaled[:, :, h_slice].contiguous(),
          q_rot[:, :, h_slice].contiguous(),
      )
      y_splits.append(y_group)
    y = torch.cat(y_splits, dim=2)

  if pad:
    y = y[:, :seq_len]
    qk_dot_gamma = qk_dot_gamma[:, :seq_len]
    v = v[:, :seq_len]

  if d is not None:
    y = y + (d[None, None, :] + qk_dot_gamma)[..., None] * v
  else:
    y = y + qk_dot_gamma[..., None] * v

  return y


def _execute_cpu_siso_scan(
    v: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    b_bias: torch.Tensor,
    c_bias: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap_raw: torch.Tensor,
    angle_raw: torch.Tensor,
    d: Optional[torch.Tensor],
) -> torch.Tensor:
  """Vectorized PyTorch CPU execution fallback for SISO Mamba-3 scan."""
  batch_size, seq_len, nheads, head_dim = v.shape
  d_state = b.shape[-1]

  trap = torch.sigmoid(trap_raw)
  dt_shifted = Fn.pad(dt[:, 1:], (0, 0, 0, 1))
  trap_shifted = Fn.pad(trap[:, 1:], (0, 0, 0, 1))
  gamma = dt * trap
  shifted_gamma = dt_shifted * (1.0 - trap_shifted)
  scale = gamma + shifted_gamma

  q_pre = c + c_bias[None, None, :, :]
  k_pre = b + b_bias[None, None, :, :]

  qk_dot_gamma = torch.sum(q_pre * k_pre, dim=-1) * gamma

  vals = torch.tanh(angle_raw) * torch.pi * dt[..., None]
  angle_cumsum = torch.cumsum(vals, dim=1) % (2 * torch.pi)
  cos = torch.cos(angle_cumsum)
  sin = torch.sin(angle_cumsum)

  def _rotate(inp):
    inp = inp.reshape(batch_size, seq_len, nheads, d_state // 2, 2)
    inp0, inp1 = inp[..., 0], inp[..., 1]
    out0 = inp0 * cos - inp1 * sin
    out1 = inp0 * sin + inp1 * cos
    return torch.stack([out0, out1], dim=-1).reshape(
        batch_size, seq_len, nheads, d_state
    )

  q_rot = _rotate(q_pre)
  k_rot = _rotate(k_pre)
  k_scaled = k_rot * scale[..., None]

  decay = torch.exp(adt)
  h = torch.zeros(
      batch_size,
      nheads,
      d_state,
      head_dim,
      device=v.device,
      dtype=torch.float32,
  )
  ys = []
  for l in range(seq_len):
    h = h * decay[:, l, :, None, None]
    y_l = torch.einsum("bhn,bhnp->bhp", q_rot[:, l].float(), h).to(v.dtype)
    ys.append(y_l)
    h = h + k_scaled[:, l, :, :, None].float() * v[:, l, :, None, :].float()
  y = torch.stack(ys, dim=1)

  if d is not None:
    y = y + (d[None, None, :] + qk_dot_gamma)[..., None] * v
  else:
    y = y + qk_dot_gamma[..., None] * v

  return y


def _execute_tpu_mimo_scan(
    v: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    b_bias: torch.Tensor,
    c_bias: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap_raw: torch.Tensor,
    angle_raw: torch.Tensor,
    d: Optional[torch.Tensor],
    chunk_size: int,
    psi: torch.Tensor,
    phi: torch.Tensor,
    zeta: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
  """Executes the fused Pallas TPU scan for MIMO multi-channel configuration."""
  batch_size, seq_len, nheads, head_dim = v.shape
  d_state = b.shape[-1]
  dt_bht = dt.transpose(1, 2).contiguous()
  trap_bht = trap_raw.transpose(1, 2).contiguous()
  v = v.contiguous()
  b = b.contiguous()
  c = c.contiguous()
  adt = adt.contiguous()
  angle_raw = angle_raw.contiguous()
  c_bias = c_bias.contiguous()
  b_bias = b_bias.contiguous()
  if psi is not None:
    psi = psi.contiguous()

  effective_chunk_size = max(chunk_size, _CHUNK_SIZE)
  pad = (-seq_len) % effective_chunk_size
  if pad:
    v = Fn.pad(v, (0, 0, 0, 0, 0, pad)).contiguous()
    b = Fn.pad(b, (0, 0, 0, 0, 0, 0, 0, pad)).contiguous()
    c = Fn.pad(c, (0, 0, 0, 0, 0, 0, 0, pad)).contiguous()
    angle_raw = Fn.pad(angle_raw, (0, 0, 0, 0, 0, pad)).contiguous()
    dt_bht = Fn.pad(dt_bht, (0, pad)).contiguous()
    adt = Fn.pad(adt, (0, 0, 0, pad)).contiguous()
    trap_bht = Fn.pad(trap_bht, (0, pad)).contiguous()

  initial_state = torch.zeros(
      batch_size,
      nheads,
      d_state,
      head_dim,
      device=v.device,
      dtype=v.dtype,
  ).contiguous()
  y_full, _ = _pallas_ssd3_mimo_fwd(
      v,
      c,
      b,
      c_bias,
      b_bias,
      adt,
      dt_bht,
      trap_bht,
      angle_raw,
      psi,
      initial_state,
  )
  if pad:
    y_full = y_full[:, :seq_len]
    v = v[:, :seq_len]

  psi_v_torch = v[:, :, :, None, :] * psi[None, None]
  if d is not None:
    y_full = y_full + d[None, None, :, None, None] * psi_v_torch

  z_reshaped = z.reshape(batch_size, seq_len, nheads, 1, head_dim)
  gated = y_full * Fn.silu(z_reshaped * zeta[None, None])
  y = torch.sum(gated * phi[None, None], dim=3)
  return y


def _execute_cpu_mimo_scan(
    v: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    b_bias: torch.Tensor,
    c_bias: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap_raw: torch.Tensor,
    angle_raw: torch.Tensor,
    d: Optional[torch.Tensor],
    psi: torch.Tensor,
    phi: torch.Tensor,
    zeta: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
  """Vectorized PyTorch CPU execution fallback for MIMO multi-channel Mamba-3 scan."""
  batch_size, seq_len, nheads, head_dim = v.shape
  d_state = b.shape[-1]

  trap = torch.sigmoid(trap_raw)
  dt_shifted = Fn.pad(dt[:, 1:], (0, 0, 0, 1))
  trap_shifted = Fn.pad(trap[:, 1:], (0, 0, 0, 1))
  gamma = dt * trap
  shifted_gamma = dt_shifted * (1.0 - trap_shifted)
  scale = gamma + shifted_gamma

  q_pre = c + c_bias[None, None, :, :, :]
  k_pre = b + b_bias[None, None, :, :, :]

  qk_pre = torch.einsum("bthrn,bthsn->bthrs", q_pre.float(), k_pre.float())
  psi_contract = torch.einsum(
      "bthrs,hsp->bthrp", qk_pre, psi.float()
  ).to(v.dtype)
  diag = gamma[..., None, None] * v[:, :, :, None, :] * psi_contract

  vals = torch.tanh(angle_raw) * torch.pi * dt[..., None]
  angle_cumsum = torch.cumsum(vals, dim=1) % (2 * torch.pi)
  cos = torch.cos(angle_cumsum)[:, :, :, None, :]
  sin = torch.sin(angle_cumsum)[:, :, :, None, :]

  num_rope_angles = angle_raw.shape[-1]
  half = d_state // 2

  def _rotate_split(inp):
    x0 = inp[..., :num_rope_angles]
    x_mid = inp[..., num_rope_angles:half]
    x1 = inp[..., half : half + num_rope_angles]
    x_tail = inp[..., half + num_rope_angles :]
    xo0 = x0 * cos - x1 * sin
    xo1 = x0 * sin + x1 * cos
    return torch.cat([xo0, x_mid, xo1, x_tail], dim=-1)

  q_rot = _rotate_split(q_pre)
  k_rot = _rotate_split(k_pre)
  k_scaled = k_rot * scale[..., None, None]
  psi_v = v[:, :, :, None, :] * psi[None, None]

  decay = torch.exp(adt)
  h = torch.zeros(
      batch_size,
      nheads,
      d_state,
      head_dim,
      device=v.device,
      dtype=torch.float32,
  )
  ys = []
  for l in range(seq_len):
    h = h * decay[:, l, :, None, None]
    y_l = torch.einsum("bhrn,bhnp->bhrp", q_rot[:, l].float(), h).to(v.dtype)
    ys.append(y_l)
    h_update = torch.einsum(
        "bhrn,bhrp->bhnp", k_scaled[:, l].float(), psi_v[:, l].float()
    )
    h = h + h_update

  y_full = torch.stack(ys, dim=1) + diag
  psi_v_torch = v[:, :, :, None, :] * psi[None, None]
  if d is not None:
    y_full = y_full + d[None, None, :, None, None] * psi_v_torch

  z_reshaped = z.reshape(batch_size, seq_len, nheads, 1, head_dim)
  gated = y_full * Fn.silu(z_reshaped * zeta[None, None])
  y = torch.sum(gated * phi[None, None], dim=3)
  return y
