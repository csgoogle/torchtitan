"""Chunked gated delta rule on TPU with Tokamax's Pallas KDA kernels.

GatedDeltaNet's recurrence is Kimi Delta Attention (KDA) with one decay per
head instead of one per key channel: per token, ``S = exp(g) S``, then the delta
update ``S += beta k (v - k^T S)^T`` and ``o = scale * q^T S`` with
L2-normalized ``q`` and ``k``. ``tokamax_gated_delta`` broadcasts the per-head
log decay over the key channels and runs Tokamax's Mosaic TPU KDA forward and
backward kernels, which keep each 64-token chunk in VMEM, unlike the XLA
``chunked_gated_delta``, whose ``[chunk, chunk]`` intermediates round-trip
through HBM.

The JAX functions are bridged to PyTorch with ``torch_tpu``'s ``jax_op``. The
forward op also returns the backward kernel's residuals (the normalized
``q``/``k``, cumulative decays, chunk states, ...: the leaves of the
``jax.vjp`` closure), and the backward op runs only the backward kernels on
them instead of re-running the forward kernel. Under full rematerialization
the residuals live from the recomputed forward to the backward of the same
layer.
"""

from collections.abc import Sequence
import functools

import jax
import jax.numpy as jnp

# Workaround for https://github.com/openxla/tokamax/issues/240 (as in
# `kernels/tokamax_splash_attention.py`): older Tokamax releases apply
# `@jaxtyping.jaxtyped` at import time, and typeguard's AST instrumentation
# cannot parse jaxtyping shape strings. Swap in a pass-through decorator for the
# duration of the Tokamax import.
import jaxtyping

_real_jaxtyped = jaxtyping.jaxtyped


def _dummy_jaxtyped(*args, **kwargs):
  del kwargs
  if len(args) == 1 and callable(args[0]):
    return args[0]
  return lambda f: f


jaxtyping.jaxtyped = _dummy_jaxtyped
try:
  # pylint: disable=g-import-not-at-top
  from tokamax._src.ops.kda import api as kda_api
  # Registers the Mosaic TPU backend with `kda_api` (`api.py` imports it behind
  # a `try`, so without this dependency the call fails over to XLA).
  from tokamax._src.ops.kda import pallas_mosaic_tpu  # pylint: disable=unused-import
  # pylint: enable=g-import-not-at-top
except ImportError as e:
  raise ImportError(
      "TpuGatedDeltaKernel backend='tokamax' needs a Tokamax build with the "
      "Kimi Delta Attention op (tokamax._src.ops.kda): github.com/openxla/"
      "tokamax HEAD; the tokamax wheels up to 0.1.0 do not have it."
  ) from e
finally:
  jaxtyping.jaxtyped = _real_jaxtyped

# pylint: disable=g-import-not-at-top,g-bad-import-order
import torch
from torch_tpu._internal import pallas

# pylint: enable=g-import-not-at-top,g-bad-import-order

# Default lower bound on the per-token log decay. The Mosaic kernel factors the
# decay of each 16-token sub-block around the sub-block's midpoint (its
# "safe gate"), so intermediates grow like ``exp(-8 * g)``; Tokamax validates
# it for log decays in ``[-5, 0)`` (KDA's own ``lower_bound`` range). Faster
# decays are clamped: a decay of ``exp(-5)`` = 0.7% per token already discards
# the state within two tokens.
DEFAULT_MIN_LOG_DECAY = -5.0


def _heads_first(x: jax.Array) -> jax.Array:
  """``[B, T, H, ...]`` -> ``[H, B, T, ...]`` (Tokamax KDA's layout)."""
  return jnp.moveaxis(x, 2, 0)


def _gated_delta(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    min_log_decay: float,
) -> jax.Array:
  """Gated delta rule via Mosaic KDA; ``[B, T, H, D]`` in and out."""
  q_h, k_h, v_h = _heads_first(q), _heads_first(k), _heads_first(v)
  log_decay = jnp.maximum(_heads_first(g).astype(jnp.float32), min_log_decay)
  gate = jnp.broadcast_to(log_decay[..., None], q_h.shape)
  out, _ = kda_api.kimi_delta_attention(
      q_h,
      k_h,
      v_h,
      gate,
      _heads_first(beta).astype(q.dtype),
      use_qk_l2norm=True,
      implementation="mosaic",
  )
  return jnp.moveaxis(out, 0, 2)


def _gated_delta_vjp(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    min_log_decay: float,
):
  """Returns ``(out, vjp)``; ``vjp`` is a pytree of the backward's residuals."""
  return jax.vjp(
      functools.partial(_gated_delta, min_log_decay=min_log_decay),
      q,
      k,
      v,
      g,
      beta,
  )


def _input_spec(*tensors: torch.Tensor) -> str:
  """Encodes input shapes and dtypes as one static ``jax_op`` argument."""
  return ";".join(
      ",".join(str(d) for d in t.shape) + ":" + str(t.dtype).split(".")[-1]
      for t in tensors
  )


def _parse_input_spec(spec: str) -> list[jax.ShapeDtypeStruct]:
  """Inverse of `_input_spec`: the inputs' shapes and dtypes."""
  structs = []
  for item in spec.split(";"):
    shape, dtype = item.split(":")
    structs.append(
        jax.ShapeDtypeStruct(
            tuple(int(d) for d in shape.split(",") if d), jnp.dtype(dtype)
        )
    )
  return structs


@pallas.jax_op("tt_qwen35::gated_delta_fwd_res")
def _gated_delta_fwd_res(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    min_log_decay: float,
) -> list[jax.Array]:
  """Returns ``[out, *residuals]``."""
  out, vjp = _gated_delta_vjp(q, k, v, g, beta, min_log_decay)
  return [out, *jax.tree_util.tree_leaves(vjp)]


@pallas.jax_op("tt_qwen35::gated_delta_bwd_res")
def _gated_delta_bwd_res(
    residuals: Sequence[jax.Array],
    grad_out: jax.Array,
    input_spec: str,
    min_log_decay: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Runs the backward kernels on ``_gated_delta_fwd_res``'s residuals."""
  structs = _parse_input_spec(input_spec)
  # The structure (and static backward jaxpr) of the forward's VJP closure,
  # traced abstractly from the same input shapes; its leaves are `residuals`.
  _, vjp_struct = jax.eval_shape(
      functools.partial(_gated_delta_vjp, min_log_decay=min_log_decay),
      *structs,
  )
  vjp = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(vjp_struct), list(residuals)
  )
  return vjp(grad_out.astype(structs[0].dtype))


def _setup_context_res(ctx, inputs, output):
  *tensors, min_log_decay = inputs
  residuals = output[1:]
  ctx.mark_non_differentiable(*residuals)
  ctx.set_materialize_grads(False)
  ctx.save_for_backward(*residuals)
  ctx.input_spec = _input_spec(*tensors)
  ctx.min_log_decay = min_log_decay


def _backward_res(ctx, grad_outputs):
  grad_out = grad_outputs[0]
  if grad_out is None:
    return (None,) * 6
  grads = _gated_delta_bwd_res(
      list(ctx.saved_tensors), grad_out, ctx.input_spec, ctx.min_log_decay
  )
  return (*grads, None)


_gated_delta_fwd_res.register_autograd(
    _backward_res, setup_context=_setup_context_res
)


def tokamax_gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    min_log_decay: float = DEFAULT_MIN_LOG_DECAY,
) -> torch.Tensor:
  """Gated delta rule with Tokamax's Mosaic TPU KDA kernels.

  Args:
    q: ``(bs, seq_len, n_heads, key_head_dim)`` un-normalized queries.
    k: ``(bs, seq_len, n_heads, key_head_dim)`` un-normalized keys.
    v: ``(bs, seq_len, n_heads, value_head_dim)`` values.
    g: ``(bs, seq_len, n_heads)`` log-space decay (negative).
    beta: ``(bs, seq_len, n_heads)`` update gate in ``(0, 1)``.
    min_log_decay: ``g`` is clamped to at least this value (see
      ``DEFAULT_MIN_LOG_DECAY``).

  Returns:
    ``(bs, seq_len, n_heads, value_head_dim)`` output in ``q.dtype``. Applies
    Q/K L2 normalization and ``1 / sqrt(key_head_dim)`` query scaling, as
    ``gated_delta.chunked_gated_delta`` does. ``seq_len`` must be a multiple of
    64 (the kernel's chunk size).
  """
  return _gated_delta_fwd_res(q, k, v, g, beta, float(min_log_decay))[0]
