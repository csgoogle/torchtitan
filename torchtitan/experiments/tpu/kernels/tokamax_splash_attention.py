"""Splash Attention integration for torch_tpu via Pallas and Tokamax.

Wraps Tokamax's splash_attention_kernel for use with PyTorch tensors on TPU.
Uses torch_tpu._internal.pallas.pallas.jax_op to bridge JAX -> PyTorch.
"""

import inspect
import math
import types
from typing import Optional

# JAX imports
import jax
import jax.numpy as jnp
import numpy as np

# Workaround for https://github.com/openxla/tokamax/issues/240 (same as
# linear_softmax_cross_entropy_loss.py): tokamax applies @jaxtyping.jaxtyped
# at import time, and typeguard's AST instrumentation cannot parse jaxtyping
# shape-string annotations ("*B T H d"). Swap in a pass-through decorator for
# the duration of the tokamax import.
import jaxtyping

_real_jaxtyped = jaxtyping.jaxtyped


def _dummy_jaxtyped(*args, **kwargs):
  def decorator(f):
    return f

  if len(args) == 1 and callable(args[0]):
    return args[0]
  return decorator


jaxtyping.jaxtyped = _dummy_jaxtyped
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel  # noqa: E402  pylint: disable=g-import-not-at-top
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask  # noqa: E402  pylint: disable=g-import-not-at-top


# With ``fuse_reciprocal`` (the default) the forward kernel still writes a
# ``max_logits`` residual - an f32 [heads, seq, 128] lane-broadcast tensor as
# large as ``logsumexp`` - that neither this wrapper nor the backward reads.
# Re-execute the kernel module with that output dropped. Each edit must match
# exactly once so a tokamax upgrade fails loudly instead of silently.
_MAX_LOGITS_EDITS = (
    (
        """        # max_logits
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32),
    ]""",
        """        # max_logits
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32)
        if not fuse_reciprocal
        else None,
    ]""",
    ),
    (
        """        pl.BlockSpec(
            (num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map
        ),
    ]
  else:
    out_shapes += [None, None, None]""",
        """        pl.BlockSpec(
            (num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map
        )
        if not fuse_reciprocal
        else None,
    ]
  else:
    out_shapes += [None, None, None]""",
    ),
    (
        """    assert max_logits is not None
    max_logits = init_if_empty(max_logits[..., 0], mask_value)
""",
        """    if max_logits is not None:
      max_logits = init_if_empty(max_logits[..., 0], mask_value)
""",
    ),
)


# The ``logsumexp`` residual is also lane-broadcast: every one of its 128 lanes
# holds the row's value, and the wrapper keeps only lane 0. Store it transposed
# instead, as [heads, 8, seq] with the sequence along lanes (1/16 of the
# bytes). The values are unchanged; the backward reads the same [heads, seq].
_COMPACT_LOGSUMEXP_EDITS = (
    (
        """        # logsumexp
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32)
        if fuse_reciprocal""",
        """        # logsumexp
        jax.ShapeDtypeStruct((num_q_heads, NUM_SUBLANES, q_seq_len), jnp.float32)
        if fuse_reciprocal""",
    ),
    (
        """    out_specs += [
        pl.BlockSpec(
            (num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map
        )
        if fuse_reciprocal""",
        """    out_specs += [
        pl.BlockSpec(
            (num_stacked_q_heads, NUM_SUBLANES, bq),
            unravel(lambda h_block, i, j, *_: (h_block, 0, i)),
        )
        if fuse_reciprocal""",
    ),
    (
        """      assert logsumexp_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
      log = jnp.log2 if config.use_base2_exp else jnp.log
      logsumexp = m + log(l)
      logsumexp_ref[...] = logsumexp.astype(logsumexp_ref.dtype)""",
        """      assert logsumexp_ref.shape == (num_stacked_q_heads, NUM_SUBLANES, bq)
      log = jnp.log2 if config.use_base2_exp else jnp.log
      logsumexp = m + log(l)
      logsumexp = jnp.swapaxes(logsumexp, -1, -2)[:, :NUM_SUBLANES, :]
      logsumexp_ref[...] = logsumexp.astype(logsumexp_ref.dtype)""",
    ),
    (
        """      logsumexp = init_if_empty(logsumexp[..., 0], mask_value)""",
        """      logsumexp = init_if_empty(logsumexp[..., 0, :], mask_value)""",
    ),
)


def _patched_splash_kernel(module):
  source = inspect.getsource(module)
  for old, new in _MAX_LOGITS_EDITS + _COMPACT_LOGSUMEXP_EDITS:
    if source.count(old) != 1:
      raise RuntimeError(
          f"tokamax splash kernel changed; edit anchor found {source.count(old)} times"
      )
    source = source.replace(old, new)
  old = 'stats["max_logits"] = stats["max_logits"] / LOG2E'
  if source.count(old) != 2:
    raise RuntimeError("tokamax splash kernel changed; max_logits rescale not found")
  source = source.replace(
      old,
      'stats["max_logits"] = None if stats["max_logits"] is None '
      'else stats["max_logits"] / LOG2E',
  )
  patched = types.ModuleType(module.__name__)
  patched.__dict__.update(
      __file__=module.__file__, __package__=module.__package__
  )
  exec(compile(source, module.__file__, "exec"), patched.__dict__)
  return patched


splash_attention_kernel = _patched_splash_kernel(splash_attention_kernel)

jaxtyping.jaxtyped = _real_jaxtyped

import torch  # noqa: E402  pylint: disable=g-import-not-at-top,g-bad-import-order
# torch_tpu Pallas bridge
from torch_tpu._internal import pallas


# Taken from tokamax._src.ops.experimental.tpu.splash_attention.base.DEFAULT_MASK_VALUE.
_DEFAULT_MASK_VALUE = -0.7 * float(
    np.finfo(np.dtype("float32")).max
)


class TokamaxConfig:
  """Global configuration container for hardware-static hyperparameters."""
  block_q: int = 512
  block_kv: int = 512
  block_kv_compute: int = 512
  block_q_dkv: int = 512
  block_kv_dkv: int = 512
  block_kv_dkv_compute: int = 512
  block_q_dq: Optional[int] = None
  block_kv_dq: Optional[int] = None
  use_fused_bwd_kernel: bool = True
  use_vmap_bwd: bool = False
  q_layout: str = "HEAD_DIM_MINOR"
  k_layout: str = "HEAD_DIM_MINOR"
  v_layout: str = "HEAD_DIM_MINOR"

  @classmethod
  def update(cls, **kwargs):
    for k, v in kwargs.items():
      if v is not None:
        setattr(cls, k, v)


def _make_splash_attention_fn(
    seq_len: int,
    n_heads: int,
    n_kv_heads: int,
    is_causal: bool = True,
    local_window_size: int | None = None,
    block_q: int = 512,
    block_kv: int = 512,
    block_kv_compute: int = 512,
    block_q_dkv: int = 512,
    block_kv_dkv: int = 512,
    block_kv_dkv_compute: int = 512,
    block_q_dq: int | None = None,
    block_kv_dq: int | None = None,
    use_fused_bwd_kernel: bool = True,
    q_layout: str = "HEAD_DIM_MINOR",
    k_layout: str = "HEAD_DIM_MINOR",
    v_layout: str = "HEAD_DIM_MINOR",
    use_vmap_bwd: bool = False,
):
  """Create JAX splash attention forward and backward functions using Tokamax.

  Returns:
    (splash_fn, splash_bwd_fn): forward function taking (q, k, v) returning
      (out, logsumexp), and backward function taking (q, k, v, out, logsumexp,
      grad_out) returning (grad_q, grad_k, grad_v) WITHOUT re-running the
      forward.
  """
  del n_heads


  def _get_layout(layout_str):
    if layout_str == "HEAD_DIM_MINOR":
      return splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR
    elif layout_str == "SEQ_MINOR":
      return splash_attention_kernel.QKVLayout.SEQ_MINOR
    else:
      raise ValueError(f"Unknown layout: {layout_str}")


  config = splash_attention_kernel.SplashConfig(
      block_q=min(block_q, seq_len),
      block_kv=min(block_kv, seq_len),
      block_kv_compute=min(block_kv_compute, seq_len),
      block_q_dkv=min(block_q_dkv, seq_len),
      block_kv_dkv=min(block_kv_dkv, seq_len),
      block_kv_dkv_compute=min(block_kv_dkv_compute, seq_len),
      block_q_dq=min(block_q_dq, seq_len)
      if not use_fused_bwd_kernel and block_q_dq is not None
      else None,
      block_kv_dq=min(block_kv_dq, seq_len)
      if not use_fused_bwd_kernel and block_kv_dq is not None
      else None,
      use_fused_bwd_kernel=use_fused_bwd_kernel,
      q_layout=_get_layout(q_layout),
      k_layout=_get_layout(k_layout),
      v_layout=_get_layout(v_layout),
  )

  mask_shape = (seq_len, seq_len)
  if local_window_size is not None:
    single_mask = splash_attention_mask.LocalMask(
        shape=mask_shape,
        window_size=(local_window_size, 0),
        offset=0,
    )
  elif is_causal:
    single_mask = splash_attention_mask.CausalMask(shape=mask_shape)
  else:
    single_mask = splash_attention_mask.FullMask(mask_shape)

  is_mqa = n_kv_heads == 1

  if is_mqa:
    splash_kernel = splash_attention_kernel.make_splash_mqa_single_device(
        mask=single_mask,
        config=config,
    )
  else:
    splash_kernel = splash_attention_kernel.make_splash_mha_single_device(
        mask=single_mask,
        config=config,
    )

  # Static mask infos and kernel kwargs from creation time.
  # TODO: once segment_ids is wired in, also apply _document_aware_mask_info to
  # fwd_mask_info/dkv_mask_info so entirely cross-document blocks are skipped.
  fwd_mask_info = splash_kernel.fwd_mask_info
  dkv_mask_info = splash_kernel.dkv_mask_info
  _mask_value = splash_kernel.kwargs.get("mask_value", _DEFAULT_MASK_VALUE)
  _mask_function = splash_kernel.kwargs.get("mask_function", None)
  _fwd_mask_sparsity = splash_kernel.kwargs.get("fwd_mask_sparsity", 1.0)
  _dkv_mask_sparsity = splash_kernel.kwargs.get("dkv_mask_sparsity", 1.0)

  # TODO: wire segment_ids through for document packing
  def _single_fwd_with_lse(q_b, k_b, v_b):
    """Forward for one batch element; returns (out_b, logsumexp_b)."""
    # q_b is already scaled (q * 1/sqrt(head_dim)) from the caller.
    # pylint: disable=protected-access
    out_b, stats_b = splash_attention_kernel._splash_attention_forward(
        fwd_mask_info,
        q_b,
        k_b,
        v_b,
        segment_ids=None,
        sinks=None,
        mask_value=_mask_value,
        is_mqa=is_mqa,
        config=config,
        save_residuals=True,
        mask_function=_mask_function,
        fwd_mask_sparsity=_fwd_mask_sparsity,
    )
    # pylint: enable=protected-access
    return out_b, stats_b["logsumexp"]


  @jax.jit
  def splash_fn(q, k, v):
    """Forward: returns (out, logsumexp) for the whole batch."""
    head_dim = q.shape[-1]
    scale = jnp.array(1.0 / math.sqrt(head_dim), dtype=q.dtype)
    q_scaled = q * scale
    out, logsumexp = jax.vmap(_single_fwd_with_lse)(q_scaled, k, v)
    return out, logsumexp

  def _single_bwd(q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b):
    """Backward for one batch element using precomputed residuals.

    Calls _splash_attention_bwd directly — no extra forward pass.
    """
    res = (
        q_scaled_b,
        k_b,
        v_b,
        None,  # segment_ids
        None,  # sinks
        out_b,
        logsumexp_b,
        dkv_mask_info,
        None,  # prng_key (added in upstream tokamax)
    )
    # pylint: disable=protected-access
    all_grads = splash_attention_kernel._splash_attention_bwd(
        False,  # save_residuals (nondiff)
        _mask_value,  # mask_value (nondiff)
        is_mqa,  # is_mqa (nondiff)
        config,  # config (nondiff)
        _mask_function,  # mask_function (nondiff)
        _fwd_mask_sparsity,  # fwd_mask_sparsity (nondiff)
        _dkv_mask_sparsity,  # dkv_mask_sparsity (nondiff)
        res,  # residuals
        g_b,  # do (output gradient)
    )
    # pylint: enable=protected-access
    return all_grads[2], all_grads[3], all_grads[4]  # dq_scaled, dk, dv


  def _scan_bwd_body(carry, args):
    """Single-element backward body for jax.lax.scan (sequential over batch)."""
    q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b = args
    dq_scaled_b, dk_b, dv_b = _single_bwd(
        q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b
    )
    return carry, (dq_scaled_b, dk_b, dv_b)

  @jax.jit
  def splash_bwd_fn(q, k, v, out, logsumexp, g):
    """Backward: uses saved (out, logsumexp) — no extra forward pass."""
    head_dim = q.shape[-1]
    scale = jnp.array(1.0 / math.sqrt(head_dim), dtype=q.dtype)
    q_scaled = q * scale

    if use_vmap_bwd:
      dq_scaled, dk, dv = jax.vmap(_single_bwd)(
          q_scaled, k, v, out, logsumexp, g
      )
    else:
      _, (dq_scaled, dk, dv) = jax.lax.scan(
          _scan_bwd_body,
          None,
          (q_scaled, k, v, out, logsumexp, g),
      )
    # Chain rule: q_scaled = q * scale  =>  dq = dq_scaled * scale
    dq = dq_scaled * scale
    return dq, dk, dv

  return splash_fn, splash_bwd_fn


def jax_attention_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    is_causal: bool,
    local_window_size: int,  # Sentinel int: -1 means None
    scale: float,
) -> tuple[jax.Array, jax.Array]:
  """Forward attention in JAX using Pallas.

  Accepts raw PyTorch layouts: (batch, seq_len, heads, head_dim)
  """
  seq_len = q.shape[1]
  n_heads = q.shape[2]
  n_kv_heads = k.shape[2]

  local_window_size_opt = local_window_size if local_window_size != -1 else None

  # 1. Transpose/Squeeze inside JAX so JIT fuses operations
  q_tr = jnp.transpose(q, (0, 2, 1, 3))

  if n_kv_heads == 1:
    k_tr = jnp.squeeze(k, axis=2)
    v_tr = jnp.squeeze(v, axis=2)
  else:
    k_tr = jnp.transpose(k, (0, 2, 1, 3))
    v_tr = jnp.transpose(v, (0, 2, 1, 3))

  splash_fn, _ = _make_splash_attention_fn(
      seq_len=seq_len,
      n_heads=n_heads,
      n_kv_heads=n_kv_heads,
      is_causal=is_causal,
      local_window_size=local_window_size_opt,
      block_q=TokamaxConfig.block_q,
      block_kv=TokamaxConfig.block_kv,
      block_kv_compute=TokamaxConfig.block_kv_compute,
      block_q_dkv=TokamaxConfig.block_q_dkv,
      block_kv_dkv=TokamaxConfig.block_kv_dkv,
      block_kv_dkv_compute=TokamaxConfig.block_kv_dkv_compute,
      block_q_dq=TokamaxConfig.block_q_dq,
      block_kv_dq=TokamaxConfig.block_kv_dq,
      use_fused_bwd_kernel=TokamaxConfig.use_fused_bwd_kernel,
      use_vmap_bwd=TokamaxConfig.use_vmap_bwd,
      q_layout=TokamaxConfig.q_layout,
      k_layout=TokamaxConfig.k_layout,
      v_layout=TokamaxConfig.v_layout,
  )

  # Apply scaling inside the compiled JAX pipeline
  default_scale = 1.0 / math.sqrt(q.shape[-1])
  if abs(scale - default_scale) > 1e-6:
    q_tr = q_tr * (scale / default_scale)

  out_tr, logsumexp = splash_fn(q_tr, k_tr, v_tr)

  # Transpose back: (B, H, S, d) -> (B, S, H, d)
  out = jnp.transpose(out_tr, (0, 2, 1, 3))
  return out, logsumexp


def jax_attention_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    logsumexp: jax.Array,
    grad_out: jax.Array,
    is_causal: bool,
    local_window_size: int,  # Sentinel int: -1 means None
    scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Backward attention in JAX calling direct Pallas kernel backward."""
  seq_len = q.shape[1]
  n_heads = q.shape[2]
  n_kv_heads = k.shape[2]

  local_window_size_opt = local_window_size if local_window_size != -1 else None

  # 1. Transpose/Squeeze inputs inside JAX
  q_tr = jnp.transpose(q, (0, 2, 1, 3))
  out_tr = jnp.transpose(out, (0, 2, 1, 3))
  grad_out_tr = jnp.transpose(grad_out, (0, 2, 1, 3))

  if n_kv_heads == 1:
    k_tr = jnp.squeeze(k, axis=2)
    v_tr = jnp.squeeze(v, axis=2)
  else:
    k_tr = jnp.transpose(k, (0, 2, 1, 3))
    v_tr = jnp.transpose(v, (0, 2, 1, 3))

  _, splash_bwd_fn = _make_splash_attention_fn(
      seq_len=seq_len,
      n_heads=n_heads,
      n_kv_heads=n_kv_heads,
      is_causal=is_causal,
      local_window_size=local_window_size_opt,
      block_q=TokamaxConfig.block_q,
      block_kv=TokamaxConfig.block_kv,
      block_kv_compute=TokamaxConfig.block_kv_compute,
      block_q_dkv=TokamaxConfig.block_q_dkv,
      block_kv_dkv=TokamaxConfig.block_kv_dkv,
      block_kv_dkv_compute=TokamaxConfig.block_kv_dkv_compute,
      block_q_dq=TokamaxConfig.block_q_dq,
      block_kv_dq=TokamaxConfig.block_kv_dq,
      use_fused_bwd_kernel=TokamaxConfig.use_fused_bwd_kernel,
      use_vmap_bwd=TokamaxConfig.use_vmap_bwd,
      q_layout=TokamaxConfig.q_layout,
      k_layout=TokamaxConfig.k_layout,
      v_layout=TokamaxConfig.v_layout,
  )

  default_scale = 1.0 / math.sqrt(q.shape[-1])
  if abs(scale - default_scale) > 1e-6:
    q_tr = q_tr * (scale / default_scale)

  dq_tr, dk_tr, dv_tr = splash_bwd_fn(q_tr, k_tr, v_tr, out_tr, logsumexp, grad_out_tr)

  # Chain rule for the q pre-scaling above: splash_bwd_fn returns the
  # gradient wrt the SCALED q, so a non-default scale needs the same
  # ratio applied to dq (without it, dq is off by default_scale/scale).
  if abs(scale - default_scale) > 1e-6:
    dq_tr = dq_tr * (scale / default_scale)

  # Transpose/Expand back to original PyTorch layouts
  dq = jnp.transpose(dq_tr, (0, 2, 1, 3))

  if n_kv_heads == 1:
    dk = jnp.expand_dims(dk_tr, axis=2)
    dv = jnp.expand_dims(dv_tr, axis=2)
  else:
    dk = jnp.transpose(dk_tr, (0, 2, 1, 3))
    dv = jnp.transpose(dv_tr, (0, 2, 1, 3))

  return dq, dk, dv


# Register standard top-level global Custom Operators exactly like call_kernel.py!
splash_op: torch._library.custom_ops.CustomOpDef = pallas.jax_op(
    "pallas::tokamax_splash_attn_fwd", jax_attention_fwd
)

splash_op_backward: torch._library.custom_ops.CustomOpDef = pallas.jax_op(
    "pallas::tokamax_splash_attn_bwd", jax_attention_bwd
)


def setup_context(ctx, inputs, output, **kwargs):
  del kwargs
  # inputs contains exactly the 6 positional dynamic custom op parameters:
  # q, k, v, is_causal, local_window_size, scale
  q, k, v, is_causal, local_window_size, scale = inputs
  out, logsumexp = output
  ctx.save_for_backward(q, k, v, out, logsumexp)

  ctx.is_causal = is_causal
  ctx.local_window_size = local_window_size
  ctx.scale = scale


def backward(ctx, grad_out, grad_lse):
  del grad_lse  # Unused
  q, k, v, out, logsumexp = ctx.saved_tensors

  # Call our clean JAX backward global top-level custom operator!
  dq, dk, dv = splash_op_backward(
      q,
      k,
      v,
      out,
      logsumexp,
      grad_out,
      ctx.is_causal,
      ctx.local_window_size,
      ctx.scale,
  )
  # Returns clean, standard 6-element gradient unpack block!
  return dq, dk, dv, None, None, None


# Register the autograd!
splash_op.register_autograd(backward, setup_context=setup_context)


def splash_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    enable_gqa: bool = False,
    block_q: Optional[int] = None,
    block_kv: Optional[int] = None,
    block_dkv: Optional[int] = None,
    block_kv_compute: Optional[int] = None,
    block_q_dkv: Optional[int] = None,
    block_kv_dkv: Optional[int] = None,
    block_kv_dkv_compute: Optional[int] = None,
    block_q_dq: Optional[int] = None,
    block_kv_dq: Optional[int] = None,
    use_fused_bwd_kernel: Optional[bool] = None,
    use_vmap_bwd: Optional[bool] = None,
    q_layout: Optional[str] = None,
    k_layout: Optional[str] = None,
    v_layout: Optional[str] = None,
) -> torch.Tensor:
  """Replacement for F.scaled_dot_product_attention using Tokamax splash attention."""
  # These are accepted for signature compatibility with the JAX splash wrapper
  # but are NOT the channel that reaches the kernel: splash_op reads the
  # process-global TokamaxConfig, which the patch populates once at setup.
  # Passing a value here that disagrees with TokamaxConfig would silently do
  # nothing, so refuse it rather than discard it -- a caller who tiles through
  # this path is wrong about how the kernel is configured.
  _passed = {
      "block_q": block_q, "block_kv": block_kv, "block_dkv": block_dkv,
      "block_kv_compute": block_kv_compute, "block_q_dkv": block_q_dkv,
      "block_kv_dkv": block_kv_dkv, "block_kv_dkv_compute": block_kv_dkv_compute,
      "block_q_dq": block_q_dq, "block_kv_dq": block_kv_dq,
      "use_fused_bwd_kernel": use_fused_bwd_kernel, "use_vmap_bwd": use_vmap_bwd,
      "q_layout": q_layout, "k_layout": k_layout, "v_layout": v_layout,
  }
  _conflicts = {
      k: (v, getattr(TokamaxConfig, k))
      for k, v in _passed.items()
      if v is not None and hasattr(TokamaxConfig, k) and v != getattr(TokamaxConfig, k)
  }
  if _conflicts:
    raise ValueError(
        "splash_sdpa received per-call tiling that disagrees with the active"
        " TokamaxConfig; this path cannot honor per-call tiles, so the values"
        " would be silently ignored. Set them on the SplashAttentionKernelConfig"
        f" instead. Conflicts (passed, active): {_conflicts}"
    )
  # Validate enable_gqa rather than ignore it. The stock splash_sdpa in
  # splash_attention.py enforces this same invariant, and the two functions are
  # meant to be interchangeable via a config flag -- so silently accepting a
  # wrong value here would make the two kernels disagree on identical input:
  # stock raises, tokamax runs on. That check catches a real mistake (passing
  # enable_gqa=False for a GQA shape); dropping it when the flag is flipped
  # would lose the assertion precisely when the kernel changes underneath you.
  n_heads, n_kv_heads = q.shape[2], k.shape[2]
  if enable_gqa != (n_heads != n_kv_heads):
    raise ValueError(
        f"enable_gqa ({enable_gqa}) does not match inputs: "
        f"n_heads={n_heads}, n_kv_heads={n_kv_heads}"
    )

  head_dim = q.shape[-1]



  if scale is None:
    scale = 1.0 / math.sqrt(head_dim)

  local_window_size_val = local_window_size if local_window_size is not None else -1

  # Call global custom operator passing exactly the 6 positional parameters!
  out, _ = splash_op(
      q,
      k,
      v,
      is_causal,
      local_window_size_val,
      scale,
  )
  return out
