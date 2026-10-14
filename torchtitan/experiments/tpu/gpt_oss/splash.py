"""Tokamax's native learned-sink attention through the TorchTPU bridge.

Unlike the generic Splash wrapper, this passes sinks into the softmax and
returns their gradients. Backward consumes saved output/LSE, without another
forward or a quadratic PyTorch score tensor. Kernel implementations are
Tokamax's; this module only adapts layouts, state and autograd.
"""

import jax
import jax.numpy as jnp
from torch_tpu._internal import pallas

# Reuse the repository's Tokamax import compatibility shim.
from torchtitan.experiments.tpu.kernels.tokamax_splash_attention import (
    _DEFAULT_MASK_VALUE,
    splash_attention_kernel as kernel,
    splash_attention_mask as masks,
)


def _kernel(length, kv_heads, window, block_size):
    block = min(length, block_size)
    config = kernel.SplashConfig(
        block_q=block,
        block_kv=block,
        block_kv_compute=block,
        block_q_dkv=block,
        block_kv_dkv=block,
        block_kv_dkv_compute=block,
        use_fused_bwd_kernel=True,
    )
    shape = (length, length)
    mask = (
        masks.CausalMask(shape)
        if window == -1
        else masks.LocalMask(shape, window_size=(window - 1, 0), offset=0)
    )
    make = (
        kernel.make_splash_mqa_single_device
        if kv_heads == 1
        else kernel.make_splash_mha_single_device
    )
    return make(mask=mask, config=config), config


def _layout(q, k, v, scale):
    q = (q.astype(jnp.float32) * scale).astype(q.dtype).transpose(0, 2, 1, 3)
    if k.shape[2] == 1:
        return q, k.squeeze(2), v.squeeze(2)
    return q, k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)


def _forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sinks: jax.Array,
    segments: jax.Array,
    window: int,
    scale: float,
    block_size: int,
) -> tuple[jax.Array, jax.Array]:
    op, config = _kernel(q.shape[1], k.shape[2], window, block_size)
    is_mqa = k.shape[2] == 1
    q, k, v = _layout(q, k, v, scale)

    def one(q, k, v, ids):
        out, stats = kernel._splash_attention_forward(
            op.fwd_mask_info,
            q,
            k,
            v,
            segment_ids=kernel.SegmentIds(q=ids, kv=ids),
            sinks=sinks,
            mask_value=op.kwargs.get("mask_value", _DEFAULT_MASK_VALUE),
            is_mqa=is_mqa,
            config=config,
            save_residuals=True,
            mask_function=op.kwargs.get("mask_function"),
            fwd_mask_sparsity=op.kwargs.get("fwd_mask_sparsity", 1.0),
        )
        return out, stats["logsumexp"]

    out, lse = jax.vmap(one)(q, k, v, segments)
    return out.transpose(0, 2, 1, 3), lse


def _backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sinks: jax.Array,
    segments: jax.Array,
    out: jax.Array,
    lse: jax.Array,
    grad: jax.Array,
    window: int,
    scale: float,
    block_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    op, config = _kernel(q.shape[1], k.shape[2], window, block_size)
    is_mqa = k.shape[2] == 1
    q, k, v = _layout(q, k, v, scale)

    def one(q, k, v, ids, out, lse, grad):
        residuals = (
            q,
            k,
            v,
            kernel.SegmentIds(q=ids, kv=ids),
            sinks,
            out,
            lse,
            op.dkv_mask_info,
            None,
        )
        grads = kernel._splash_attention_bwd(
            False,
            op.kwargs.get("mask_value", _DEFAULT_MASK_VALUE),
            is_mqa,
            config,
            op.kwargs.get("mask_function"),
            op.kwargs.get("fwd_mask_sparsity", 1.0),
            op.kwargs.get("dkv_mask_sparsity", 1.0),
            residuals,
            grad,
        )
        return grads[2], grads[3], grads[4], grads[6]

    dq, dk, dv, dsinks = jax.vmap(one)(
        q,
        k,
        v,
        segments,
        out.transpose(0, 2, 1, 3),
        lse,
        grad.transpose(0, 2, 1, 3),
    )
    dq = (dq.astype(jnp.float32) * scale).astype(q.dtype).transpose(0, 2, 1, 3)
    if is_mqa:
        dk, dv = jnp.expand_dims(dk, 2), jnp.expand_dims(dv, 2)
    else:
        dk, dv = dk.transpose(0, 2, 1, 3), dv.transpose(0, 2, 1, 3)
    return dq, dk, dv, dsinks.astype(jnp.float32).sum(0).astype(sinks.dtype)


sink_splash = pallas.jax_op("pallas::gpt_oss_sink_splash_fwd", _forward)
sink_splash_backward = pallas.jax_op("pallas::gpt_oss_sink_splash_bwd", _backward)


def _setup_context(ctx, inputs, output):
    q, k, v, sinks, segments, *config = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, sinks, segments, out, lse)
    ctx.config = config
    # LSE is a backward residual; sinks are differentiated inside Tokamax.
    ctx.mark_non_differentiable(lse)


def _autograd_backward(ctx, grad, grad_lse):
    del grad_lse
    grads = sink_splash_backward(*ctx.saved_tensors, grad, *ctx.config)
    return (*grads, None, None, None, None)


sink_splash.register_autograd(_autograd_backward, setup_context=_setup_context)
