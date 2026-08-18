"""MegaBlox grouped matmul for dynamically routed TPU expert layers.

For expert ``e``, forward computes ``Y_e = X_e @ W_e``. Backward computes
``dX_e = dY_e @ W_e.T`` with another GMM and ``dW_e = X_e.T @ dY_e`` with
TGMM. dInput uses MegaBlox's logical rhs transpose so kernel-native weights do
not require a physical layout conversion. dWeight accumulates in FP32 across
token tiles.

The module-level custom operators are stable Dynamo/AOTAutograd boundaries.
Tiling arguments are compile-time constants, while ``group_sizes`` remains a
runtime int32 TPU tensor, so routing changes do not trigger specialization.
"""

import jax
import jax.numpy as jnp
import torch
from jax.experimental.pallas.ops.tpu.megablox.gmm import (
    gmm as megablox_gmm,
    tgmm as megablox_tgmm,
)
from torch_tpu._internal import pallas


def _tiling(
    tile_m: int,
    tile_k: int,
    tile_n: int,
    k_dim: int,
    n_dim: int,
) -> tuple[int, int, int]:
    """Resolve zero-valued K/N tiles to the corresponding full dimension."""
    return (
        tile_m,
        tile_k if tile_k > 0 else k_dim,
        tile_n if tile_n > 0 else n_dim,
    )


@pallas.jax_op("tt_experts::grouped_mm")
def _grouped_mm_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    tile_m: int,
    tile_k: int,
    tile_n: int,
    tgmm_tile_m: int,
    tgmm_tile_k: int,
    tgmm_tile_n: int,
) -> jax.Array:
    del tgmm_tile_m, tgmm_tile_k, tgmm_tile_n
    _, k_dim, n_dim = rhs.shape
    return megablox_gmm(
        lhs,
        rhs,
        group_sizes=group_sizes,
        preferred_element_type=lhs.dtype,
        tiling=_tiling(tile_m, tile_k, tile_n, k_dim, n_dim),
    )
@pallas.jax_op("tt_experts::grouped_mm_bwd")
def _grouped_mm_bwd(
    dout: jax.Array,
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    tile_m: int,
    tile_k: int,
    tile_n: int,
    tgmm_tile_m: int,
    tgmm_tile_k: int,
    tgmm_tile_n: int,
) -> tuple[jax.Array, jax.Array]:
    _, k_dim, n_dim = rhs.shape
    grad_lhs = megablox_gmm(
        dout,
        rhs,
        group_sizes=group_sizes,
        preferred_element_type=dout.dtype,
        tiling=_tiling(
            tile_m,
            tile_n,
            tile_k,
            n_dim,
            k_dim,
        ),
        transpose_rhs=True,
    )
    grad_rhs = megablox_tgmm(
        jnp.swapaxes(lhs, 0, 1),
        dout,
        group_sizes=group_sizes,
        preferred_element_type=jnp.float32,
        tiling=_tiling(
            tgmm_tile_m,
            tgmm_tile_k,
            tgmm_tile_n,
            lhs.shape[1],
            dout.shape[1],
        ),
        interpret=False,
    )
    return grad_lhs, grad_rhs


def _setup_context(ctx, inputs, output):
    del output
    lhs, rhs, group_sizes, *static_args = inputs
    ctx.save_for_backward(lhs, rhs, group_sizes)
    ctx.static_args = tuple(static_args)


def _backward(ctx, grad_output):
    lhs, rhs, group_sizes = ctx.saved_tensors
    grad_lhs, grad_rhs = _grouped_mm_bwd(
        grad_output, lhs, rhs, group_sizes, *ctx.static_args
    )
    return grad_lhs, grad_rhs, None, None, None, None, None, None, None


_grouped_mm_fwd.register_autograd(_backward, setup_context=_setup_context)


def segment_matmul_pallas_2d(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    tile_m: int = 1024,
    tile_k: int = 0,
    tile_n: int = 0,
    tgmm_tiling: tuple[int, int, int] = (128, 128, 128),
) -> torch.Tensor:
    """Compute per-expert matmuls without moving group sizes to the host.

    Zero-valued K/N tile dimensions select the corresponding full matrix
    dimension. Distinct tiling values produce distinct compiled specializations.
    """
    if group_sizes.dtype != torch.int32:
        group_sizes = group_sizes.to(torch.int32)
    return _grouped_mm_fwd(
        lhs,
        rhs,
        group_sizes,
        tile_m,
        tile_k,
        tile_n,
        *tgmm_tiling,
    )


__all__ = ["segment_matmul_pallas_2d"]
