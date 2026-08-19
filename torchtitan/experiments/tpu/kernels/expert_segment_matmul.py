"""MegaBlox grouped matmul for dynamically routed TPU expert layers.

For expert ``e``, forward computes ``Y_e = X_e @ W_e``. Backward has two
different contractions: dInput is another row-grouped matmul,
``dX_e = dY_e @ W_e.T``, while dWeight is a transposed grouped matmul (TGMM),
``dW_e = X_e.T @ dY_e``. Both primitives come from MegaBlox.

``group_sizes`` remains an int32 TPU tensor at runtime, so routing changes do
not require a device-to-host copy or a separately specialized kernel.
"""

from functools import lru_cache
from typing import Tuple, cast

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox import ops as gmm_ops
import torch
from torch_tpu._internal import pallas


@lru_cache(maxsize=None)
def _build_grouped_matmul_kernels(tile_m: int):
  """Build GMM forward/dInput and TGMM dWeight with runtime group sizes."""
  with jax.default_device(jax.local_devices()[0]):

    @pallas.jax_op(f"tt_experts::grouped_mm_fwd_tm{tile_m}")
    def fwd_op(
        lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array
    ) -> jax.Array:
      return gmm_ops.gmm(
          lhs,
          rhs,
          group_sizes=group_sizes,
          preferred_element_type=jnp.bfloat16,
          tiling=(tile_m, rhs.shape[1], rhs.shape[2]),
      )

    @pallas.jax_op(f"tt_experts::grouped_mm_bwd_tm{tile_m}")
    def bwd_op(
        dout: jax.Array,
        lhs: jax.Array,
        rhs: jax.Array,
        group_sizes: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
      # dInput preserves the packed token rows, so it is another GMM with the
      # per-expert weight matrices transposed.
      grad_lhs = gmm_ops.gmm(
          dout,
          jnp.swapaxes(rhs, 1, 2),
          group_sizes=group_sizes,
          preferred_element_type=jnp.bfloat16,
          tiling=(tile_m, rhs.shape[2], rhs.shape[1]),
      )
      # dWeight reduces each expert's variable-size token group into one
      # matrix, which is the contraction implemented by MegaBlox TGMM.
      grad_rhs = gmm_ops.backend.tgmm(
          jnp.swapaxes(lhs, 0, 1),
          dout,
          group_sizes=group_sizes,
          preferred_element_type=jnp.float32,
          tiling=(128, 128, 128),
          interpret=False,
      )
      return grad_lhs, grad_rhs

  return fwd_op, bwd_op


class _GroupedMatmulAutograd(torch.autograd.Function):
  @staticmethod
  def forward(ctx, lhs, rhs, group_sizes, tile_m):
    ctx.save_for_backward(lhs, rhs, group_sizes)
    ctx.tile_m = tile_m
    fwd_op, _ = _build_grouped_matmul_kernels(tile_m)
    return cast(torch.Tensor, fwd_op(lhs, rhs, group_sizes))

  @staticmethod
  def backward(ctx, grad_output):  # pyrefly: ignore[bad-override]
    lhs, rhs, group_sizes = ctx.saved_tensors
    _, bwd_op = _build_grouped_matmul_kernels(ctx.tile_m)
    grad_lhs, grad_rhs = cast(
        Tuple[torch.Tensor, torch.Tensor],
        bwd_op(grad_output, lhs, rhs, group_sizes),
    )
    return grad_lhs, grad_rhs, None, None


def segment_matmul_pallas_2d(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    tile_m: int = 1024,
) -> torch.Tensor:
  """Compute per-expert matmuls without moving group sizes to the host."""
  if group_sizes.dtype != torch.int32:
    group_sizes = group_sizes.to(torch.int32)
  return _GroupedMatmulAutograd.apply(lhs, rhs, group_sizes, tile_m)
