"""Paired MegaBlox grouped matmul over packed equal-shaped projections.

The two weights are stored as ``(E, 2, K, N)`` so FSDP can communicate them as
one parameter. The block specs load full K and N dimensions. This implementation
targets the Qwen expert shape rather than serving as a general grouped-matmul
kernel.
"""

from functools import lru_cache, partial
from typing import Tuple, cast

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.megablox import ops as gmm_ops
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas


_make_group_metadata = gmm_ops.backend.make_group_metadata
_get_group_size = gmm_ops.backend._get_group_size
_get_store_mask = gmm_ops.backend._get_store_mask


def _validate_shapes(lhs, rhs13, group_sizes, tile_m):
  if tile_m <= 0:
    raise ValueError("tile_m must be positive")
  if lhs.ndim != 2 or rhs13.ndim != 4:
    raise ValueError("Paired grouped matmul expects 2D lhs and 4D weights")
  if group_sizes.ndim != 1 or group_sizes.dtype != jnp.int32:
    raise ValueError("group_sizes must be a 1D int32 tensor")
  if rhs13.shape[1] != 2:
    raise ValueError("Packed paired weights must have two projections")
  if rhs13.shape[0] == 0:
    raise ValueError("Paired grouped matmul requires at least one expert")
  if lhs.shape[1] != rhs13.shape[2]:
    raise ValueError("Paired grouped matmul contraction dimensions must match")
  if rhs13.shape[0] != group_sizes.shape[0]:
    raise ValueError("group_sizes must have one entry per expert")
  if lhs.shape[0] % tile_m:
    raise ValueError("The total row count must be divisible by tile_m")
  if lhs.dtype != jnp.bfloat16 or rhs13.dtype != jnp.bfloat16:
    raise ValueError("Paired grouped matmul currently supports bfloat16 inputs")


def _metadata(group_sizes, *, m, tile_m, groups, visit_empty_groups):
  return _make_group_metadata(
      group_sizes=group_sizes,
      m=m,
      tm=tile_m,
      start_group=jnp.array(0, dtype=jnp.int32),
      num_nonzero_groups=groups,
      visit_empty_groups=visit_empty_groups,
  )


@partial(jax.jit, static_argnames=("tile_m",))
def _paired_forward(lhs, rhs13, group_sizes, *, tile_m):
  _validate_shapes(lhs, rhs13, group_sizes, tile_m)
  m, k = lhs.shape
  groups, _, _, n = rhs13.shape
  group_metadata, num_active_tiles = _metadata(
      group_sizes,
      m=m,
      tile_m=tile_m,
      groups=groups,
      visit_empty_groups=False,
  )
  group_offset = jnp.array([0], dtype=jnp.int32)

  def kernel(
      group_metadata,
      group_offset,
      lhs,
      rhs13,
      out1,
      out3,
      acc1,
      acc3,
  ):
    del group_offset
    grid_id = pl.program_id(1)

    acc1[...] = jnp.zeros_like(acc1)
    acc3[...] = jnp.zeros_like(acc3)
    loaded_lhs = lhs[...]
    acc1[...] += lax.dot(
        loaded_lhs,
        rhs13[0, ...],
        preferred_element_type=jnp.float32,
    )
    acc3[...] += lax.dot(
        loaded_lhs,
        rhs13[1, ...],
        preferred_element_type=jnp.float32,
    )

    mask = _get_store_mask(
        grid_id=grid_id,
        group_metadata=group_metadata,
        tm=tile_m,
        tn=n,
    )
    out1[...] = lax.select(
        mask[...], acc1[...], out1[...].astype(jnp.float32)
    ).astype(jnp.bfloat16)
    out3[...] = lax.select(
        mask[...], acc3[...], out3[...].astype(jnp.float32)
    ).astype(jnp.bfloat16)

  def lhs_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[2][grid_id], 0

  def rhs_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[1][grid_id], 0, 0, 0

  def out_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[2][grid_id], 0

  lhs_spec = pl.BlockSpec((tile_m, k), lhs_indices)
  rhs_spec = pl.BlockSpec((None, 2, k, n), rhs_indices)
  out_spec = pl.BlockSpec((tile_m, n), out_indices)
  return pl.pallas_call(
      kernel,
      out_shape=[
          jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
          jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[lhs_spec, rhs_spec],
          out_specs=[out_spec, out_spec],
          grid=(1, num_active_tiles, 1),
          scratch_shapes=[
              pltpu.VMEM((tile_m, n), jnp.float32),
              pltpu.VMEM((tile_m, n), jnp.float32),
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary")
      ),
  )(group_metadata, group_offset, lhs, rhs13)


@partial(jax.jit, static_argnames=("tile_m",))
def _paired_dinput(dout1, dout3, rhs13, group_sizes, *, tile_m):
  if dout1.shape != dout3.shape or rhs13.ndim != 4 or rhs13.shape[1] != 2:
    raise ValueError("Paired dInput operands must have compatible shapes")
  m, n = dout1.shape
  groups, _, k, rhs_n = rhs13.shape
  if n != rhs_n or m % tile_m:
    raise ValueError("Invalid paired dInput dimensions")
  group_metadata, num_active_tiles = _metadata(
      group_sizes,
      m=m,
      tile_m=tile_m,
      groups=groups,
      visit_empty_groups=False,
  )
  group_offset = jnp.array([0], dtype=jnp.int32)

  def kernel(
      group_metadata,
      group_offset,
      dout1,
      dout3,
      rhs13,
      out,
      acc,
  ):
    del group_offset
    grid_id = pl.program_id(1)
    acc[...] = jnp.zeros_like(acc)
    acc[...] += lax.dot_general(
        dout1[...],
        rhs13[0, ...],
        preferred_element_type=jnp.float32,
        dimension_numbers=(((1,), (1,)), ((), ())),
    )
    acc[...] += lax.dot_general(
        dout3[...],
        rhs13[1, ...],
        preferred_element_type=jnp.float32,
        dimension_numbers=(((1,), (1,)), ((), ())),
    )

    mask = _get_store_mask(
        grid_id=grid_id,
        group_metadata=group_metadata,
        tm=tile_m,
        tn=k,
    )
    out[...] = lax.select(
        mask[...], acc[...], out[...].astype(jnp.float32)
    ).astype(jnp.bfloat16)

  def dout_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[2][grid_id], 0

  def rhs_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[1][grid_id], 0, 0, 0

  def out_indices(n_i, grid_id, k_i, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[2][grid_id], 0

  dout_spec = pl.BlockSpec((tile_m, n), dout_indices)
  rhs_spec = pl.BlockSpec((None, 2, k, n), rhs_indices)
  out_spec = pl.BlockSpec((tile_m, k), out_indices)
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, k), jnp.bfloat16),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[dout_spec, dout_spec, rhs_spec],
          out_specs=out_spec,
          grid=(1, num_active_tiles, 1),
          scratch_shapes=[pltpu.VMEM((tile_m, k), jnp.float32)],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary")
      ),
  )(group_metadata, group_offset, dout1, dout3, rhs13)


@partial(jax.jit, static_argnames=("tile_m", "output_dtype"))
def _paired_dweight(
    lhs,
    dout1,
    dout3,
    group_sizes,
    *,
    tile_m,
    output_dtype=jnp.bfloat16,
):
  """Compute packed dWeight with either BF16 or BF16-rounded FP32 output.

  The BF16 path uses separate FP32 accumulators. The FP32-output path instead
  uses the output ref as the persistent accumulator across the sequential grid
  to stay below the 32 MiB scoped VMEM limit. Both paths zero at every group
  boundary and round once to BF16 after the final tile for that group. The
  FP32-output values therefore exactly match the BF16 output widened to FP32.
  """
  if output_dtype not in (jnp.bfloat16, jnp.float32):
    raise ValueError("Paired dWeight output must be bfloat16 or float32")
  if dout1.shape != dout3.shape or lhs.shape[0] != dout1.shape[0]:
    raise ValueError("Invalid paired dWeight dimensions")
  m, k = lhs.shape
  n = dout1.shape[1]
  groups = group_sizes.shape[0]
  if m % tile_m:
    raise ValueError("The total row count must be divisible by tile_m")
  group_metadata, num_active_tiles = _metadata(
      group_sizes,
      m=m,
      tile_m=tile_m,
      groups=groups,
      visit_empty_groups=True,
  )
  group_offset = jnp.array([0], dtype=jnp.int32)

  def kernel(
      group_metadata,
      group_offset,
      lhs,
      dout1,
      dout3,
      out13,
      *scratch,
  ):
    del group_offset
    grid_id = pl.program_id(2)
    group_ids = group_metadata[1]
    group = group_ids[grid_id]
    previous_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
    group_has_changed = jnp.logical_or(
        grid_id == 0, group_ids[previous_grid_id] != group
    )

    @pl.when(group_has_changed)
    def _zero_accumulators():
      if output_dtype == jnp.float32:
        out13[0, ...] = jnp.zeros_like(out13[0, ...])
        out13[1, ...] = jnp.zeros_like(out13[1, ...])
      else:
        scratch[0][...] = jnp.zeros_like(scratch[0])
        scratch[1][...] = jnp.zeros_like(scratch[1])

    group_has_rows = (
        _get_group_size(
            grid_id=grid_id, group_metadata=group_metadata
        )
        > 0
    )

    @pl.when(group_has_rows)
    def _accumulate():
      lhs_mask = _get_store_mask(
          grid_id=grid_id,
          group_metadata=group_metadata,
          tm=tile_m,
          tn=k,
      )
      dout_mask = _get_store_mask(
          grid_id=grid_id,
          group_metadata=group_metadata,
          tm=tile_m,
          tn=n,
      )
      loaded_lhs = lax.select(
          lhs_mask[...],
          lhs[...].astype(jnp.float32),
          jnp.zeros_like(lhs, jnp.float32),
      ).swapaxes(0, 1)
      loaded_dout1 = lax.select(
          dout_mask[...],
          dout1[...].astype(jnp.float32),
          jnp.zeros_like(dout1, jnp.float32),
      )
      loaded_dout3 = lax.select(
          dout_mask[...],
          dout3[...].astype(jnp.float32),
          jnp.zeros_like(dout3, jnp.float32),
      )
      update1 = lax.dot(
          loaded_lhs.astype(jnp.bfloat16),
          loaded_dout1.astype(jnp.bfloat16),
          preferred_element_type=jnp.float32,
      )
      update3 = lax.dot(
          loaded_lhs.astype(jnp.bfloat16),
          loaded_dout3.astype(jnp.bfloat16),
          preferred_element_type=jnp.float32,
      )
      if output_dtype == jnp.float32:
        out13[0, ...] += update1
        out13[1, ...] += update3
      else:
        scratch[0][...] += update1
        scratch[1][...] += update3

    is_last_grid_id = grid_id == pl.num_programs(2) - 1
    next_grid_id = jnp.where(is_last_grid_id, grid_id, grid_id + 1)
    group_is_changing = jnp.logical_or(
        is_last_grid_id, group_ids[next_grid_id] != group
    )

    @pl.when(group_is_changing)
    def _store_accumulators():
      if output_dtype == jnp.float32:
        out13[0, ...] = out13[0, ...].astype(jnp.bfloat16).astype(jnp.float32)
        out13[1, ...] = out13[1, ...].astype(jnp.bfloat16).astype(jnp.float32)
      else:
        out13[0, ...] = scratch[0][...].astype(jnp.bfloat16)
        out13[1, ...] = scratch[1][...].astype(jnp.bfloat16)

  def lhs_indices(n_i, k_i, grid_id, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[2][grid_id], 0

  def dout_indices(n_i, k_i, grid_id, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[2][grid_id], 0

  def out_indices(n_i, k_i, grid_id, group_metadata, group_offset):
    del n_i, k_i, group_offset
    return group_metadata[1][grid_id], 0, 0, 0

  lhs_spec = pl.BlockSpec((tile_m, k), lhs_indices)
  dout_spec = pl.BlockSpec((tile_m, n), dout_indices)
  out_spec = pl.BlockSpec((None, 2, k, n), out_indices)
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((groups, 2, k, n), output_dtype),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[lhs_spec, dout_spec, dout_spec],
          out_specs=out_spec,
          grid=(1, 1, num_active_tiles),
          scratch_shapes=(
              []
              if output_dtype == jnp.float32
              else [
                  pltpu.VMEM((k, n), jnp.float32),
                  pltpu.VMEM((k, n), jnp.float32),
              ]
          ),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary")
      ),
  )(group_metadata, group_offset, lhs, dout1, dout3)


@lru_cache(maxsize=None)
def _build_paired_ragged_dot_kernels(tile_m: int):
  with jax.default_device(jax.local_devices()[0]):

    @pallas.jax_op(f"tt_redacted::paired_ragged_dot_fwd_tm{tile_m}")
    def fwd_op(
        lhs: jax.Array,
        rhs13: jax.Array,
        group_sizes: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
      return _paired_forward(lhs, rhs13, group_sizes, tile_m=tile_m)

    @pallas.jax_op(f"tt_redacted::paired_ragged_dot_bwd_tm{tile_m}")
    def bwd_op(
        dout1: jax.Array,
        dout3: jax.Array,
        lhs: jax.Array,
        rhs13: jax.Array,
        group_sizes: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
      grad_lhs = _paired_dinput(
          dout1, dout3, rhs13, group_sizes, tile_m=tile_m
      )
      grad_rhs13 = _paired_dweight(
          lhs, dout1, dout3, group_sizes, tile_m=tile_m
      )
      return grad_lhs, grad_rhs13

  return fwd_op, bwd_op


@lru_cache(maxsize=None)
def _build_paired_bf16_rounded_fp32_backward(tile_m: int):
  with jax.default_device(jax.local_devices()[0]):

    @pallas.jax_op(
        f"tt_redacted::paired_ragged_dot_bwd_bf16_rounded_fp32_tm{tile_m}"
    )
    def bwd_op(
        dout1: jax.Array,
        dout3: jax.Array,
        lhs: jax.Array,
        rhs13: jax.Array,
        group_sizes: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
      grad_lhs = _paired_dinput(
          dout1, dout3, rhs13, group_sizes, tile_m=tile_m
      )
      grad_rhs13 = _paired_dweight(
          lhs,
          dout1,
          dout3,
          group_sizes,
          tile_m=tile_m,
          output_dtype=jnp.float32,
      )
      return grad_lhs, grad_rhs13

  return bwd_op


class _PairedRaggedDotAutograd(torch.autograd.Function):

  @staticmethod
  def forward(
      ctx,
      lhs,
      rhs13,
      group_sizes,
      tile_m,
      emit_bf16_rounded_fp32_dweight,
  ):
    ctx.save_for_backward(lhs, rhs13, group_sizes)
    ctx.tile_m = tile_m
    ctx.emit_bf16_rounded_fp32_dweight = emit_bf16_rounded_fp32_dweight
    fwd_op, _ = _build_paired_ragged_dot_kernels(tile_m)
    return cast(
        Tuple[torch.Tensor, torch.Tensor],
        fwd_op(lhs, rhs13, group_sizes),
    )

  @staticmethod
  def backward(ctx, grad_output1, grad_output3):
    lhs, rhs13, group_sizes = ctx.saved_tensors
    if ctx.emit_bf16_rounded_fp32_dweight:
      bwd_op = _build_paired_bf16_rounded_fp32_backward(ctx.tile_m)
    else:
      _, bwd_op = _build_paired_ragged_dot_kernels(ctx.tile_m)
    grad_lhs, grad_rhs13 = cast(
        Tuple[torch.Tensor, torch.Tensor],
        bwd_op(grad_output1, grad_output3, lhs, rhs13, group_sizes),
    )
    return grad_lhs, grad_rhs13, None, None, None


def paired_segment_matmul_pallas_2d(
    lhs: torch.Tensor,
    rhs13: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    tile_m: int = 1024,
    emit_bf16_rounded_fp32_dweight: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Apply two equal-shaped per-expert projections while sharing lhs loads.

  ``rhs13`` has shape ``(E, 2, K, N)``; the two outputs correspond to indices
  zero and one on its projection dimension.

  ``group_sizes`` must sum exactly to ``lhs.shape[0]``. The output tail is not
  defined for capacity-padded inputs.

  ``emit_bf16_rounded_fp32_dweight`` stores the usual BF16-rounded dWeight
  values in FP32 so GraphTrainer can feed them directly to FP32 FSDP reduction.
  """
  if rhs13.ndim != 4 or rhs13.shape[1] != 2:
    raise ValueError("Packed paired weights must have shape (E, 2, K, N)")
  if group_sizes.dtype != torch.int32:
    group_sizes = group_sizes.to(torch.int32)
  return _PairedRaggedDotAutograd.apply(
      lhs,
      rhs13,
      group_sizes,
      tile_m,
      emit_bf16_rounded_fp32_dweight,
  )
