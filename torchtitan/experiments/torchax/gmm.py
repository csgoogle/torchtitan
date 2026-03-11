import functools
import jax
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.megablox.ops import gmm as megablox_gmm
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import torchtitan.tools.logging
import torch
import torch_xla2

logger = torchtitan.tools.logging.logger

# TPU MXU size is 256x256 for v6 and later.
# Change to 128 for v5 and earlier may have slightly better performance.
BLOCK_SIZE = 256

# Tiling Strategy Format: (Tokens, In_Features, Out_Features)
TILING = (1024, 512, 512)


def _pad_to_block(x, axis, block_size=256):
  """Pads a specific axis to the nearest multiple of block_size."""
  dim = x.shape[axis]
  remainder = dim % block_size
  if remainder == 0:
    return x, 0

  pad_amt = block_size - remainder
  pad_width = [(0, 0)] * x.ndim
  pad_width[axis] = (0, pad_amt)
  x_padded = jnp.pad(x, pad_width, mode='constant', constant_values=0)
  return x_padded, pad_amt


def _gmm_tpu_kernel_impl(
    x: jax.Array,
    weights: jax.Array,
    offsets: jax.Array,
) -> jax.Array:
  """Core Kernel Implementation."""

  if x.dtype != jnp.bfloat16:
    x = x.astype(jnp.bfloat16)
  if weights.dtype != jnp.bfloat16:
    weights = weights.astype(jnp.bfloat16)

  num_experts = weights.shape[0]
  offsets = jnp.reshape(offsets, (-1,))[:num_experts]
  offsets_padded = jnp.pad(offsets, (1, 0), constant_values=0)
  group_sizes = jnp.maximum(
      (offsets_padded[1:] - offsets_padded[:-1]).astype(jnp.int32), 0
  )

  total_tokens, _ = x.shape
  out_dim = weights.shape[2]

  x, token_pad = _pad_to_block(x, 0, BLOCK_SIZE)
  if token_pad > 0:
    group_sizes = group_sizes.at[-1].add(token_pad)

  x, _ = _pad_to_block(x, 1, BLOCK_SIZE)
  weights, _ = _pad_to_block(weights, 1, BLOCK_SIZE)
  weights, out_pad = _pad_to_block(weights, 2, BLOCK_SIZE)

  t0 = min(TILING[0], x.shape[0])
  t1 = min(TILING[1], x.shape[1])
  t2 = min(TILING[2], weights.shape[2])

  out = megablox_gmm(
      lhs=x,
      rhs=weights,
      group_sizes=group_sizes,
      preferred_element_type=jnp.bfloat16,
      tiling=(t0, t1, t2),
  )

  if token_pad > 0:
    out = out[:total_tokens, :]
  if out_pad > 0:
    out = out[:, :out_dim]
  return out


def _fsdp_runner_w_sharding(mesh, x, weights, offsets_stacked):
  def kernel_wrapper(local_x, local_weights, local_offsets):
    return _gmm_tpu_kernel_impl(local_x, local_weights, local_offsets)

  return shard_map.shard_map(
      kernel_wrapper,
      mesh=mesh,
      in_specs=(P('fsdp', None), P(None, None, None), P('fsdp', None)),
      out_specs=P('fsdp', None),
      check_rep=False,
  )(x, weights, offsets_stacked)


def _tp_col_runner_w_sharding(mesh, x, weights, offsets_stacked):
  def kernel_wrapper(local_x, local_weights, local_offsets):
    return _gmm_tpu_kernel_impl(local_x, local_weights, local_offsets)

  return shard_map.shard_map(
      kernel_wrapper,
      mesh=mesh,
      in_specs=(P('fsdp', None), P(None, None, 'tp'), P('fsdp', None)),
      out_specs=P('fsdp', 'tp'),
      check_rep=False,
  )(x, weights, offsets_stacked)


def _tp_row_runner_w_sharding(mesh, x, weights, offsets_stacked):
  def kernel_wrapper(local_x, local_weights, local_offsets):
    out = _gmm_tpu_kernel_impl(local_x, local_weights, local_offsets)
    out = jax.lax.psum(out, 'tp')
    return out

  return shard_map.shard_map(
      kernel_wrapper,
      mesh=mesh,
      in_specs=(P('fsdp', 'tp'), P(None, 'tp', None), P('fsdp', None)),
      out_specs=P('fsdp', None),
      check_rep=False,
  )(x, weights, offsets_stacked)


def declare_gmm_kernel(env, mesh):
  """Declares and overrides torch._grouped_mm with a TPU-optimized kernel.

  This function sets up different sharded runners (FSDP, TP_COL, TP_ROW)
  for grouped matrix multiplication based on the provided mesh and input
  shapes, and then overrides the `torch._grouped_mm` operator in the
  given environment to use these optimized kernels.

  Args:
    env: The environment object where the op definition will be overridden.
    mesh: The JAX device mesh used for sharding.

  Returns:
    True if the kernel was successfully declared and overridden, False
    otherwise.
  """
  try:
    runner_fsdp = jax.jit(functools.partial(_fsdp_runner_w_sharding, mesh))
    runner_tp_col = jax.jit(functools.partial(_tp_col_runner_w_sharding, mesh))
    runner_tp_row = jax.jit(functools.partial(_tp_row_runner_w_sharding, mesh))

    tp_size = mesh.shape['tp']

    def custom_grouped_mm(mat1, mat2, *, offs=None, **kwargs):
      if offs is None:
        raise ValueError('TPU GMM Wrapper requires "offs".')

      # Check if offs is already sharded/stacked (2D)
      assert (
          offs.dim() == 2
      ), 'JAX GMM kenerl requires 2D offsets [Num_Chips, Experts].'

      j_mat1, j_mat2, j_offs_stacked = torch_xla2.interop.jax_view(
          (mat1, mat2, offs)
      )

      in_features = j_mat2.shape[1]
      out_features = j_mat2.shape[2]

      # Heuristic to decide which runner to use.
      if tp_size <= 1:
        res = runner_fsdp(j_mat1, j_mat2, j_offs_stacked)
      else:
        if in_features > out_features:
          # Assuming FNN/MoE layers always scale up and then scale down.
          res = runner_tp_row(j_mat1, j_mat2, j_offs_stacked)
        else:
          res = runner_tp_col(j_mat1, j_mat2, j_offs_stacked)

      return torch_xla2.interop.torch_view(res)

    logger.info(
        'Overriding torch._grouped_mm with TPU grouped MM kernel...'
    )
    env.override_op_definition(torch._grouped_mm, custom_grouped_mm)
    return True
  except Exception as e:
    logger.error('Failed to declare GMM kernel: %s', e)
    return False
