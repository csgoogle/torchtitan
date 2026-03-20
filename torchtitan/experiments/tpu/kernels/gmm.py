"""Grouped Matrix Multiply (GMM) wrapped for TPU."""

from typing import Tuple, cast
import jax
from jax.experimental.pallas.ops.tpu.megablox import ops as jax_gmm_ops
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

# Caches compiled TPU kernels for each unique `group_sizes` configuration.
# Used in place of static_argnums to prevent nested @jax.jit tracer collisions.
_gmm_fwd_cache = {}
_gmm_bwd_cache = {}


def _get_gmm_kernels(group_sizes_tpu: Tuple[int, ...]):
  """Creates and caches the TorchTPU bridging kernels for Grouped Matrix Multiply.

  This function dynamically wraps the underlying JAX Pallas kernels for the
  forward and backward passes into closures, baking in `group_sizes_tpu`
  locally. This prevents unhashable TypeErrors and nested `@jax.jit` tracer
  collisions that can occur when treating group_sizes arg as `static_argnums`.
  Caching ensures the bridging kernels are only instantiated once per unique
  group sizes configuration.

  Args:
    group_sizes_tpu: A tuple of integers representing the sizes of the groups in
      the grouped matrix multiply.

  Returns:
    A tuple `(fwd_kernel, bwd_kernel)` where each is a callable wrapped with
    `pallas.custom_jax_kernel`, ready to be invoked inside a PyTorch
    `autograd.Function`.
  """
  if group_sizes_tpu not in _gmm_fwd_cache:
    # Wrap the JAX kernels in a pure Python closure.
    @jax.jit
    def fwd_fn(lhs_j, rhs_j):
      gs = jnp.array(group_sizes_tpu, dtype=jnp.int32)
      return jax_gmm_ops.gmm(
          lhs_j, rhs_j, group_sizes=gs, tiling=(128, 128, 128), interpret=False
      )

    @jax.jit
    def bwd_fn(dout_j, lhs_j, rhs_j):
      gs = jnp.array(group_sizes_tpu, dtype=jnp.int32)
      grad_lhs = jax_gmm_ops.gmm(
          dout_j,
          rhs_j,
          group_sizes=gs,
          tiling=(128, 128, 128),
          interpret=False,
          transpose_rhs=True,
      )
      grad_rhs = jax_gmm_ops.backend.tgmm(
          lhs_j.swapaxes(0, 1),
          dout_j,
          group_sizes=gs,
          preferred_element_type=jnp.float32,
          tiling=(128, 128, 128),
          interpret=False,
      )
      return grad_lhs, grad_rhs

    _gmm_fwd_cache[group_sizes_tpu] = pallas.custom_jax_kernel(
        fwd_fn, name="gmm_fwd"
    )
    _gmm_bwd_cache[group_sizes_tpu] = pallas.custom_jax_kernel(
        bwd_fn, name="gmm_bwd"
    )

  return _gmm_fwd_cache[group_sizes_tpu], _gmm_bwd_cache[group_sizes_tpu]


class TorchGMM(torch.autograd.Function):
  """Custom PyTorch autograd function for Grouped Matrix Multiply."""

  @staticmethod
  def forward(ctx, lhs, rhs, group_sizes_tpu):
    ctx.save_for_backward(lhs, rhs)
    ctx.group_sizes_tpu = group_sizes_tpu
    fwd_fn, _ = _get_gmm_kernels(group_sizes_tpu)

    result = fwd_fn(lhs, rhs)
    return cast(torch.Tensor, result)

  @staticmethod
  def backward(ctx, grad_output):
    lhs, rhs = ctx.saved_tensors
    _, bwd_fn = _get_gmm_kernels(ctx.group_sizes_tpu)

    out = cast(Tuple[torch.Tensor, torch.Tensor], bwd_fn(grad_output, lhs, rhs))
    grad_lhs, grad_rhs = out[0], out[1]

    return grad_lhs, grad_rhs, None


def grouped_matrix_multiply(
    lhs: torch.Tensor, rhs: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
  """Executes Grouped Matrix Multiply on TPU.

  Mathematically: out_i = lhs_i @ rhs_i
  Where lhs implies shapes [sum_i(group_sizes_i), K] and rhs implies
  [num_groups, K, N].

  Args:
    lhs: Left-hand side tensor.
    rhs: Right-hand side tensor.
    offs: Cumulative offsets of group sizes (matching torch._grouped_mm).

  Returns:
    Output tensor of shape [sum_i(group_sizes_i), N].
  """
  # PyTorch's F.grouped_mm API uses an `offs` tensor (cumulative boundaries)
  # because its ATen GPU kernels rely directly on pointer offsets. The
  # underlying JAX Megablox TPU kernel, however, evaluates block
  # distributions by iterating over explicit bucket sizes (`group_sizes`).
  # We decode PyTorch's offsets back into discrete sizes here to pass to the
  # JAX kernel.
  group_sizes = torch.empty_like(offs)
  group_sizes[0] = offs[0]
  if offs.size(0) > 1:
    group_sizes[1:] = offs[1:] - offs[:-1]

  group_sizes_tpu = tuple(group_sizes.cpu().numpy().tolist())
  return TorchGMM.apply(lhs, rhs, group_sizes_tpu)
