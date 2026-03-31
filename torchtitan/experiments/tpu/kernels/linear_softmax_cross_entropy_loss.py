# Copyright 2026 The TorchTitan Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Linear Softmax Cross-Entropy Loss wrapped for TPU."""

# Workaround for https://github.com/openxla/tokamax/issues/240
import jaxtyping
# 1. Save the real decorator
_real_jaxtyped = jaxtyping.jaxtyped


# 2. Create a dummy pass-through
def _dummy_jaxtyped(*args, **kwargs):
  def decorator(f):
    return f

  if len(args) == 1 and callable(args[0]):
    return args[0]
  return decorator

# 3. Swap the dummy in, import tokamax, and swap the real one back
jaxtyping.jaxtyped = _dummy_jaxtyped
import tokamax  # pylint: disable=g-import-not-at-top
jaxtyping.jaxtyped = _real_jaxtyped

import jax  # pylint: disable=g-import-not-at-top, g-bad-import-order
import torch
from torch_tpu._internal.pallas import pallas


def _my_fwd(
    x,
    labels,
    weights,
    reduction,
    implementation,
    b_block_size,
    h_block_size,
    v_block_size,
):
  if implementation == "xla":
    from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference  # pylint: disable=g-import-not-at-top

    loss, lse = reference.linear_softmax_cross_entropy_loss_fwd_reference(
        x, labels, weights, reduction=reduction
    )
    return loss, lse

  from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu_kernel as kernel

  loss, lse = kernel.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
      x,
      labels,
      weights,
      b_block_size=b_block_size,
      h_block_size=h_block_size,
      v_block_size=v_block_size,
      reduction=reduction,
  )
  return loss, lse


def _my_bwd(
    dout,
    lse,
    x,
    labels,
    weights,
    reduction,
    implementation,
    b_block_size,
    h_block_size,
    v_block_size,
):
  if implementation == "xla":
    from tokamax._src.ops.linear_softmax_cross_entropy_loss import reference  # pylint: disable=g-import-not-at-top

    grad_x, grad_w = reference.linear_softmax_cross_entropy_loss_bwd_reference(
        dout, lse, x, labels, weights, reduction=reduction
    )
    return grad_x, grad_w

  from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu_kernel as kernel  # pylint: disable=g-import-not-at-top

  grad_x, grad_w = (
      kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
          dout,
          lse,
          x,
          labels,
          weights,
          b_block_size=b_block_size,
          h_block_size=h_block_size,
          v_block_size=v_block_size,
          reduction=reduction,
      )
  )
  return grad_x, grad_w


compiled_fwd = pallas.custom_jax_kernel(_my_fwd, static_argnums=(3, 4, 5, 6, 7))
compiled_bwd = pallas.custom_jax_kernel(_my_bwd, static_argnums=(5, 6, 7, 8, 9))


class TorchLinearLoss(torch.autograd.Function):
  """Linear Softmax Cross-Entropy Loss autograd function."""

  @staticmethod
  def forward(
      ctx,
      x,
      labels,
      weights,
      reduction="sum",
      implementation="mosaic_tpu",
      b_block_size=1024,
      h_block_size=512,
      v_block_size=2048,
  ):
    labels_int32 = labels.to(torch.int32)

    loss, lse = compiled_fwd(  # type: ignore
        x,
        labels_int32,
        weights,
        reduction,
        implementation,
        b_block_size,
        h_block_size,
        v_block_size,
    )
    ctx.save_for_backward(x, labels_int32, weights, lse)
    ctx.reduction = reduction
    ctx.implementation = implementation
    ctx.b_block_size = b_block_size
    ctx.h_block_size = h_block_size
    ctx.v_block_size = v_block_size

    return loss

  @staticmethod
  def backward(ctx, grad_output):
    x, labels, weights, lse = ctx.saved_tensors
    reduction = ctx.reduction
    implementation = ctx.implementation
    b_block_size = ctx.b_block_size
    h_block_size = ctx.h_block_size
    v_block_size = ctx.v_block_size

    grad_x, grad_w = compiled_bwd(  # type: ignore
        grad_output,
        lse,
        x,
        labels,
        weights,
        reduction,
        implementation,
        b_block_size,
        h_block_size,
        v_block_size,
    )

    return grad_x, None, grad_w, None, None, None, None, None


def linear_softmax_cross_entropy_loss(
    x: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    reduction: str = "sum",
    implementation: str = "mosaic_tpu",
    b_block_size: int = 1024,
    h_block_size: int = 512,
    v_block_size: int = 2048,
) -> torch.Tensor:
  """Linear Softmax Cross-Entropy Loss.

  Args:
      x: The last layer output in the dimension of (B, H) where B is batch size
        and H is the hidden dimension.
      labels: The ground truth labels index in the dimension of (B,).
      weights: The linear projection weight matrix in the dimension of (H, V)
        where V is the dimension of the output logits aka vocabulary size.
      reduction: The reduction method for the cross entropy loss. Can be set to
        "sum" or "mean" explicitly.
      implementation: By default "None" will be used to pick the best available
        backend. Can be set to "xla" or "mosaic_tpu" explicitly.
      b_block_size: Block size for batch dimension.
      h_block_size: Block size for hidden dimension.
      v_block_size: Block size for vocab dimension.

  Returns:
      The Cross-Entropy loss.
  """
  return TorchLinearLoss.apply(
      x,
      labels.to(torch.int32),
      weights,
      reduction,
      implementation,
      b_block_size,
      h_block_size,
      v_block_size,
  )
