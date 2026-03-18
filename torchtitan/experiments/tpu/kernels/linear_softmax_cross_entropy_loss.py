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


def _my_fwd(x, labels, weights, reduction, implementation):
  return tokamax.linear_softmax_cross_entropy_loss(
      x,
      labels,
      weights,
      reduction=reduction,
      implementation=implementation,
  )


def _my_grad_xw(
    dout, x, labels, weights, reduction, implementation
):
  def loss_fn(x, weights):
    return tokamax.linear_softmax_cross_entropy_loss(
        x,
        labels,
        weights,
        reduction=reduction,
        implementation=implementation,
    )

  _, vjp_fn = jax.vjp(loss_fn, x, weights)
  grad_x, grad_w = vjp_fn(dout)  # type: ignore
  return grad_x, grad_w


compiled_fwd = pallas.custom_jax_kernel(_my_fwd, static_argnums=(3, 4))
compiled_grad_xw = pallas.custom_jax_kernel(
    _my_grad_xw, static_argnums=(4, 5)
)


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
  ):
    ctx.save_for_backward(x, labels.to(torch.int32), weights)
    ctx.reduction = reduction
    ctx.implementation = implementation

    return compiled_fwd(  # type: ignore
        x,
        labels.to(torch.int32),
        weights,
        reduction,
        implementation,
    )

  @staticmethod
  def backward(ctx, grad_output):
    x, labels, weights = ctx.saved_tensors
    reduction = ctx.reduction
    implementation = ctx.implementation

    grad_x, grad_w = compiled_grad_xw(  # type: ignore
        grad_output, x, labels, weights, reduction, implementation
    )

    return grad_x, None, grad_w, None, None, None


def linear_softmax_cross_entropy_loss(
    x: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    reduction: str = "sum",
    implementation: str = "mosaic_tpu",
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

  Returns:
      The Cross-Entropy loss.
  """
  return TorchLinearLoss.apply(
      x,
      labels.to(torch.int32),
      weights,
      reduction,
      implementation,
  )
