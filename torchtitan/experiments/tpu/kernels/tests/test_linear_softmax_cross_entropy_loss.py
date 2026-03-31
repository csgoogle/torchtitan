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

"""Tests for linear_softmax_cross_entropy_loss."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torchtitan.components import loss
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import linear_softmax_cross_entropy_loss


class LinearSoftmaxCrossEntropyLossTest(
    base_device_test.BaseAcceleratorDeviceTest
):
  """Tests for LinearSoftmaxCrossEntropyLoss."""

  @parameterized.parameters(
      (torch.float32, "sum", "mosaic_tpu", 1e-4, 1e-5),
      (torch.float32, "sum", "xla", 1e-4, 1e-5),
      (torch.float32, "mean", "mosaic_tpu", 1e-4, 1e-5),
      (torch.float32, "mean", "xla", 1e-4, 1e-5),
      (torch.bfloat16, "sum", "mosaic_tpu", 1.6e-2, 1e-5),
      (torch.bfloat16, "sum", "xla", 1.6e-2, 1e-5),
      (torch.bfloat16, "mean", "mosaic_tpu", 1.6e-2, 1e-5),
      (torch.bfloat16, "mean", "xla", 1.6e-2, 1e-5),
  )
  def test_parity(self, dtype, reduction, implementation, rtol, atol):
    device = self.accelerator_device
    b, s, h, v = 32, 32, 128, 1024  # Batch, SeqLen, Hidden, Vocab

    # Generate on CPU first
    x = torch.randn(b, s, h, dtype=dtype)
    weights = torch.randn(h, v, dtype=dtype)
    labels = torch.randint(0, v, (b, s), dtype=torch.int64)

    # Move to TPU
    x_tpu = x.to(device)
    weights_tpu = weights.to(device)
    labels_tpu = labels.to(device)

    # Reference Math on TPU
    # x is (B, S, H), weights is (H, V)
    # logits = x @ weights -> (B, S, V)
    # labels is (B, S)
    logits = x_tpu @ weights_tpu

    if reduction == "sum":
      expected_out = loss.cross_entropy_loss(logits, labels_tpu)
    else:
      expected_out = torch.nn.functional.cross_entropy(
          logits.view(-1, v).float(),
          labels_tpu.view(-1),
          reduction="mean",
      )

    # TPU Pallas
    # tokamax_loss expects x as (B, H) and labels as (B,).
    # We need to flatten x and labels before passing to tokamax_loss.
    x_flat = x_tpu.view(-1, h)
    labels_flat = labels_tpu.view(-1)

    actual_out = (
        linear_softmax_cross_entropy_loss.linear_softmax_cross_entropy_loss(
            x_flat,
            labels_flat,
            weights_tpu,
            reduction=reduction,
            implementation=implementation,
        )
    )

    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=rtol, atol=atol
    )
    logging.info(
        "Parity %s test passed for %s. Output: %s, Expected: %s",
        reduction,
        dtype,
        actual_out,
        expected_out,
    )

  @parameterized.parameters(
      (torch.float32, "sum", "mosaic_tpu", 1e-4, 1e-5, 1e-2, 1e-2),
      (torch.float32, "sum", "xla", 1e-4, 1e-5, 1e-2, 1e-2),
      (torch.float32, "mean", "mosaic_tpu", 1e-4, 1e-5, 1e-2, 1e-2),
      (torch.float32, "mean", "xla", 1e-4, 1e-5, 1e-2, 1e-2),
      (torch.bfloat16, "sum", "mosaic_tpu", 1.6e-2, 1e-5, 5e-2, 0.5),
      (torch.bfloat16, "sum", "xla", 1.6e-2, 1e-5, 2.0, 1.0),
      (torch.bfloat16, "mean", "mosaic_tpu", 1.6e-2, 1e-5, 5e-2, 1e-2),
      (torch.bfloat16, "mean", "xla", 1.6e-2, 1e-5, 5e-2, 1e-2),
  )
  def test_gradient_new(
      self,
      dtype,
      reduction,
      implementation,
      loss_rtol,
      loss_atol,
      grad_rtol,
      grad_atol,
  ):
    device = self.accelerator_device

    b, h, v = 1024, 128, 1024

    x = torch.randn(b, h, dtype=dtype)
    weights = torch.randn(h, v, dtype=dtype)
    labels = torch.randint(0, v, (b,), dtype=torch.int64)

    x_tpu = x.detach().clone().to(device).requires_grad_(True)
    weights_tpu = weights.detach().clone().to(device).requires_grad_(True)
    labels_tpu = labels.to(device)

    # Reference Math on TPU
    logits = x_tpu @ weights_tpu
    expected_out = torch.nn.functional.cross_entropy(
        logits.float(),
        labels_tpu,
        reduction=reduction,
    )
    expected_out.backward()
    expected_grad_x = x_tpu.grad.clone()
    expected_grad_weights = weights_tpu.grad.clone()

    # Zero out grads
    x_tpu.grad.zero_()
    weights_tpu.grad.zero_()

    # TPU Pallas
    def run_custom(x, l, w):
      return (
          linear_softmax_cross_entropy_loss.linear_softmax_cross_entropy_loss(
              x,
              l,
              w,
              reduction=reduction,
              implementation=implementation,
              b_block_size=128,
              v_block_size=128,
          )
      )

    actual_out = run_custom(x_tpu, labels_tpu, weights_tpu)
    actual_out.backward()
    actual_grad_x = x_tpu.grad
    actual_grad_weights = weights_tpu.grad

    # Compare

    torch.testing.assert_close(
        actual_out.cpu(), expected_out.cpu(), rtol=loss_rtol, atol=loss_atol
    )

    # Try creating a new tensor and copying
    actual_grad_x_new = torch.empty_like(actual_grad_x).copy_(actual_grad_x)
    expected_grad_x_new = torch.empty_like(expected_grad_x).copy_(
        expected_grad_x
    )

    actual_grad_x_cpu = actual_grad_x_new.cpu()
    expected_grad_x_cpu = expected_grad_x_new.cpu()

    torch.testing.assert_close(
        actual_grad_x_cpu, expected_grad_x_cpu, rtol=grad_rtol, atol=grad_atol
    )
    torch.testing.assert_close(
        actual_grad_weights.cpu(),
        expected_grad_weights.cpu(),
        rtol=grad_rtol,
        atol=grad_atol,
    )
    logging.info(
        "Gradient %s test passed for %s.",
        reduction,
        dtype,
    )


if __name__ == "__main__":
  absltest.main()
