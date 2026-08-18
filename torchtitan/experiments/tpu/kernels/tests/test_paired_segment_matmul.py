"""Tests for paired MegaBlox expert matmuls."""

import math

from absl.testing import absltest
import torch
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import expert_segment_matmul
from torchtitan.experiments.tpu.kernels import paired_segment_matmul


def _bounded_values(shape, modulus, scale, device):
  values = torch.arange(
      math.prod(shape), device=device, dtype=torch.float32
  )
  return values.remainder(modulus).mul(scale).reshape(shape).bfloat16()


def _signed_bounded_values(shape, modulus, scale, device):
  values = torch.arange(
      math.prod(shape), device=device, dtype=torch.float32
  )
  return (
      values.remainder(modulus)
      .sub(modulus // 2)
      .mul(scale)
      .reshape(shape)
      .bfloat16()
  )


class PairedSegmentMatmulTest(base_device_test.BaseAcceleratorDeviceTest):
  def test_forward_and_backward_parity(self):
    device = self.accelerator_device
    group_sizes = torch.tensor([103, 0, 17, 136], dtype=torch.int32)
    total_m = int(group_sizes.sum().item())
    k, n = 128, 128

    lhs_ref = _bounded_values(
        (total_m, k), 17, 0.003, device
    ).requires_grad_()
    rhs1_ref = _bounded_values(
        (4, k, n), 19, 0.002, device
    ).requires_grad_()
    rhs3_ref = _bounded_values(
        (4, k, n), 23, 0.0015, device
    ).requires_grad_()
    grad1 = _bounded_values((total_m, n), 29, 0.002, device)
    grad3 = _bounded_values((total_m, n), 31, 0.001, device)
    group_sizes_dev = group_sizes.to(device)

    out1_ref = expert_segment_matmul.segment_matmul_pallas_2d(
        lhs_ref, rhs1_ref, group_sizes_dev, tile_m=128
    )
    out3_ref = expert_segment_matmul.segment_matmul_pallas_2d(
        lhs_ref, rhs3_ref, group_sizes_dev, tile_m=128
    )
    grads_ref = torch.autograd.grad(
        (out1_ref, out3_ref),
        (lhs_ref, rhs1_ref, rhs3_ref),
        (grad1, grad3),
    )

    lhs = lhs_ref.detach().clone().requires_grad_(True)
    rhs13 = torch.stack(
        (rhs1_ref.detach(), rhs3_ref.detach()), dim=1
    ).requires_grad_(True)
    out1, out3 = paired_segment_matmul.paired_segment_matmul_pallas_2d(
        lhs,
        rhs13,
        group_sizes_dev,
        tile_m=128,
    )
    grads = torch.autograd.grad(
        (out1, out3),
        (lhs, rhs13),
        (grad1, grad3),
    )

    torch.testing.assert_close(out1, out1_ref, rtol=0, atol=0)
    torch.testing.assert_close(out3, out3_ref, rtol=0, atol=0)
    torch.testing.assert_close(grads[0], grads_ref[0], rtol=0, atol=2e-3)
    torch.testing.assert_close(grads[1][:, 0], grads_ref[1], rtol=0, atol=0)
    torch.testing.assert_close(grads[1][:, 1], grads_ref[2], rtol=0, atol=0)

  def test_bf16_rounded_fp32_backward_exact_parity(self):
    device = self.accelerator_device
    tile_m = 128
    group_sizes = torch.tensor(
        [0, 0, 133, 7, 0, 116, 0], dtype=torch.int32, device=device
    )
    total_m = 256
    k, n = 128, 128
    lhs = _signed_bounded_values((total_m, k), 37, 0.003, device)
    rhs13 = _signed_bounded_values(
        (group_sizes.numel(), 2, k, n), 41, 0.002, device
    )
    grad1 = _signed_bounded_values((total_m, n), 43, 0.0025, device)
    grad3 = _signed_bounded_values((total_m, n), 47, 0.0015, device)

    _, baseline_backward = (
        paired_segment_matmul._build_paired_ragged_dot_kernels(tile_m)
    )
    rounded_fp32_backward = (
        paired_segment_matmul._build_paired_bf16_rounded_fp32_backward(
            tile_m
        )
    )
    baseline_dinput, baseline_dweight = baseline_backward(
        grad1, grad3, lhs, rhs13, group_sizes
    )
    actual_dinput, actual_dweight = rounded_fp32_backward(
        grad1, grad3, lhs, rhs13, group_sizes
    )

    self.assertEqual(actual_dweight.dtype, torch.float32)
    torch.testing.assert_close(actual_dinput, baseline_dinput, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_dweight, baseline_dweight.float(), rtol=0, atol=0
    )


if __name__ == "__main__":
  absltest.main()
