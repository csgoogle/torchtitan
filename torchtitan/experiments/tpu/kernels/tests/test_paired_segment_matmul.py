"""Accelerator parity tests for the paired expert gate/up kernel."""

import math

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import expert_segment_matmul
from torchtitan.experiments.tpu.kernels import paired_segment_matmul


class PairedSegmentMatmulTest(base_device_test.BaseAcceleratorDeviceTest):
    def test_forward_and_backward_parity(self):
        device = self.accelerator_device
        group_sizes = torch.tensor([0, 103, 0, 17, 136, 0], dtype=torch.int32)
        total_m = int(group_sizes.sum().item())
        k, n = 128, 256

        def bounded_values(shape, modulus, scale):
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

        lhs_ref = bounded_values((total_m, k), 17, 0.003).requires_grad_()
        rhs1_ref = bounded_values((6, k, n), 19, 0.002).requires_grad_()
        rhs3_ref = bounded_values((6, k, n), 23, 0.0015).requires_grad_()
        grad1 = bounded_values((total_m, n), 29, 0.002)
        grad3 = bounded_values((total_m, n), 31, 0.001)
        group_sizes_dev = group_sizes.to(device)

        out1_ref = expert_segment_matmul.segment_matmul_pallas_2d(
            lhs_ref,
            rhs1_ref,
            group_sizes_dev,
            tile_m=128,
            tgmm_tiling=(128, 0, 0),
        )
        out3_ref = expert_segment_matmul.segment_matmul_pallas_2d(
            lhs_ref,
            rhs3_ref,
            group_sizes_dev,
            tile_m=128,
            tgmm_tiling=(128, 0, 0),
        )
        grads_ref = torch.autograd.grad(
            (out1_ref, out3_ref),
            (lhs_ref, rhs1_ref, rhs3_ref),
            (grad1, grad3),
        )

        lhs = lhs_ref.detach().clone().requires_grad_(True)
        rhs1 = rhs1_ref.detach().clone().requires_grad_(True)
        rhs3 = rhs3_ref.detach().clone().requires_grad_(True)
        out1, out3 = paired_segment_matmul.paired_segment_matmul_pallas_2d(
            lhs,
            rhs1,
            rhs3,
            group_sizes_dev,
            tile_m=128,
            tgmm_tile_m=128,
        )
        grads = torch.autograd.grad(
            (out1, out3),
            (lhs, rhs1, rhs3),
            (grad1, grad3),
        )

        torch.testing.assert_close(out1, out1_ref, rtol=0, atol=0)
        torch.testing.assert_close(out3, out3_ref, rtol=0, atol=0)
        torch.testing.assert_close(grads[0], grads_ref[0], rtol=0, atol=0)
        torch.testing.assert_close(grads[1], grads_ref[1], rtol=0, atol=0)
        torch.testing.assert_close(grads[2], grads_ref[2], rtol=0, atol=0)

    def _paired_vs_unpaired(self, group_sizes, k, n, *, tile_m, tgmm_tile_k):
        """Paired kernel with a K-tiled dWeight against the unpaired kernels.

        Every dWeight element is the same FP32 reduction over the group's rows
        whichever K slab it lands in, so the comparison is bit-exact.
        """
        device = self.accelerator_device
        total_m = int(group_sizes.sum().item())
        groups = group_sizes.shape[0]

        def bounded_values(shape, modulus, scale):
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

        lhs_ref = bounded_values((total_m, k), 17, 0.003).requires_grad_()
        rhs1_ref = bounded_values((groups, k, n), 19, 0.002).requires_grad_()
        rhs3_ref = bounded_values((groups, k, n), 23, 0.0015).requires_grad_()
        grad1 = bounded_values((total_m, n), 29, 0.002)
        grad3 = bounded_values((total_m, n), 31, 0.001)
        group_sizes_dev = group_sizes.to(device)
        ref_tiling = (tile_m, tgmm_tile_k, 0)

        out1_ref = expert_segment_matmul.segment_matmul_pallas_2d(
            lhs_ref, rhs1_ref, group_sizes_dev, tile_m=tile_m, tgmm_tiling=ref_tiling
        )
        out3_ref = expert_segment_matmul.segment_matmul_pallas_2d(
            lhs_ref, rhs3_ref, group_sizes_dev, tile_m=tile_m, tgmm_tiling=ref_tiling
        )
        grads_ref = torch.autograd.grad(
            (out1_ref, out3_ref), (lhs_ref, rhs1_ref, rhs3_ref), (grad1, grad3)
        )

        lhs = lhs_ref.detach().clone().requires_grad_(True)
        rhs1 = rhs1_ref.detach().clone().requires_grad_(True)
        rhs3 = rhs3_ref.detach().clone().requires_grad_(True)
        out1, out3 = paired_segment_matmul.paired_segment_matmul_pallas_2d(
            lhs,
            rhs1,
            rhs3,
            group_sizes_dev,
            tile_m=tile_m,
            tgmm_tile_m=tile_m,
            tgmm_tile_k=tgmm_tile_k,
        )
        grads = torch.autograd.grad((out1, out3), (lhs, rhs1, rhs3), (grad1, grad3))

        torch.testing.assert_close(out1, out1_ref, rtol=0, atol=0)
        torch.testing.assert_close(out3, out3_ref, rtol=0, atol=0)
        for got, want in zip(grads, grads_ref):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_k_tiled_dweight_parity(self):
        self._paired_vs_unpaired(
            torch.tensor([0, 103, 0, 17, 136, 0], dtype=torch.int32),
            256,
            256,
            tile_m=128,
            tgmm_tile_k=128,
        )

    def test_k_tiled_dweight_at_qwen3_235b_shape(self):
        # dim 4096 x moe_hidden 1536: full-K accumulators + output slabs are
        # 105 MB of VMEM, over the 64 MB scoped limit on v7x; 4 K slabs fit.
        self._paired_vs_unpaired(
            torch.tensor([300, 0, 812, 424], dtype=torch.int32),
            4096,
            1536,
            tile_m=256,
            tgmm_tile_k=1024,
        )

    def test_rejects_k_tile_that_does_not_divide_k(self):
        device = self.accelerator_device
        lhs = torch.ones((128, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        rhs = torch.ones((2, 128, 256), device=device, dtype=torch.bfloat16, requires_grad=True)
        sizes = torch.tensor([64, 64], device=device, dtype=torch.int32)
        out1, out3 = paired_segment_matmul.paired_segment_matmul_pallas_2d(
            lhs, rhs, rhs, sizes, tile_m=128, tgmm_tile_m=128, tgmm_tile_k=48
        )
        with self.assertRaisesRegex(ValueError, "tile_k must divide K"):
            torch.autograd.grad((out1, out3), (rhs,), (out1, out3))

    def test_rejects_all_empty_groups(self):
        device = self.accelerator_device
        with self.assertRaisesRegex(ValueError, "at least one routed row"):
            paired_segment_matmul.paired_segment_matmul_pallas_2d(
                torch.empty((0, 128), device=device, dtype=torch.bfloat16),
                torch.empty((4, 128, 256), device=device, dtype=torch.bfloat16),
                torch.empty((4, 128, 256), device=device, dtype=torch.bfloat16),
                torch.zeros(4, device=device, dtype=torch.int32),
                tile_m=128,
                tgmm_tile_m=128,
            )


if __name__ == "__main__":
    absltest.main()
