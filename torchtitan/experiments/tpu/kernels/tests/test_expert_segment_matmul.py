"""Accelerator tests for the MegaBlox expert grouped matmul."""

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import expert_segment_matmul


class ExpertSegmentMatmulTest(base_device_test.BaseAcceleratorDeviceTest):
    def test_dweight_output_rounds_only_at_final_store(self):
        device = self.accelerator_device
        group_sizes = torch.tensor([64, 64], dtype=torch.int32, device=device)
        lhs = torch.randn(128, 128, dtype=torch.bfloat16, device=device)
        dout = torch.randn(128, 128, dtype=torch.bfloat16, device=device)
        rhs = torch.randn(2, 128, 128, dtype=torch.bfloat16, device=device)
        args = (group_sizes, 128, 0, 0, 128, 128, 128)

        grad_lhs_fp32, grad_rhs_fp32 = expert_segment_matmul._grouped_mm_bwd(
            dout, lhs, rhs, *args, False
        )
        grad_lhs_bf16, grad_rhs_bf16 = expert_segment_matmul._grouped_mm_bwd(
            dout, lhs, rhs, *args, True
        )

        self.assertEqual(grad_rhs_fp32.dtype, torch.float32)
        self.assertEqual(grad_rhs_bf16.dtype, torch.bfloat16)
        torch.testing.assert_close(grad_lhs_bf16, grad_lhs_fp32, rtol=0, atol=0)
        torch.testing.assert_close(
            grad_rhs_bf16,
            grad_rhs_fp32.to(torch.bfloat16),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    absltest.main()
