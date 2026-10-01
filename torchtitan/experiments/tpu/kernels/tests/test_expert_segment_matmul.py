"""Accelerator tests for the MegaBlox expert grouped matmul."""

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import expert_segment_matmul


class ExpertSegmentMatmulTest(base_device_test.BaseAcceleratorDeviceTest):
    def test_dweight_is_stored_in_the_weight_dtype_and_rounds_once(self):
        """The BF16 dWeight equals the exact FP32 dWeight rounded once.

        Small integer inputs keep every FP32 partial sum exact whatever order
        the kernel accumulates in, so the reference is unambiguous. The sums
        reach past 256, where BF16 can no longer hold every integer, so the
        final store does round.
        """
        device = self.accelerator_device
        generator = torch.Generator().manual_seed(0)
        sizes = [64, 64]
        group_sizes = torch.tensor(sizes, dtype=torch.int32, device=device)

        def integers(shape):
            return torch.randint(-32, 33, shape, generator=generator).bfloat16()

        lhs = integers((128, 128))
        dout = integers((128, 128))
        rhs = integers((2, 128, 128))

        _, grad_rhs = expert_segment_matmul._grouped_mm_bwd(
            dout.to(device),
            lhs.to(device),
            rhs.to(device),
            group_sizes,
            128,
            0,
            0,
            128,
            128,
            128,
        )

        want_fp32 = torch.stack(
            [
                rows_lhs.float().T @ rows_dout.float()
                for rows_lhs, rows_dout in zip(
                    lhs.split(sizes), dout.split(sizes)
                )
            ]
        )
        want = want_fp32.to(torch.bfloat16)
        self.assertGreater(int((want.float() != want_fp32).sum()), 0)
        self.assertEqual(grad_rhs.dtype, torch.bfloat16)
        torch.testing.assert_close(grad_rhs.cpu(), want, rtol=0, atol=0)


if __name__ == "__main__":
    absltest.main()
