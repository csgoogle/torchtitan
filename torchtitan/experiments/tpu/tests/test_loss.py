# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for TPU-specific loss wrappers."""

from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu.loss import pallas_cross_entropy_loss


class PallasCrossEntropyLossTest(absltest.TestCase):

  def test_slices_physical_vocab_to_logical_size(self) -> None:
    hidden = torch.randn(4, 8)
    weights = torch.randn(8, 12)
    labels = torch.randint(0, 10, (4,))

    kernel = mock.Mock(return_value=torch.tensor(0.0))
    fake_kernel_module = SimpleNamespace(
        linear_softmax_cross_entropy_loss=kernel
    )
    with mock.patch(
        "torchtitan.experiments.tpu.kernels.linear_softmax_cross_entropy_loss",
        fake_kernel_module,
        create=True,
    ):
      pallas_cross_entropy_loss(
          (hidden, weights),
          labels,
          implementation="xla",
          logical_vocab_size=10,
      )

    self.assertEqual(kernel.call_args.args[2].shape, (8, 10))

    with self.assertRaisesRegex(ValueError, "logical_vocab_size"):
      pallas_cross_entropy_loss(
          (hidden, weights),
          labels,
          implementation="xla",
          logical_vocab_size=13,
      )


if __name__ == "__main__":
  absltest.main()
