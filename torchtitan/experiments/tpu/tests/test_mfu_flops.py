# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu.qwen3 import qwen3_configs


class MaxTextQwen3FlopsTest(absltest.TestCase):

  def test_qwen3_8b_matches_maxtext(self) -> None:
    model_config = qwen3_configs["8B"]
    with torch.device("meta"):
      model = model_config.build()

    self.assertEqual(
        model_config.get_nparams_and_flops(model, seq_len=2048)[1],
        47_220_523_008,
    )

  def test_qwen3_30b_a3b_matches_maxtext(self) -> None:
    model_config = qwen3_configs["30B-A3B"]
    with torch.device("meta"):
      model = model_config.build()

    self.assertEqual(
        model_config.get_nparams_and_flops(model, seq_len=2048)[1],
        20_665_860_096,
    )


if __name__ == "__main__":
  absltest.main()
