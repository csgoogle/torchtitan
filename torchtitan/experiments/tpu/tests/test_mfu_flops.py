# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu.examples.gemma4 import gemma4_configs
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


class MaxTextGemma4FlopsTest(absltest.TestCase):

  def test_gemma4_31b_matches_maxtext(self) -> None:
    model_config = gemma4_configs["31B"]
    with torch.device("meta"):
      model = model_config.build()

    nparams, flops = model_config.get_nparams_and_flops(model, seq_len=2048)
    self.assertGreater(nparams, 30_000_000_000)
    self.assertEqual(flops, 188_705_931_264)

  def test_gemma4_26b_a4b_matches_maxtext(self) -> None:
    model_config = gemma4_configs["26B-A4B"]
    with torch.device("meta"):
      model = model_config.build()

    nparams, flops = model_config.get_nparams_and_flops(model, seq_len=2048)
    self.assertGreater(nparams, 25_000_000_000)
    self.assertEqual(flops, 24_063_442_944)


if __name__ == "__main__":
  absltest.main()
