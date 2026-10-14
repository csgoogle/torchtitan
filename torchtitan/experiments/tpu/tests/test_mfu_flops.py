# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu.qwen3 import (
    MaxTextQwen3Config,
    model_registry as tpu_qwen3_model_registry,
    qwen3_configs,
)


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

  def test_qwen3_registry_can_opt_out_of_maxtext_flops(self) -> None:
    for flavor, attn_backend in (("30B-A3B", "flex"), ("8B", "sdpa")):
      with self.subTest(flavor=flavor):
        default = tpu_qwen3_model_registry(flavor, attn_backend=attn_backend)
        plain = tpu_qwen3_model_registry(
            flavor, attn_backend=attn_backend, maxtext_flops=False
        )
        self.assertIsInstance(default.model, MaxTextQwen3Config)
        self.assertNotIsInstance(plain.model, MaxTextQwen3Config)

    model_config = tpu_qwen3_model_registry(
        "30B-A3B", maxtext_flops=False
    ).model
    with torch.device("meta"):
      model = model_config.build()
    self.assertEqual(
        model_config.get_nparams_and_flops(model, seq_len=2048)[1],
        23_083_044_864,
    )


if __name__ == "__main__":
  absltest.main()
