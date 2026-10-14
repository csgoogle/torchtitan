# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from types import SimpleNamespace

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu.mfu_flops import (
    get_maxtext_decoder_nparams_and_flops,
    use_maxtext_flops,
)
from torchtitan.experiments.tpu.qwen3 import (
    config_registry as tpu_qwen3_config_registry,
    model_registry as tpu_qwen3_model_registry,
    qwen3_configs,
)


def _build(model_config):
  with torch.device("meta"):
    return model_config.build()


class MaxTextQwen3FlopsTest(absltest.TestCase):

  def test_qwen3_8b_matches_maxtext(self) -> None:
    model_config = qwen3_configs["8B"]
    self.assertEqual(
        get_maxtext_decoder_nparams_and_flops(
            model_config, _build(model_config), seq_len=2048
        )[1],
        47_220_523_008,
    )

  def test_qwen3_30b_a3b_matches_maxtext(self) -> None:
    model_config = qwen3_configs["30B-A3B"]
    self.assertEqual(
        get_maxtext_decoder_nparams_and_flops(
            model_config, _build(model_config), seq_len=2048
        )[1],
        20_665_860_096,
    )


class FlopsConventionTest(absltest.TestCase):

  def test_qwen3_model_config_reports_torchtitan_flops(self) -> None:
    model_config = tpu_qwen3_model_registry("30B-A3B").model
    self.assertEqual(
        model_config.get_nparams_and_flops(
            _build(model_config), seq_len=2048
        )[1],
        23_083_044_864,
    )

  def test_use_maxtext_flops_replaces_the_trainer_value(self) -> None:
    model_config = qwen3_configs["30B-A3B"]
    model = _build(model_config)
    trainer = SimpleNamespace(
        model_parts=[model],
        metrics_processor=SimpleNamespace(
            num_flops_per_token=model_config.get_nparams_and_flops(
                model, seq_len=2048
            )[1]
        ),
    )

    use_maxtext_flops(trainer, model_config, seq_len=2048)

    self.assertEqual(
        trainer.metrics_processor.num_flops_per_token, 20_665_860_096
    )

  def test_only_maxtext_recipes_report_maxtext_flops(self) -> None:
    """MaxText FLOPs belong to the ``*maxtext*`` recipes and no others."""
    recipes = {
        name: fn
        for name, fn in inspect.getmembers(
            tpu_qwen3_config_registry, inspect.isfunction
        )
        if fn.__module__ == tpu_qwen3_config_registry.__name__
        and not name.startswith("_")
        and not inspect.signature(fn).parameters
    }
    self.assertTrue(any("maxtext" in name for name in recipes))
    for name, recipe in recipes.items():
      with self.subTest(recipe=name):
        self.assertEqual(
            recipe().tpu_config.maxtext_flops, "maxtext" in name
        )


if __name__ == "__main__":
  absltest.main()
