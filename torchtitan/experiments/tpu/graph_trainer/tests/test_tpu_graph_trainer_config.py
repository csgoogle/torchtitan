# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for TPU GraphTrainer config conversion."""

import inspect
import os
import typing
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.graph_trainer import llama3 as llama3_gt
from torchtitan.experiments.graph_trainer import qwen3 as qwen3_gt
from torchtitan.experiments.tpu.graph_trainer.config import (
    to_tpu_graph_trainer_config,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.qwen3.config_registry import qwen3_8b
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig


class TpuGraphTrainerConfigTest(parameterized.TestCase):

  @parameterized.parameters("default", "full", "eager")
  def test_cli_memory_policy_is_propagated(self, policy: str) -> None:
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            f"--tpu_config.graph_trainer_memory_policy={policy}",
        ]),
    )

    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertEqual(config.tpu_config.graph_trainer_memory_policy, policy)
    self.assertEqual(graph_config.compile.memory_policy, policy)

  def test_unsupported_memory_policy_is_rejected_during_conversion(self) -> None:
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            "--tpu_config.graph_trainer_memory_policy=sac_and_offload",
        ]),
    )

    self.assertEqual(
        config.tpu_config.graph_trainer_memory_policy,
        "sac_and_offload",
    )
    with self.assertRaisesRegex(ValueError, "does not support"):
      to_tpu_graph_trainer_config(
          config,
          qwen3_gt.model_registry,
          target_cls=TpuGraphTrainer.Config,
      )

  def test_qwen3_8b_explicitly_uses_default_graph_policy(self) -> None:
    config = qwen3_8b()

    self.assertEqual(
        config.tpu_config.graph_trainer_memory_policy,
        "default",
    )

    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )
    self.assertEqual(graph_config.compile.memory_policy, "default")


class ScanWiringTest(parameterized.TestCase):
  """`--tpu-config.use-scan` has to reach the spec this lane actually runs.

  The spec `to_graph_trainer_config` returns is a *copy*, taken before
  `train.py` installs its `patched_parallelize` on the original -- so the patch
  that carries `tpu_use_scan` never arrives, and the flag was accepted, logged,
  and silently ignored. At 70B that meant asking for 166.72 GiB of a 94.75 GiB
  device while every log line said scan was on. The conversion is where the
  copy is built, so the conversion is where the wiring has to happen.
  """

  def _config(self, *extra: str) -> TPUTrainerConfig:
    return typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.llama3",
            "--config=llama3_debugmodel",
            "--parallelism.use_simple_fsdp",
            *extra,
        ]),
    )

  def _convert(self, config: TPUTrainerConfig):
    return to_tpu_graph_trainer_config(
        config,
        llama3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

  def test_use_scan_installs_the_scan_parallelize_fn(self) -> None:
    graph_config = self._convert(self._config("--tpu_config.use_scan"))

    # Compare against the un-scanned conversion rather than against a name:
    # the failure being guarded is "the graph_trainer default survived", and
    # only the two side by side can show that.
    baseline = self._convert(self._config())
    self.assertIsNot(
        graph_config.model_spec.parallelize_fn,
        baseline.model_spec.parallelize_fn,
    )
    self.assertIn(
        "parallelize_llama_scan",
        inspect.getsource(graph_config.model_spec.parallelize_fn),
    )

  def test_scan_off_leaves_the_graph_trainer_parallelize_fn_alone(self) -> None:
    graph_config = self._convert(self._config())

    self.assertNotIn(
        "parallelize_llama_scan",
        inspect.getsource(graph_config.model_spec.parallelize_fn),
    )

  @parameterized.parameters("0", "1")
  def test_peel_override_is_resolved_at_conversion_time(
      self, peel: str
  ) -> None:
    """The peel is bound into the closure, not read again at parallelize time.

    `TPU_SCAN_PEEL_FIRST_LAYER` exists so the peel's cost is measurable against
    an otherwise identical scan; a value read late would attribute the wrong
    arm.
    """
    with mock.patch.dict(
        os.environ, {"TPU_SCAN_PEEL_FIRST_LAYER": peel}, clear=False
    ):
      graph_config = self._convert(self._config("--tpu_config.use_scan"))
      bound = inspect.getclosurevars(
          graph_config.model_spec.parallelize_fn
      ).nonlocals

    self.assertEqual(bound["peel_first_layer"], peel == "1")


if __name__ == "__main__":
  absltest.main()
