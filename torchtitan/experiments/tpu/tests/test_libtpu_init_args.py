import os
from unittest import mock

from absl.testing import absltest

from torchtitan.experiments.tpu.train import _configure_libtpu_init_args
from torchtitan.experiments.tpu.qwen3.config_registry import (
    qwen3_moe_30b_fsdp_only_full_ac_profile,
)


class LibtpuInitArgsTest(absltest.TestCase):

  def test_empty_config_is_noop(self):
    with mock.patch.dict(
        os.environ, {"LIBTPU_INIT_ARGS": "--launcher-owned=true"}
    ):
      _configure_libtpu_init_args("")

      self.assertEqual(
          os.environ["LIBTPU_INIT_ARGS"], "--launcher-owned=true"
      )

  def test_preserves_and_appends(self):
    with mock.patch.dict(
        os.environ,
        {"LIBTPU_INIT_ARGS": "--existing=true --shared=true"},
    ):
      _configure_libtpu_init_args("--shared=true --configured=true")

      self.assertEqual(
          os.environ["LIBTPU_INIT_ARGS"],
          "--existing=true --shared=true --configured=true",
      )

  def test_rejects_conflict(self):
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--shared=false"}):
      with self.assertRaisesRegex(ValueError, "Conflicting LIBTPU_INIT_ARGS"):
        _configure_libtpu_init_args("--shared=true")

  def test_rejects_launcher_conflict(self):
    with mock.patch.dict(
        os.environ,
        {"LIBTPU_INIT_ARGS": "--shared=false --shared=true"},
    ):
      with self.assertRaisesRegex(ValueError, "Conflicting LIBTPU_INIT_ARGS"):
        _configure_libtpu_init_args("--configured=true")

  def test_deduplicates_equal_flags(self):
    with mock.patch.dict(
        os.environ,
        {"LIBTPU_INIT_ARGS": "--shared=true --shared=true"},
    ):
      _configure_libtpu_init_args("--configured=true --configured=true")

      self.assertEqual(
          os.environ["LIBTPU_INIT_ARGS"], "--shared=true --configured=true"
      )

  def test_rejects_non_flag(self):
    with mock.patch.dict(os.environ):
      os.environ.pop("LIBTPU_INIT_ARGS", None)

      with self.assertRaisesRegex(ValueError, "Invalid LIBTPU_INIT_ARGS token"):
        _configure_libtpu_init_args("not-a-flag")

  def test_qwen_fsdp_recipe_enables_latency_hiding_scheduler(self):
    config = qwen3_moe_30b_fsdp_only_full_ac_profile()

    self.assertEqual(
        config.tpu_config.libtpu_init_args,
        "--xla_tpu_enable_sparse_core_collective_aggregator=true "
        "--xla_tpu_enable_latency_hiding_layer_scheduler=true",
    )


if __name__ == "__main__":
  absltest.main()
