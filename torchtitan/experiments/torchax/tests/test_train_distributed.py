"""Tests for the Torchax distributed training setup."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from torchtitan.experiments.torchax import torchax_job_config
from torchtitan.experiments.torchax import train_minimal
import tyro

from internal.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

# Set standard dummy environment variables for tests if not provided
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")


class TrainTorchaxTest(parameterized.TestCase):
  """Tests the distributed training setup for torchax."""

  @parameterized.named_parameters([
      dict(
          testcase_name="redacted_tpu_scan",
          model_name="redacted",
          config_file="third_party/py/torchtitan/experiments/tpu/redacted/train_configs/debug_model.toml",
          use_scan=True,
      ),
      dict(
          testcase_name="redacted_tpu_lora_scan",
          model_name="redacted",
          config_file="third_party/py/torchtitan/experiments/tpu/redacted/train_configs/debug_model_lora.toml",
          use_scan=True,
      ),
      dict(
          testcase_name="redacted_tpu_lora_no_scan",
          model_name="redacted",
          config_file="third_party/py/torchtitan/experiments/tpu/redacted/train_configs/debug_model_lora.toml",
          use_scan=False,
      ),
  ])
  def test_train_torchax(self, model_name, config_file, use_scan):
    scan_flag = (
        "--torchax_config.use_scan"
        if use_scan
        else "--torchax_config.no_use_scan"
    )
    args = [
        f"--model.name={model_name}",
        f"--job.config_file={config_file}",
        scan_flag,
        "--training.dataset_path=third_party/py/torchtitan/tests/assets/c4_test",
        "--training.dataset=c4_test",
        "--model.hf_assets_path=third_party/py/torchtitan/tests/assets/tokenizer",
        "--training.seq_len=128",
        "--training.steps=3",
        "--training.local_batch_size=8",
    ]

    job_config = tyro.cli(torchax_job_config.TorchaxJobConfig, args=args)
    success = train_minimal.main_train_loop(job_config)
    self.assertTrue(success)


if __name__ == "__main__":
  g3_multiprocessing.handle_test_main(absltest.main)
