"""Tests for the Torchax distributed training setup.
"""

import os

from absl.testing import absltest
from absl.testing import parameterized

from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.torchax import train_minimal


# Set standard dummy environment variables for tests if not provided
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")


class TrainTorchaxTest(parameterized.TestCase):
  """Tests the distributed training setup for torchax."""

  @parameterized.named_parameters([
      dict(
          testcase_name="redacted_scan",
          module="torchtitan.experiments.torchax.redacted",
          config="redacted_debugmodel",
          use_scan=True,
      ),
      dict(
          testcase_name="redacted_no_scan",
          module="torchtitan.experiments.torchax.redacted",
          config="redacted_debugmodel",
          use_scan=False,
      ),
      dict(
          testcase_name="redacted_no_scan",
          module="torchtitan.experiments.torchax.redacted",
          config="redacted_debugmodel",
          use_scan=False,
      ),
  ])
  def test_train_torchax(self, module, config, use_scan):
    scan_flag = (
        "--torchax_config.use_scan"
        if use_scan
        else "--torchax_config.no-use-scan"
    )
    args = [
        "--module", module,
        "--config", config,
        scan_flag,
        "--dataloader.dataset_path=tests/assets/c4_test",
        "--dataloader.dataset=c4_test",
        "--hf_assets_path=tests/assets/tokenizer",
        "--training.seq_len=128",
        "--training.steps=3",
        "--training.local_batch_size=8",
    ]

    job_config = ConfigManager().parse_args(args)
    success = train_minimal.main_train_loop(job_config)
    self.assertTrue(success)


if __name__ == "__main__":
  absltest.main()
