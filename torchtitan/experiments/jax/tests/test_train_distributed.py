"""Tests for the JAX distributed training setup."""

from absl.testing import absltest
from absl.testing import parameterized

import tyro

from absl import logging
from flax import nnx
import jax
from torchtitan.experiments.jax import redacted
from torchtitan.experiments.jax import jax_job_config
from torchtitan.experiments.jax import train_minimal


class TestJaxDistributed(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="llama3_debug_scan",
          model_name="llama3",
          model_flavor="debug",
          use_scan=True,
      ),
      dict(
          testcase_name="llama3_debug_no_scan",
          model_name="llama3",
          model_flavor="debug",
          use_scan=False,
      ),
      dict(
          testcase_name="redacted_debug_scan",
          model_name="redacted",
          model_flavor="debugmodel",
          use_scan=True,
      ),
      dict(
          testcase_name="redacted_debug_no_scan",
          model_name="redacted",
          model_flavor="debugmodel",
          use_scan=False,
      ),
  ])
  def test_train_jax(self, model_name, model_flavor, use_scan):
    scan_flag = (
        "--jax_config.use_scan"
        if use_scan
        else "--jax_config.no_use_scan"
    )
    args = [
        f"--model.name={model_name}",
        f"--model.flavor={model_flavor}",
        scan_flag,
        "--jax_config.model_layer_override=1",
        "--training.dataset_path=tests/assets/c4_test",
        "--model.hf_assets_path=tests/assets/tokenizer",
        "--training.seq_len=128",
        "--training.steps=3",
        "--training.global_batch_size=8",
        "--activation_checkpoint.mode=full",
    ]

    job_config = tyro.cli(jax_job_config.JaxJobConfig, args=args)
    success = train_minimal.main_train_loop(job_config)
    self.assertTrue(success)


if __name__ == "__main__":
  absltest.main()
