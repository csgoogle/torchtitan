"""Tests for the JAX distributed training setup."""

from absl.testing import absltest
from absl.testing import parameterized

import tyro
from torchtitan.experiments.jax import jax_job_config
from torchtitan.experiments.jax import train_minimal


class TestJaxDistributed(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="llama3_8b_scan",
          model_name="llama3",
          use_scan=True,
      ),
      dict(
          testcase_name="llama3_8b_no_scan",
          model_name="llama3",
          use_scan=False,
      ),
  ])
  def test_train_jax(self, model_name, use_scan):
    scan_flag = (
        "--jax_config.use_scan"
        if use_scan
        else "--jax_config.no_use_scan"
    )
    args = [
        f"--model.name={model_name}",
        "--model.flavor=8B",
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
