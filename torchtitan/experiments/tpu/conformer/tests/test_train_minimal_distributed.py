"""Tests for the minimal distributed training setup for Conformer."""

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu.conformer import train_minimal as conformer_train_minimal




class TrainMinimalDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests train_minimal execution in distributed settings for Conformer."""

  @parameterized.named_parameters([
      dict(
          testcase_name="conformer_ddp",
          model_name="conformer_tpu",
          config_file="torchtitan/experiments/tpu/conformer/train_configs/conformer.toml",
          data_parallel_shard_degree=1,
          data_parallel_replicate_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=None,
      ),
      dict(
          testcase_name="conformer_fsdp",
          model_name="conformer_tpu",
          config_file="torchtitan/experiments/tpu/conformer/train_configs/conformer.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=None,
      ),
  ])
  def test_conformer_train_minimal(
      self,
      model_name,
      config_file,
      data_parallel_shard_degree,
      tensor_parallel_degree,
      data_parallel_replicate_degree=None,
      extra_args=None,
      skip_devices=None,
  ):
    """Runs execution test specifically for Conformer's minimal trainer."""

    config_args = [
        f"--model.name={model_name}",
        f"--job.config_file={config_file}",
        "--model.flavor=test",
        "--training.seq_len=128",  # Override to smaller length for test speed
        "--training.dataset=random",
        "--training.steps=5",
        "--training.local_batch_size=4",
    ]
    if extra_args:
      config_args.extend(extra_args)

    self._test_train_distributed(
        config_args=config_args,
        tensor_parallel_degree=tensor_parallel_degree,
        data_parallel_shard_degree=data_parallel_shard_degree,
        data_parallel_replicate_degree=data_parallel_replicate_degree,
        skip_devices=skip_devices,
        start_trainer=conformer_train_minimal.start_trainer,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  absltest.main()
