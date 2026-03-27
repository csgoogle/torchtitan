"""Tests for the minimal distributed training setup for Llama3 and dense Qwen3 models."""

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_distributed_device_test
import torchtitan.experiments.tpu.train_minimal

from internal.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


class TrainMinimalDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest):
  """Tests the minimal distributed training setup."""

  @parameterized.named_parameters([

      # llama3 (with TT dtensor implementation)
      dict(
          testcase_name="llama3_tp_dtensor",
          model_name="llama3",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
      dict(
          testcase_name="llama3_tp_dtensor_compile",
          model_name="llama3",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          enable_compile=True,
      ),
      dict(
          testcase_name="llama3_fsdp_dtensor",
          model_name="llama3",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
      ),
      dict(
          testcase_name="llama3_fsdp_dtensor_compile",
          model_name="llama3",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          enable_compile=True,
      ),
      # qwen3 (with TT dtensor implementation)
      dict(
          testcase_name="qwen3_tp_dtensor",
          model_name="qwen3",
          config_file="third_party/py/torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
      dict(
          testcase_name="qwen3_tp_dtensor_compile",
          model_name="qwen3",
          config_file="third_party/py/torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          enable_compile=True,
      ),
      dict(
          testcase_name="qwen3_fsdp_dtensor",
          model_name="qwen3",
          config_file="third_party/py/torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
      ),
      dict(
          testcase_name="qwen3_fsdp_dtensor_compile",
          model_name="qwen3",
          config_file="third_party/py/torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          enable_compile=True,
      ),
      # deepseek_v3 (dtensor)
      dict(
          testcase_name="deepseek_v3_tp_dtensor",
          model_name="deepseek_v3",
          config_file="third_party/py/torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          skip_devices=[
              # TODO: b/482057603 - Issue with mixing DTensor and Tensor.
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="deepseek_v3_fsdp_dtensor",
          model_name="deepseek_v3",
          config_file="third_party/py/torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
      ),
    ])
  def test_train_minimal_distributed(
      self,
      model_name,
      config_file,
      data_parallel_shard_degree,
      tensor_parallel_degree,
      skip_devices=None,
      optimizer_implementation="foreach",
      enable_compile=False,
  ):
    self._test_train_distributed(
        [
            f"--model.name={model_name}",
            f"--job.config_file={config_file}",
            f"--optimizer.implementation={optimizer_implementation}",
            "--model.hf_assets_path=third_party/py/torchtitan/tests/assets/tokenizer",
            "--training.dataset_path=third_party/py/torchtitan/tests/assets/c4_test",
            "--training.seq_len=128",
            "--training.dataset=c4_test",
            "--training.steps=3",
            # Adding flags to avoid b/471023749
            "--training.mixed_precision_param=float32",
            "--training.mixed_precision_reduce=float32",
            "--training.local_batch_size=4",
        ],
        data_parallel_shard_degree,
        tensor_parallel_degree,
        skip_devices,
        torchtitan.experiments.tpu.train_minimal.start_trainer,
        enable_compile=enable_compile)

if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
