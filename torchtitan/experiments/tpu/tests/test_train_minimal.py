"""Tests for the minimal training setup for Llama3 models."""

import os
from absl.testing import absltest
from absl.testing import parameterized
import torch.distributed as dist
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
import torchtitan.experiments.tpu.tpu_job_config as tpu_job_config_module
import torchtitan.experiments.tpu.train_minimal


def _setup_and_maybe_init_pg_for_single_process():
  """Sets up the distributed environment for single-process runs."""

  # Set environment variables for torch.distributed initialization.
  # Some models (e.g. DeepSeek V3) trigger distributed components (DeviceMesh)
  # even in single-process runs, requiring these vars.
  os.environ.setdefault("RANK", "0")
  os.environ.setdefault("WORLD_SIZE", "1")
  os.environ.setdefault("MASTER_ADDR", "localhost")
  os.environ.setdefault("MASTER_PORT", "12355")

  # Initialize process group with gloo to avoid PrivateUse1HooksInterface
  # error on TPU/PrivateUse1 backends when running in a single process without
  # full distributed runtime.
  if not dist.is_initialized():
    dist.init_process_group(backend="gloo", init_method="env://")


class TrainMinimalTest(
    base_device_test.BaseAcceleratorDeviceTest):
  """Tests the minimal training setup for Llama3 models."""

  def tearDown(self):
    if dist.is_initialized():
      dist.destroy_process_group()
    super().tearDown()

  @parameterized.named_parameters([
      dict(
          testcase_name="llama3",
          model_name="llama3",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
      ),
      dict(
          testcase_name="llama3_compile",
          model_name="llama3",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
          enable_compile=True,
          # skip on cpu
          skip_devices=[
              # b/327271919 - No inductor CPU support
              device_type.AcceleratorDeviceType.CPU,
          ]
      ),
      dict(
          testcase_name="qwen3",
          model_name="qwen3",
          config_file="third_party/py/torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
      ),
      dict(
          testcase_name="qwen3_compile",
          model_name="qwen3",
          config_file="third_party/py/torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          enable_compile=True,
          # skip on cpu
          skip_devices=[
              # b/327271919 - No inductor CPU support
              device_type.AcceleratorDeviceType.CPU,
          ]
      ),
      dict(
          testcase_name="deepseek_v3",
          model_name="deepseek_v3",
          config_file="third_party/py/torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
      ),
      dict(
          testcase_name="deepseek_v3_compile",
          model_name="deepseek_v3",
          config_file="third_party/py/torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
          enable_compile=True,
      )])
  def test_train_minimal(
      self,
      model_name: str,
      config_file: str,
      skip_devices: list[device_type.AcceleratorDeviceType] | None = None,
      enable_compile: bool = False):
    if skip_devices and self.accelerator_device_type in skip_devices:
      self.skipTest(
          f"Skipping test for device type: {self.accelerator_device_type}")
      return

    config_manager = torchtitan.config.ConfigManager(
        tpu_job_config_module.TPUJobConfig)
    args = [
        f"--model.name={model_name}",
        f"--job.config_file={config_file}",
        "--model.hf_assets_path=third_party/py/torchtitan/tests/assets/tokenizer",
        "--optimizer.implementation=foreach",
        "--training.dataset_path=third_party/py/torchtitan/tests/assets/c4_test",
        "--training.seq_len=128",
        "--training.dataset=c4_test",
        "--training.steps=3",
    ]
    if enable_compile:
      args.append("--compile.enable")
      if self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
        args.append("--compile.backend=tpu")
    config = config_manager.parse_args(args)
    if model_name == "deepseek_v3":
      _setup_and_maybe_init_pg_for_single_process()
    torchtitan.experiments.tpu.train_minimal.start_trainer(config)

  @parameterized.named_parameters([
      dict(
          testcase_name="llama3_parity",
          model_name="llama3",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
          loss_atol=5e-1, loss_rtol=5e-1,  # TODO(tbajpai): Make tighter.
          grad_atol=5e-1, grad_rtol=5e-1,
          param_atol=5e-1, param_rtol=5e-1,
      ),
      dict(
          testcase_name="qwen3_parity",
          model_name="qwen3",
          config_file="third_party/py/torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          loss_atol=5e-1, loss_rtol=5e-1,  # TODO(tbajpai): Make tighter.
          grad_atol=5e-1, grad_rtol=5e-1,
          param_atol=5e-1, param_rtol=5e-1,
      ),
      dict(
          testcase_name="deepseek_v3_parity",
          model_name="deepseek_v3",
          config_file="third_party/py/torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
          skip_devices=[
              # TODO: b/477389342 - Enable when triton CPU tensor error fixed.
              device_type.AcceleratorDeviceType.CUDA,
          ],
          loss_atol=5e-1, loss_rtol=5e-1,  # TODO: Make tighter.
          grad_atol=5e-1, grad_rtol=5e-1,
          param_atol=5e-1, param_rtol=5e-1,
      ),
  ])
  def test_train_minimal_parity(
      self,
      model_name: str,
      config_file: str,
      skip_devices: list[device_type.AcceleratorDeviceType] | None = None,
      loss_atol: float = 1e-3,
      loss_rtol: float = 1e-3,
      grad_atol: float = 1e-3,
      grad_rtol: float = 1e-3,
      param_atol: float = 5e-3,
      param_rtol: float = 5e-3,
  ):
    """Verifies numerical equivalence between Device (TPU/GPU) and CPU."""
    if skip_devices and self.accelerator_device_type in skip_devices:
      self.skipTest(
          f"Skipping test for device type: {self.accelerator_device_type}")
      return

    config_manager = torchtitan.config.ConfigManager(
        tpu_job_config_module.TPUJobConfig)

    config = config_manager.parse_args([
        f"--model.name={model_name}",
        f"--job.config_file={config_file}",
        "--model.hf_assets_path=third_party/py/torchtitan/tests/assets/tokenizer",
        "--optimizer.implementation=foreach",
        "--training.dataset_path=third_party/py/torchtitan/tests/assets/c4_test",
        "--training.seq_len=128",
        "--training.dataset=c4_test",
        "--training.steps=3",
        # Ensure tests are for non-distributed training.
        "--parallelism.tensor_parallel_degree=1",
        "--parallelism.data_parallel_shard_degree=1",
    ])

    if model_name == "deepseek_v3":
      _setup_and_maybe_init_pg_for_single_process()
    self._run_trainer_device_parity_test(config,
        loss_atol=loss_atol,
        loss_rtol=loss_rtol,
        grad_atol=grad_atol,
        grad_rtol=grad_rtol,
        param_atol=param_atol,
        param_rtol=param_rtol,
    )

if __name__ == "__main__":
  absltest.main()
