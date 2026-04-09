"""Tests for the minimal distributed training setup (includes parity checks for generic minimal trainer and standard execution checks for custom trainers like redacted)."""

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_distributed_device_test




class TrainMinimalDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests train_minimal execution in distributed settings."""

  @parameterized.named_parameters([
      dict(
          testcase_name="redacted_ddp",
          model_name="redacted_tpu",
          config_file="torchtitan/experiments/tpu/redacted/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          data_parallel_replicate_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=None,
      ),
      dict(
          testcase_name="redacted_fsdp",
          model_name="redacted_tpu",
          config_file="torchtitan/experiments/tpu/redacted/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=None,
      ),
  ])
  def test_redacted_train_minimal(
      self,
      model_name,
      config_file,
      data_parallel_shard_degree,
      tensor_parallel_degree,
      data_parallel_replicate_degree=None,
      extra_args=None,
      skip_devices=None,
  ):
    """Runs execution test specifically for redacted's minimal trainer (experiments/tpu/redacted/train_minimal.py)."""
    from torchtitan.experiments.tpu.redacted import train_minimal as redacted_train_minimal

    config_args = [
        f"--model.name={model_name}",
        f"--job.config_file={config_file}",
        "--model.hf_assets_path=tests/assets/tokenizer",
        "--training.dataset_path=tests/assets/c4_test",
        "--training.seq_len=128",
        "--training.dataset=c4_test",
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
        start_trainer=redacted_train_minimal.start_trainer,
    )

  @parameterized.named_parameters([
      # deepseek_v3
      dict(
          testcase_name="deepseek_v3_tp",
          model_name="deepseek_v3",
          config_file="torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          skip_devices=[
              # TODO: b/482057603 - Issue with mixing DTensor and Tensor.
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
              device_type.AcceleratorDeviceType.TPU,
          ],
          loss_atol=7e-1,
          loss_rtol=5e-3,
          grad_atol=3e-1,
          grad_rtol=5e-3,
          param_atol=5e-3,
          param_rtol=5e-3,
      ),
      dict(
          testcase_name="deepseek_v3_fsdp",
          model_name="deepseek_v3",
          config_file="torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=[
              # TODO(abrauckmann): Error initializing torch.distributed
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
              # b/487028245 - Timing out w/Math Backend
              device_type.AcceleratorDeviceType.TPU,
          ],
          loss_atol=2e-1,
          loss_rtol=5e-3,
          grad_atol=2e-8,
          grad_rtol=5e-3,
          param_atol=5e-3,
          param_rtol=5e-3,
      ),
      # llama3
      dict(
          testcase_name="llama3_tp",
          model_name="llama3_tpu",
          config_file="torchtitan/models/llama3/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          loss_atol=1e2,
          loss_rtol=5e-3,
          grad_atol=1.5e1,
          grad_rtol=5e-2,
          param_atol=5e-2,
          param_rtol=5e-2,
      ),
      dict(
          testcase_name="llama3_fsdp",
          model_name="llama3_tpu",
          config_file="torchtitan/models/llama3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          # b/495494788: Trainer/parallelize differences causing large
          # numerical discrepancies.
          loss_atol=1e2,
          loss_rtol=5e-2,
          grad_atol=5e2,
          grad_rtol=5e-2,
          param_atol=5e-1,
          param_rtol=5e-2,
      ),
      # qwen3
      dict(
          testcase_name="qwen3_tp",
          model_name="qwen3_tpu",
          config_file="torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          loss_atol=1e2,
          loss_rtol=5e-3,
          grad_atol=1e1,
          grad_rtol=5e-2,
          param_atol=5e-2,
          param_rtol=5e-2,
      ),
      dict(
          testcase_name="qwen3_fsdp",
          model_name="qwen3_tpu",
          config_file="torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          # b/495494788: Trainer/parallelize differences causing large
          # numerical discrepancies.
          loss_atol=2000.0,
          loss_rtol=5e-2,
          grad_atol=10000.0,
          grad_rtol=5e-2,
          param_atol=5e1,
          param_rtol=5e-2,
      ),
  ])
  def test_train_minimal_distributed_parity(
      self,
      model_name,
      config_file,
      data_parallel_shard_degree,
      tensor_parallel_degree,
      skip_devices=None,
      loss_atol=5e-3, loss_rtol=5e-3,  # Default atol/rtols
      grad_atol=5e-3, grad_rtol=5e-3,
      param_atol=5e-3, param_rtol=5e-3,
      optimizer_implementation="foreach",
      enable_compile=False,
  ):
    """Runs a distributed parameter and gradient parity test against a CPU reference.

    This explicitly tests the generic minimal training script
    (experiments/tpu/train_minimal.py).
    This test executes a few training steps on a mock distributed topology (like
    FSDP),
    recording the loss, gradients, and parameter states. It then mathematically
    verifies
    that a standard single-device CPU model executing identical inputs achieves
    the perfectly equivalent numerical values, proving the distributed
    implementation
    is correct.
    """
    self._run_trainer_distributed_parity_test(
        config_args=[
            f"--model.name={model_name}",
            f"--job.config_file={config_file}",
            f"--optimizer.implementation={optimizer_implementation}",
            "--model.hf_assets_path=tests/assets/tokenizer",
            "--training.dataset_path=tests/assets/c4_test",
            "--training.seq_len=128",
            "--training.dataset=c4_test",
            "--training.steps=3",
            "--training.mixed_precision_param=float32",
            "--training.mixed_precision_reduce=float32",
            "--training.local_batch_size=4",
        ],
        tensor_parallel_degree=tensor_parallel_degree,
        data_parallel_shard_degree=data_parallel_shard_degree,
        skip_devices=skip_devices,
        loss_atol=loss_atol,
        loss_rtol=loss_rtol,
        grad_atol=grad_atol,
        grad_rtol=grad_rtol,
        param_atol=param_atol,
        param_rtol=param_rtol,
        enable_compile=enable_compile,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  absltest.main()
