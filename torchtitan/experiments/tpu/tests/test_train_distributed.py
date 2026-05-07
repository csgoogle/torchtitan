"""Tests for the minimal distributed training setup for Llama3 models."""

import os
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_distributed_device_test
import torchtitan.experiments.tpu.train




class TrainDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest):
  """Tests the distributed training setup."""

  @parameterized.named_parameters([
      # redacted_tpu
      dict(
          testcase_name="redacted_fsdp",
          model_name="redacted_tpu",
          config_file="torchtitan/experiments/tpu/redacted/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=[
              # b/510378879 - Hanging on Cloud VM.
              device_type.AcceleratorDeviceType.TPU,
          ],
      ),
      # redacted
      dict(
          testcase_name="redacted_ddp",
          model_name="redacted_tpu",
          config_file="torchtitan/experiments/tpu/redacted/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=1,
          data_parallel_replicate_degree=-1,
      ),
      dict(
          testcase_name="redacted_fsdp",
          model_name="redacted_tpu",
          config_file="torchtitan/experiments/tpu/redacted/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
      ),
      dict(
          testcase_name="redacted_fsdp_compile",
          model_name="redacted_tpu",
          config_file="torchtitan/experiments/tpu/redacted/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          enable_compile=True,
          skip_devices=[
              # b/501466433 - Bug with torch.compile
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      # deepseek_v3
      dict(
          testcase_name="deepseek_v3_tp",
          model_name="deepseek_v3",
          config_file="torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
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
          testcase_name="deepseek_v3_tp_compile",
          model_name="deepseek_v3",
          config_file="torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          enable_compile=True,
          skip_devices=[
              # TODO: b/482057603 - Issue with mixing DTensor and Tensor.
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="deepseek_v3_fsdp",
          model_name="deepseek_v3",
          config_file="torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=[
              # b/487028245 - Timing out w/Math Backend
              device_type.AcceleratorDeviceType.TPU,
          ],
      ),
      dict(
          testcase_name="deepseek_v3_fsdp_compile",
          model_name="deepseek_v3",
          config_file="torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          enable_compile=True,
          skip_devices=[
              # b/487028245 - Timing out w/Math Backend
              device_type.AcceleratorDeviceType.TPU,
              # b/501466433 - Bug with torch.compile
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      # flux
      dict(
          testcase_name="flux_tp",
          model_name="flux",
          config_file="torchtitan/experiments/tpu/flux/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          dataset_path="tests/assets/cc12m_test",
          dataset="cc12m-test",
          skip_devices=[
              # b/501467380 - Error during backward pass
              device_type.AcceleratorDeviceType.TPU,
          ]
      ),
      dict(
          testcase_name="flux_tp_compile",
          model_name="flux",
          config_file="torchtitan/experiments/tpu/flux/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          dataset_path="tests/assets/cc12m_test",
          dataset="cc12m-test",
          enable_compile=True,
          skip_devices=[
              # b/501466433 - Bug with torch.compile
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="flux_fsdp",
          model_name="flux",
          config_file="torchtitan/experiments/tpu/flux/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          dataset_path="tests/assets/cc12m_test",
          dataset="cc12m-test",
          skip_devices=[
              # b/501467380 - Error during backward pass
              device_type.AcceleratorDeviceType.TPU,
          ],
      ),
      dict(
          testcase_name="flux_fsdp_compile",
          model_name="flux",
          config_file="torchtitan/experiments/tpu/flux/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          dataset_path="tests/assets/cc12m_test",
          dataset="cc12m-test",
          enable_compile=True,
          skip_devices=[
              # b/501466433 - Bug with torch.compile
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      # llama3
      dict(
          testcase_name="llama3_tp",
          model_name="llama3_tpu",
          config_file="torchtitan/models/llama3/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
      dict(
          testcase_name="llama3_tp_compile",
          model_name="llama3_tpu",
          config_file="torchtitan/models/llama3/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          enable_compile=True,
          skip_devices=[
              # b/501466433 - Bug with torch.compile
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="llama3_fsdp",
          model_name="llama3_tpu",
          config_file="torchtitan/models/llama3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
      ),
      dict(
          testcase_name="llama3_fsdp_compile",
          model_name="llama3_tpu",
          config_file="torchtitan/models/llama3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          enable_compile=True,
          skip_devices=[
              # b/501466433 - Bug with torch.compile
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="llama3_fsdp_checkpoint",
          model_name="llama3_tpu",
          config_file="torchtitan/models/llama3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          checkpoint_enabled=True,
          skip_devices=[
              # TODO: remove once libtpu version is updated
              device_type.AcceleratorDeviceType.TPU,
          ],
      ),
      dict(
          testcase_name="llama3_fsdp_use_loss_kernel",
          model_name="llama3_tpu",
          config_file="torchtitan/models/llama3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--training.dtype=bfloat16",
              "--loss_kernel.use_loss_kernel",
              "--training.seq_len=1024",
              "--loss_kernel.loss_b_block_size=128",
              "--loss_kernel.loss_h_block_size=128",
              "--loss_kernel.loss_v_block_size=128",
          ],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="llama3_fsdp_use_splash_attention_kernel",
          model_name="llama3_tpu",
          config_file="torchtitan/models/llama3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--splash_attention_kernel.use_splash_attention_kernel"],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      # qwen3
      dict(
          testcase_name="qwen3_tp",
          model_name="qwen3_tpu",
          config_file="torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
      dict(
          testcase_name="qwen3_tp_compile",
          model_name="qwen3_tpu",
          config_file="torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          enable_compile=True,
          skip_devices=[
              # b/501466433 - Bug with torch.compile
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="qwen3_fsdp",
          model_name="qwen3_tpu",
          config_file="torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
      ),
      dict(
          testcase_name="qwen3_fsdp_compile",
          model_name="qwen3_tpu",
          config_file="torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          enable_compile=True,
          skip_devices=[
              # b/501466433 - Bug with torch.compile
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="qwen3_fsdp_use_loss_kernel",
          model_name="qwen3_tpu",
          config_file="torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--training.dtype=bfloat16",
              "--loss_kernel.use_loss_kernel",
              "--training.seq_len=1024",
              "--loss_kernel.loss_b_block_size=128",
              "--loss_kernel.loss_h_block_size=128",
              "--loss_kernel.loss_v_block_size=128",
          ],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="qwen3_fsdp_use_splash_attention_kernel",
          model_name="qwen3_tpu",
          config_file="torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--splash_attention_kernel.use_splash_attention_kernel"],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="qwen3_fsdp_moe_use_gmm_kernel",
          model_name="qwen3_tpu",
          config_file="torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--model.flavor=testmodel_moe",  # NOTE: this model has use_grouped_mm=True
              "--qwen3.use_gmm_kernel",
          ],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="qwen3_fsdp_moe_use_gmm_and_fill_indices_kernels",
          model_name="qwen3_tpu",
          config_file="torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--model.flavor=testmodel_moe",  # NOTE: this model has use_grouped_mm=True
              "--qwen3.use_gmm_kernel",
              "--qwen3.use_fill_indices_kernel",
          ],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
  ])
  def test_train_distributed(
      self,
      model_name,
      config_file,
      data_parallel_shard_degree,
      tensor_parallel_degree,
      data_parallel_replicate_degree=None,
      skip_devices=None,
      optimizer_implementation="foreach",
      enable_compile=False,
      checkpoint_enabled=False,
      dataset_path="tests/assets/c4_test",
      dataset="c4_test",
      args: list[str] | None = None,
  ):
    default_args = [
        f"--model.name={model_name}",
        f"--job.config_file={config_file}",
        f"--optimizer.implementation={optimizer_implementation}",
        f"--training.dataset_path={dataset_path}",
        f"--training.dataset={dataset}",
        "--model.hf_assets_path=tests/assets/tokenizer",
        "--training.seq_len=128",
        "--training.steps=3",
        "--training.mixed_precision_param=float32",
        "--training.mixed_precision_reduce=float32",
        f"--training.local_batch_size={1 if model_name == 'flux' else 4}",
    ]
    if not args:
      args = []

    # Combined args: default args first, then test case specific args.
    # This allows test cases to override default arguments.
    combined_args = default_args + args

    if model_name == "flux":
      combined_args.extend([
          "--encoder.t5_encoder=tests/assets/flux_test_encoders/t5-micro",
          "--encoder.clip_encoder=tests/assets/flux_test_encoders/clip-micro",
      ])
    if checkpoint_enabled:
      test_tmp_dir = tempfile.mkdtemp()
      self.addCleanup(shutil.rmtree, test_tmp_dir, ignore_errors=True)
      combined_args.extend([
          "--checkpoint.enable",
          "--checkpoint.interval=2",
          f"--checkpoint.folder={os.path.join(test_tmp_dir, 'checkpoint')}",
      ])

    self._test_train_distributed(
        config_args=combined_args,
        data_parallel_shard_degree=data_parallel_shard_degree,
        tensor_parallel_degree=tensor_parallel_degree,
        data_parallel_replicate_degree=data_parallel_replicate_degree,
        skip_devices=skip_devices,
        start_trainer=torchtitan.experiments.tpu.train.start_trainer,
        enable_compile=enable_compile,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  absltest.main()
