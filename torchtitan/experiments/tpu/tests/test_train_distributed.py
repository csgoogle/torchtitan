"""Tests for the minimal distributed training setup for Llama3 models."""

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_distributed_device_test
import torchtitan.experiments.tpu.train

from internal.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


class TrainDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest):
  """Tests the distributed training setup."""

  @parameterized.named_parameters([
      # llama3_tpu (with local fairscale implementation)
      dict(
          testcase_name="llama3_tpu_tp_fairscale",
          model_name="llama3_tpu",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
          use_fairscale=True,
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
      dict(
          testcase_name="llama3_tpu_fsdp_use_splash_attention_kernel",
          model_name="llama3_tpu",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--tpu_config.use_splash_attention_kernel"],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="llama3_tpu_fsdp_use_loss_kernel",
          model_name="llama3_tpu",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--training.dtype=bfloat16",
              "--tpu_config.use_loss_kernel",
              "--training.seq_len=1024"
          ],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      # qwen3_tpu (with local fairscale implementation)
      dict(
          testcase_name="qwen3_tpu_tp_fairscale",
          model_name="qwen3_tpu",
          config_file="third_party/py/torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          use_fairscale=True,
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          skip_devices=[
              # b/480969610 - Issue with matmul shapes
              device_type.AcceleratorDeviceType.TPU,
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="qwen3_tpu_fsdp_use_splash_attention_kernel",
          model_name="qwen3_tpu",
          config_file="third_party/py/torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=["--tpu_config.use_splash_attention_kernel"],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      dict(
          testcase_name="qwen3_tpu_fsdp_use_loss_kernel",
          model_name="qwen3_tpu",
          config_file="third_party/py/torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--training.dtype=bfloat16",
              "--tpu_config.use_loss_kernel",
              "--training.seq_len=1024"
          ],
          skip_devices=[
              device_type.AcceleratorDeviceType.CPU,
              device_type.AcceleratorDeviceType.CUDA,
          ],
      ),
      # llama3 (with TT dtensor implementation)
      dict(
          testcase_name="llama3_tp_dtensor",
          model_name="llama3",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
      dict(
          testcase_name="llama3_tp_dtensor_compile",
          model_name="llama3",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          enable_compile=True,
      ),
      dict(
          testcase_name="llama3_fsdp_dtensor",
          model_name="llama3",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
      ),
      dict(
          testcase_name="llama3_fsdp_dtensor_compile",
          model_name="llama3",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          enable_compile=True,
      ),
      dict(
          testcase_name="llama3_fsdp_dtensor_checkpoint",
          model_name="llama3",
          config_file="third_party/py/torchtitan/models/llama3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          checkpoint_enabled=True,
          skip_devices=[
              # b/489217116 - DCP not supported due to gather() unimplemented.
              device_type.AcceleratorDeviceType.TPU,
          ],
      ),
      # qwen3 (with TT dtensor implementation)
      dict(
          testcase_name="qwen3_tp_dtensor",
          model_name="qwen3",
          config_file="third_party/py/torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
      dict(
          testcase_name="qwen3_tp_dtensor_compile",
          model_name="qwen3",
          config_file="third_party/py/torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          enable_compile=True,
      ),
      dict(
          testcase_name="qwen3_fsdp_dtensor",
          model_name="qwen3",
          config_file="third_party/py/torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
      ),
      dict(
          testcase_name="qwen3_fsdp_dtensor_compile",
          model_name="qwen3",
          config_file="third_party/py/torchtitan/experiments/tpu/qwen3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          enable_compile=True,
      ),
      dict(
          testcase_name="deepseek_v3_tp_dtensor",
          model_name="deepseek_v3",
          config_file="third_party/py/torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
          use_fairscale=False,
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
          testcase_name="deepseek_v3_tp_dtensor_compile",
          model_name="deepseek_v3",
          config_file="third_party/py/torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
          use_fairscale=False,
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
          testcase_name="deepseek_v3_fsdp_dtensor",
          model_name="deepseek_v3",
          config_file="third_party/py/torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=[
              # b/487028245 - Timing out w/Math Backend
              device_type.AcceleratorDeviceType.TPU,
          ],
      ),
      dict(
          testcase_name="deepseek_v3_fsdp_dtensor_compile",
          model_name="deepseek_v3",
          config_file="third_party/py/torchtitan/experiments/tpu/deepseek_v3/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          enable_compile=True,
          skip_devices=[
              # b/487028245 - Timing out w/Math Backend
              device_type.AcceleratorDeviceType.TPU,
          ],
      ),
      dict(
          testcase_name="flux_tp_dtensor",
          model_name="flux",
          config_file="third_party/py/torchtitan/experiments/tpu/flux/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          dataset_path="third_party/py/torchtitan/tests/assets/cc12m_test",
          dataset="cc12m-test",
      ),
      dict(
          testcase_name="flux_tp_dtensor_compile",
          model_name="flux",
          config_file="third_party/py/torchtitan/experiments/tpu/flux/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          dataset_path="third_party/py/torchtitan/tests/assets/cc12m_test",
          dataset="cc12m-test",
          enable_compile=True,
      ),
      dict(
          testcase_name="flux_fsdp_dtensor",
          model_name="flux",
          config_file="third_party/py/torchtitan/experiments/tpu/flux/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          dataset_path="third_party/py/torchtitan/tests/assets/cc12m_test",
          dataset="cc12m-test",
      ),
      dict(
          testcase_name="flux_fsdp_dtensor_compile",
          model_name="flux",
          config_file="third_party/py/torchtitan/experiments/tpu/flux/train_configs/debug_model.toml",
          use_fairscale=False,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          dataset_path="third_party/py/torchtitan/tests/assets/cc12m_test",
          dataset="cc12m-test",
          enable_compile=True,
      ),
  ])
  def test_train_distributed(
      self,
      model_name,
      config_file,
      use_fairscale,
      data_parallel_shard_degree,
      tensor_parallel_degree,
      skip_devices=None,
      optimizer_implementation="foreach",
      enable_compile=False,
      checkpoint_enabled=False,
      dataset_path="third_party/py/torchtitan/tests/assets/c4_test",
      dataset="c4_test",
      args: list[str] | None = None,
  ):
    default_args = [
        f"--model.name={model_name}",
        f"--job.config_file={config_file}",
        f"--optimizer.implementation={optimizer_implementation}",
        f"--training.dataset_path={dataset_path}",
        f"--training.dataset={dataset}",
        "--model.hf_assets_path=third_party/py/torchtitan/tests/assets/tokenizer",
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
          "--encoder.t5_encoder=third_party/py/torchtitan/tests/assets/flux_test_encoders/t5-micro",
          "--encoder.clip_encoder=third_party/py/torchtitan/tests/assets/flux_test_encoders/clip-micro",
      ])
    if checkpoint_enabled:
      combined_args.extend([
          "--checkpoint.enable",
          "--checkpoint.interval=2",
      ])

    self._test_train_distributed(
        combined_args,
        use_fairscale,
        data_parallel_shard_degree,
        tensor_parallel_degree,
        skip_devices,
        torchtitan.experiments.tpu.train.start_trainer,
        run_init_process_group=False,
        enable_compile=enable_compile,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
