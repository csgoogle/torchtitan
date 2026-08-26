"""Tests for the minimal distributed training setup under TpuGraphTrainer."""

import os
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_distributed_device_test
import torchtitan.experiments.tpu.train



class TpuGraphTrainerDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests the distributed training setup under TpuGraphTrainer."""

  @parameterized.named_parameters([
      dict(
          testcase_name="qwen3_simple_fsdp",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--tpu_config.use_simple_fsdp",
          ],
      ),
      # The 70B target configuration, on a debug model and the devices of one
      # host. Every part of it is covered somewhere already -- the scan lowers
      # to one `stablehlo.while` (`tests/test_scan_lowering.py`), the loop
      # survives XLA (`tests/test_scan_compiled_hlo.py`), the in-body gather
      # lands inside the body (`tests/test_scan_fsdp_lowering.py`), the
      # compiled optimizer steps on device
      # (`compiled_optimizers/tests/test_functional.py`) -- each in isolation
      # and most of them structurally. Nothing ran the four together, which is
      # the only form the target actually uses, so until now the first check
      # that the combination trains at all was a 70B job on four slices.
      dict(
          testcase_name="llama3_scan_compiled_optimizer",
          module="torchtitan.experiments.tpu.llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              # Both spellings: the graph-trainer lane reads
              # `parallelism.use_simple_fsdp` (scan's in-body gather is a
              # simple_fsdp parametrization plan) while the TPU trainer reads
              # its own. `llama3_70b` sets the first from the config function;
              # the debug config does not, so the target's shape has to be
              # asked for here.
              "--parallelism.use_simple_fsdp",
              "--tpu_config.use_simple_fsdp",
              "--tpu_config.use_scan",
              "--tpu_config.compile_optimizer",
              "--tpu_config.graph_trainer_memory_policy=full",
          ],
      ),
  ])
  def test_graph_trainer_distributed(
      self,
      module,
      config,
      data_parallel_shard_degree,
      tensor_parallel_degree,
      data_parallel_replicate_degree=None,
      skip_devices=None,
      optimizer_implementation="foreach",
      enable_compile=True,
      dataset_path="tests/assets/c4_test",
      dataset="c4_test",
      args: list[str] | None = None,
  ):
    default_args = [
        f"--module={module}",
        f"--config={config}",
        f"--optimizer.implementation={optimizer_implementation}",
        f"--dataloader.dataset_path={dataset_path}",
        f"--dataloader.dataset={dataset}",
        "--hf_assets_path=tests/assets/tokenizer",
        "--training.seq_len=128",
        "--training.steps=3",
        "--training.mixed_precision_param=float32",
        "--training.mixed_precision_reduce=float32",
        "--training.local_batch_size=4",
        "--splash_attention_kernel.use_splash_attention_kernel",
        "--tpu_config.use_graph_trainer",
        "--tpu_config.compile_mode=aot_fx_trace",
    ]
    if not args:
      args = []

    combined_args = default_args + args

    self._test_train_distributed(
        config_args=combined_args,
        data_parallel_shard_degree=data_parallel_shard_degree,
        tensor_parallel_degree=tensor_parallel_degree,
        data_parallel_replicate_degree=data_parallel_replicate_degree,
        skip_devices=skip_devices,
        start_trainer=torchtitan.experiments.tpu.train.start_trainer,  # pyrefly: ignore[bad-argument-type]
        enable_compile=enable_compile,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
