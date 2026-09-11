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

  # Tests only for Qwen3 Dense model for now
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
      # Same model, same everything, plus stacking -- so a failure here is the
      # stack and nothing else. Qwen3 rather than llama3 on purpose: the flag
      # reaches it through the shared `apply_simple_fsdp` with no
      # model-specific code, and this is what holds that true.
      dict(
          testcase_name="qwen3_simple_fsdp_stacked_layers",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--tpu_config.use_simple_fsdp",
              "--tpu_config.use_stacked_layers",
          ],
      ),
      # And the same again as a scan, so the pair isolates "runs as one loop"
      # from "stored as one tensor".
      dict(
          testcase_name="qwen3_simple_fsdp_scan_layers",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          args=[
              "--tpu_config.use_simple_fsdp",
              "--tpu_config.use_scan_layers",
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
      args: list[str] | None = None,
  ):
    default_args = [
        f"--module={module}",
        f"--config={config}",
        f"--optimizer.implementation={optimizer_implementation}",
        "--hf_assets_path=tests/assets/tokenizer",
        # 4 rows of 128 tokens per rank.
        "--training.max_context_length=128",
        "--training.num_tokens_per_microbatch_per_dp_rank=512",
        "--training.steps=3",
        "--training.mixed_precision_param=float32",
        "--training.mixed_precision_reduce=float32",
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
