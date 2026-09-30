"""Distributed tests for Gemma 4 DTensor/SPMD parallelization."""

import os

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchtitan.config
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.tpu import base_distributed_device_test
import torchtitan.experiments.tpu.gemma4  # trigger model registration
from torchtitan.experiments.tpu.gemma4.model import Gemma4Model
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
import torchtitan.experiments.tpu.tpu_job_config
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch
from torchtitan.tools import utils


def _setup_logger():
  if not dist.is_initialized() or dist.get_rank() == 0:
    logging.set_verbosity(logging.INFO)
  else:
    logging.set_verbosity(logging.ERROR)


def _run_gemma4_distributed(
    train_config: torchtitan.experiments.tpu.tpu_job_config.TPUTrainerConfig,
):
  use_cpu_safe_histc_patch()
  device = utils.get_local_device()
  world_size = int(os.environ.get("WORLD_SIZE", 1))
  if world_size > 1:
    dist_utils.init_distributed(
        train_config.comm,
        enable_cpu_backend=train_config.training.enable_cpu_offload,
        base_folder=train_config.dump_folder,
    )
  _setup_logger()

  assert train_config.model_spec is not None
  model_spec = train_config.model_spec
  model_args = model_spec.model
  assert isinstance(model_args, Gemma4Model.Config)
  if device.type == "cpu":
    for layer in model_args.layers:
      if layer.moe is not None and isinstance(
          layer.moe.experts, TpuGroupedExperts.Config
      ):
        layer.moe.experts.use_pallas_grouped_mm = False
  model_args.update_from_config(config=train_config)

  parallelism_config = train_config.parallelism
  parallel_dims = torchtitan.distributed.ParallelDims(
      dp_shard=parallelism_config.data_parallel_shard_degree,
      dp_replicate=parallelism_config.data_parallel_replicate_degree,
      cp=parallelism_config.context_parallel_degree,
      tp=parallelism_config.tensor_parallel_degree,
      pp=parallelism_config.pipeline_parallel_degree,
      ep=parallelism_config.expert_parallel_degree,
      world_size=world_size,
  )

  with (
      torch.device("meta"),
      utils.set_default_dtype(
          torchtitan.config.TORCH_DTYPE_MAP[train_config.training.dtype]
      ),
  ):
    model = model_args.build()

  model = model_spec.parallelize_fn(
      model,
      parallel_dims=parallel_dims,
      training=train_config.training,
      parallelism=train_config.parallelism,
      compile_config=train_config.compile,
      ac_config=train_config.activation_checkpoint,
      dump_folder=train_config.dump_folder,
  )
  model.to_empty(device=device)
  with torch.no_grad():
    model.init_weights(buffer_device=None)

  optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, fused=False)
  loss_fn = train_config.loss.build(compile_config=train_config.compile)
  if isinstance(loss_fn, torchtitan.components.loss.ChunkedLossWrapper):
    loss_fn.set_lm_head(model.lm_head)

  train_context = dist_utils.get_spmd_context(
      parallel_dims=parallel_dims, spmd_typechecking=False
  )

  n_steps = train_config.training.steps
  batch_size = train_config.training.local_batch_size
  seq_len = train_config.training.seq_len
  for _ in range(n_steps):
    tokens_cpu = torch.randint(
        0, model_args.vocab_size, (batch_size, seq_len), device="cpu"
    )
    tokens_device = tokens_cpu.to(device)
    optimizer.zero_grad()
    with train_context():  # pytype: disable=not-callable
      out_device = model(tokens_device)
      loss = loss_fn(out_device, tokens_device)
      if isinstance(loss, tuple):
        loss, _ = loss
      loss.backward()
    optimizer.step()


class TestDTensorParallelizeGemma4(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests parallelize_gemma4 with DTensor TP + FSDP on CPU and TPU."""

  @parameterized.named_parameters([
      dict(
          testcase_name="gemma4_dense_tp_dtensor",
          config="gemma4_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
      dict(
          testcase_name="gemma4_moe_tp_dtensor",
          config="gemma4_moe_testmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
  ])
  def test_model_distributed(
      self,
      config,
      data_parallel_shard_degree,
      tensor_parallel_degree,
  ):
    args = [
        "--module=torchtitan.experiments.tpu.gemma4",
        f"--config={config}",
        "--optimizer.implementation=foreach",
        "--hf_assets_path=third_party/py/torchtitan/tests/assets/tokenizer",
        "--dataloader.dataset_path=third_party/py/torchtitan/tests/assets/c4_test",
        "--dataloader.dataset=c4_test",
        "--training.seq_len=32",
        "--training.steps=2",
        "activation-checkpoint:none",
    ]
    self._test_train_distributed(
        args,
        data_parallel_shard_degree,
        tensor_parallel_degree,
        skip_devices=None,
        start_trainer=_run_gemma4_distributed,
        enable_compile=False,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  absltest.main()
