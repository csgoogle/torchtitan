"""Distributed tests for Gemma 4 DTensor/SPMD parallelization."""

from functools import partial
import os

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
import torch.multiprocessing as mp
from torch import nn
import torchtitan.config
from torchtitan.config.configs import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu.base_distributed_device_test import (
    InputDistribution,
)
import torchtitan.experiments.tpu.gemma4 as gemma4_tpu
from torchtitan.experiments.tpu.gemma4.infra import (
    parallelize as gemma4_parallelize,
)
from torchtitan.experiments.tpu.gemma4.model import Gemma4Model
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
import torchtitan.experiments.tpu.tpu_job_config
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch
from torchtitan.tools import utils


TEST_BATCH_SIZE = 8
TEST_SEQ_LEN = 32
TEST_TRAINING_STEPS = 3


def _setup_logger():
  if not dist.is_initialized() or dist.get_rank() == 0:
    logging.set_verbosity(logging.INFO)
  else:
    logging.set_verbosity(logging.ERROR)


def _get_gemma4_model_config(
    flavor: str = "testmodel",
    *,
    enable_sp: bool = False,
    enable_ep: bool = False,
    disable_weight_tying: bool = False,
) -> Gemma4Model.Config:
  return gemma4_tpu._build_flavor_config(
      flavor,
      enable_sp=enable_sp,
      enable_ep=enable_ep,
      enable_weight_tying=not disable_weight_tying,
  )


class _Gemma4DistributedRunner(
    base_distributed_device_test.DistributedUnitTestRunner
):
  """DistributedUnitTestRunner that unshards DTensor outputs before MSELoss."""

  def _create_parallel_model(self, func):
    torch.manual_seed(42)
    super()._create_parallel_model(func)
    with torch.no_grad():
      for param in self.parallel_model.parameters():
        if isinstance(param, DTensor) and all(
            p.is_replicate() for p in param.placements
        ):
          torch.distributed.broadcast(param._local_tensor, src=0)

  def _run_step(
      self,
      model: nn.Module,
      inputs: torch.Tensor,
      targets: torch.Tensor = None,
      backward: bool = False,
  ):
    if backward:
      model.zero_grad()

    output = model(inputs)
    loss = None

    if backward and targets is not None:
      output_for_loss = (
          output.full_tensor() if isinstance(output, DTensor) else output
      )
      if targets.dtype != output_for_loss.dtype:
        targets = targets.to(output_for_loss.dtype)
      loss = nn.MSELoss()(output_for_loss, targets)
      loss.backward()

    return output, loss


def _verify_dtensor_gemma4_tp_forward_worker(
    device: torch.device,
    rank: int,
    world_size: int,
    flavor: str = "testmodel",
):
  """Verifies numerical equivalence of forward pass of Gemma4 model with DTensor TP."""
  use_cpu_safe_histc_patch()
  config = _get_gemma4_model_config(flavor, enable_sp=False, enable_ep=False)

  def apply_tp_wrapper(model):
    parallel_dims = ParallelDims(
        dp_shard=1,
        dp_replicate=1,
        cp=1,
        tp=world_size,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    return gemma4_parallelize.parallelize_gemma4(
        model,
        ac_config=SelectiveAC.Config(),
        compile_config=CompileConfig(),
        dump_folder="",
        parallel_dims=parallel_dims,
        parallelism=ParallelismConfig(
            tensor_parallel_degree=world_size,
            enable_sequence_parallel=False,
        ),
        training=TrainingConfig(),
    )

  runner = _Gemma4DistributedRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=Gemma4Model,
      model_args=config,
      parallelism_func=apply_tp_wrapper,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=True,
  )
  runner.run_forward_parity(
      TEST_BATCH_SIZE, TEST_SEQ_LEN, atol=1e-1, rtol=1e-1
  )


def _verify_dtensor_gemma4_tp_backward_worker(
    device: torch.device,
    rank: int,
    world_size: int,
    flavor: str = "testmodel",
):
  """Verifies numerical equivalence of forward + backward pass of Gemma4 with DTensor TP."""
  use_cpu_safe_histc_patch()
  config = _get_gemma4_model_config(flavor, enable_sp=False, enable_ep=False)

  def apply_tp_wrapper(model):
    parallel_dims = ParallelDims(
        dp_shard=1,
        dp_replicate=1,
        cp=1,
        tp=world_size,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    return gemma4_parallelize.parallelize_gemma4(
        model,
        ac_config=SelectiveAC.Config(),
        compile_config=CompileConfig(),
        dump_folder="",
        parallel_dims=parallel_dims,
        parallelism=ParallelismConfig(
            tensor_parallel_degree=world_size,
            enable_sequence_parallel=False,
        ),
        training=TrainingConfig(),
    )

  runner = _Gemma4DistributedRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=Gemma4Model,
      model_args=config,
      parallelism_func=apply_tp_wrapper,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=True,
  )
  runner.run_backward_parity(
      num_steps=1,
      batch_size=TEST_BATCH_SIZE,
      seq_len=TEST_SEQ_LEN,
      atol_loss=5e-2,
      rtol_loss=5e-2,
      atol_grad=1e-1,
      rtol_grad=1e-1,
  )


def _verify_dtensor_gemma4_fsdp_training_loop_worker(
    device: torch.device,
    rank: int,
    world_size: int,
    flavor: str = "testmodel",
    use_simple_fsdp: bool = False,
):
  """Verifies numerical equivalence of FSDP / SimpleFSDP multi-step loop on Gemma4."""
  use_cpu_safe_histc_patch()
  config = _get_gemma4_model_config(
      flavor,
      enable_sp=False,
      enable_ep=False,
      disable_weight_tying=use_simple_fsdp,
  )

  def apply_fsdp_wrapper(model):
    parallel_dims = ParallelDims(
        dp_shard=world_size,
        dp_replicate=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    return gemma4_parallelize.parallelize_gemma4(
        model,
        ac_config=SelectiveAC.Config(),
        compile_config=CompileConfig(),
        dump_folder="",
        parallel_dims=parallel_dims,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=world_size,
            use_simple_fsdp=use_simple_fsdp,
        ),
        training=TrainingConfig(
            mixed_precision_param="float32",
            mixed_precision_reduce="float32",
        ),
    )

  runner = _Gemma4DistributedRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=Gemma4Model,
      model_args=config,
      parallelism_func=apply_fsdp_wrapper,
      input_distribution=InputDistribution.SPLIT_BATCH,
      use_meta_init=not use_simple_fsdp,
  )
  runner.run_backward_parity(
      num_steps=TEST_TRAINING_STEPS,
      batch_size=TEST_BATCH_SIZE,
      seq_len=TEST_SEQ_LEN,
      atol_loss=3,
      rtol_loss=3,
      atol_grad=9e-1,
      rtol_grad=9e-1,
  )


def _verify_dtensor_gemma4_ep_fsdp_moe_worker(
    device: torch.device,
    rank: int,
    world_size: int,
):
  """Verifies numerical equivalence of Expert Parallelism (EP) + FSDP on Gemma4 MoE."""
  use_cpu_safe_histc_patch()
  config = _get_gemma4_model_config(
      "testmodel_moe",
      enable_sp=False,
      enable_ep=True,
  )

  def apply_ep_fsdp_wrapper(model):
    parallel_dims = ParallelDims(
        dp_shard=world_size,
        dp_replicate=1,
        cp=1,
        tp=1,
        pp=1,
        ep=world_size,
        world_size=world_size,
    )
    return gemma4_parallelize.parallelize_gemma4(
        model,
        ac_config=SelectiveAC.Config(),
        compile_config=CompileConfig(),
        dump_folder="",
        parallel_dims=parallel_dims,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=world_size,
            expert_parallel_degree=world_size,
            enable_sequence_parallel=False,
        ),
        training=TrainingConfig(
            mixed_precision_param="float32",
            mixed_precision_reduce="float32",
        ),
    )

  runner = _Gemma4DistributedRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=Gemma4Model,
      model_args=config,
      parallelism_func=apply_ep_fsdp_wrapper,
      input_distribution=InputDistribution.SPLIT_BATCH,
      use_meta_init=True,
  )
  runner.run_backward_parity(
      num_steps=1,
      batch_size=TEST_BATCH_SIZE,
      seq_len=TEST_SEQ_LEN,
      atol_loss=3,
      rtol_loss=3,
      atol_grad=9e-1,
      rtol_grad=9e-1,
  )


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
  """Tests parallelize_gemma4 with DTensor TP, FSDP, SimpleFSDP, and EP on CPU/TPU."""

  @parameterized.named_parameters(
      ("dense", "testmodel"),
      ("moe", "testmodel_moe"),
  )
  def test_apply_tp_forward_equivalence_distributed(self, flavor):
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=partial(_verify_dtensor_gemma4_tp_forward_worker, flavor=flavor),
    )

  @parameterized.named_parameters(
      ("dense", "testmodel"),
      ("moe", "testmodel_moe"),
  )
  def test_apply_tp_backward_equivalence_distributed(self, flavor):
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=partial(_verify_dtensor_gemma4_tp_backward_worker, flavor=flavor),
    )

  @parameterized.named_parameters(
      ("dense_fsdp", "testmodel", False),
      ("moe_fsdp", "testmodel_moe", False),
      ("dense_simple_fsdp", "testmodel", True),
      ("moe_simple_fsdp", "testmodel_moe", True),
  )
  def test_apply_fsdp_full_training_loop_equivalence_distributed(
      self, flavor, use_simple_fsdp
  ):
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=partial(
            _verify_dtensor_gemma4_fsdp_training_loop_worker,
            flavor=flavor,
            use_simple_fsdp=use_simple_fsdp,
        ),
    )

  def test_apply_ep_fsdp_moe_backward_equivalence_distributed(self):
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_dtensor_gemma4_ep_fsdp_moe_worker,
    )

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
        "--hf_assets_path=tests/assets/tokenizer",
        "--dataloader.dataset_path=tests/assets/c4_test",
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
