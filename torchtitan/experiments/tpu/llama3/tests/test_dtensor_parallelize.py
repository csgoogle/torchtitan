"""Distributed tests for the Llama3 DTensor parallelization."""

from absl import logging
from absl.testing import absltest
from torch.distributed.device_mesh import DeviceMesh
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
)
import torch.nn as nn
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu.base_distributed_device_test import InputDistribution
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu import test_utils
from torchtitan.models.llama3.infra import parallelize as llama3_dtensor_parallelize
from torchtitan.models.llama3.model import model as llama3_model




# Constants for test parameters
# To use OVERRIDEABLE SDP on TPU, need to ensure local batch size is
# >= 4, i.e. TEST_BATCH_SIZE / world_size >= 4.
TEST_BATCH_SIZE = 32

# Need to make sure this is large enough for world size for sequence sharding.
# To use OVERRIDEABLE SDP on TPU, sequence length needs to be multiple of 512.
TEST_SEQ_LEN = 512
# If this is too small for worldsize/model, then the following assertion error
# is raised within the model code: freqs_cis.shape != (seqlen, x.shape[-1])
MAX_SEQ_LEN = 512

# Constants for training parameters
TEST_TRAINING_STEPS = 3
TEST_LR = 0.

# Constants for model parameters
# To use OVERRIDEABLE SDP on TPU, head dimension (which is dim / n_heads)
# should be either < 128 OR divisible by 128.
MODEL_DIM = 64
MODEL_N_HEADS = 8


def _get_llama3_model_args(
    vocab_size=128, dim=MODEL_DIM, n_layers=1, n_heads=MODEL_N_HEADS, multiple_of=16,
) -> llama3_model.TransformerModelArgs:
  """Returns model arguments for Llama3 model for testing."""
  return llama3_model.TransformerModelArgs(
      dim=dim,
      n_layers=n_layers,
      n_heads=n_heads,
      n_kv_heads=n_heads,
      vocab_size=vocab_size,
      multiple_of=multiple_of,
      max_seq_len=MAX_SEQ_LEN,
  )


def _verify_dtensor_llama3_tp_forward_worker(
    device: torch.device, rank: int, world_size: int
):
  """Verifies numerical equivalence of forward pass of Llama3 model with DTensor TP."""

  # Apply TP wrapper
  def apply_tp_wrapper(model):
    tp_mesh = DeviceMesh(device_type=device.type, mesh=torch.arange(world_size))
    llama3_dtensor_parallelize.apply_tp(
        model, tp_mesh, loss_parallel=False, enable_float8_tensorwise_tp=False
    )

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=llama3_model.Transformer,
      model_args=_get_llama3_model_args(),
      parallelism_func=apply_tp_wrapper,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=True,
  )

  runner.run_forward_parity(
      batch_size=TEST_BATCH_SIZE,
      seq_len=TEST_SEQ_LEN,
      atol=5e-2,
      rtol=5e-2
  )


def _verify_dtensor_llama3_tp_backward_worker(
    device: torch.device, rank: int, world_size: int
):
  """Worker function to run forward and backward pass on model after apply_tp is called."""

  # Apply TP wrapper
  def apply_tp_wrapper(model):
    tp_mesh = DeviceMesh(device_type=device.type, mesh=torch.arange(world_size))
    llama3_dtensor_parallelize.apply_tp(
        model, tp_mesh, loss_parallel=False, enable_float8_tensorwise_tp=False
    )

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=llama3_model.Transformer,
      model_args=_get_llama3_model_args(),
      parallelism_func=apply_tp_wrapper,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=True,
  )

  # Verify Loss and Gradients
  runner.run_backward_parity(
      num_steps=1,
      batch_size=TEST_BATCH_SIZE,
      seq_len=TEST_SEQ_LEN,
      atol_loss=3e-2,
      rtol_loss=3e-2,
      atol_grad=5e-3,
      rtol_grad=5e-3,
  )


def _verify_dtensor_llama3_fsdp_training_loop_worker(
    device: torch.device, rank: int, world_size: int
):
  """Worker function to verify FSDP numerical equivalence on Llama3 model."""

  # Apply FSDP wrapper
  def apply_fsdp_wrapper(model):
    dp_mesh = DeviceMesh(device_type=device.type, mesh=torch.arange(world_size))
    llama3_dtensor_parallelize.apply_fsdp(
        model,
        dp_mesh,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        pp_enabled=False,
        cpu_offload=False
    )

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=llama3_model.Transformer,
      model_args=_get_llama3_model_args(),
      parallelism_func=apply_fsdp_wrapper,
      input_distribution=InputDistribution.SPLIT_BATCH,  # FSDP = Split Batch
      use_meta_init=True,
  )

  # Verify loop (checks gradients on multiple random data batches)
  runner.run_backward_parity(
      num_steps=TEST_TRAINING_STEPS,
      batch_size=TEST_BATCH_SIZE,
      seq_len=TEST_SEQ_LEN,
      atol_loss=3,  # TODO(tbajpai): investigate tolerance issues b/495494788
      rtol_loss=3,
      atol_grad=9e-1,
      rtol_grad=9e-1,
  )


def _dtensor_issue_worker(device: torch.device, rank: int, world_size: int):
  """Worker function to test DTensor issue on a simple model."""
  class SimpleModel(nn.Module):

    def __init__(self):
      super().__init__()
      self.output = nn.Linear(in_features=64, out_features=128, bias=False)

    def forward(self, x):
      return self.output(x)

  model = SimpleModel()
  model = model.to(device)

  # Create the DeviceMesh for DTensor and apply TP
  tp_mesh = DeviceMesh(device_type=device.type, mesh=torch.arange(world_size))

  top_level_plan = {
      "output": ColwiseParallel(
          input_layouts=Shard(1),
          output_layouts=Replicate(),
          use_local_output=True,
      ),  # pytype: disable=wrong-arg-types
  }
  parallelize_module(model, tp_mesh, top_level_plan)

  torch.manual_seed(0)
  input_tensor = torch.randn(8, 64, 64, device=device)

  model.zero_grad()
  output_local = model(input_tensor)

  parallel_loss = torch.nn.MSELoss()(
      output_local, torch.randn_like(output_local)
  )
  parallel_loss.backward()


class Llama3DTensorParallelizeTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests the DTensor-based `apply_tp` in a distributed environment."""

  def test_dtensor_issue_distributed(self):
    logging.info(
        "Launching DTensor issue test with %d devices.",
        self.num_devices,
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_dtensor_issue_worker,
    )  # type: ignore
    logging.info("DTensor issue test finished.")

  # TP Tests are run with loss_parallel=False.
  def test_apply_tp_forward_equivalence_distributed(self):
    """Verifies numerical equivalence of the DTensor parallel model."""
    logging.info(
        "Launching DTensor TP forward numerical equivalence test with %d"
        " devices on Llama3 model.",
        self.num_devices,
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_dtensor_llama3_tp_forward_worker,
    )  # type: ignore
    logging.info(
        "Distributed DTensor TP forward numerical equivalence test finished."
    )

  def test_apply_tp_backward_equivalence_distributed(self):
    """Tests if the backward pass runs on model after apply_tp."""
    logging.info(
        "Launching DTensor TP backward numerical equivalence test with %d"
        " devices on Llama3 model.",
    )
    distributed.run_distributed(
        self.num_devices,
        self.accelerator_device_type,
        _verify_dtensor_llama3_tp_backward_worker,
    )
    logging.info(
        "Distributed DTensor backward numerical equivalence test finished."
    )

  def test_apply_fsdp_full_training_loop_equivalence_distributed(self):
    """Verifies numerical equivalence of a full training loop with FSDP."""
    logging.info(
        "Launching FSDP training equivalence test with %d devices on Llama3"
        " model.",
        self.num_devices,
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_dtensor_llama3_fsdp_training_loop_worker,
    )  # type: ignore
    logging.info("Distributed FSDP training equivalence test finished.")

if __name__ == "__main__":
  absltest.main()
