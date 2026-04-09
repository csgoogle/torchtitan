"""Distributed tests for the Qwen3 DTensor TP."""

from absl import logging
from absl.testing import absltest
from torch.distributed.device_mesh import DeviceMesh
import torch
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu.base_distributed_device_test import InputDistribution
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu import test_utils

from torchtitan.models.qwen3.infra import parallelize as qwen3_dtensor_parallelize

from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from torchtitan.models.qwen3.model import model as qwen3_model




# Constants for test parameters
# To use OVERRIDEABLE SDP on TPU, need to ensure local batch size is
# >= 4, i.e. TEST_BATCH_SIZE / world_size >= 4.
TEST_BATCH_SIZE = 32

# Need to make sure this is large enough for world size for sequence sharding.
# To use OVERRIDEABLE SDP on TPU, sequence length needs to be multiple of 512.
TEST_SEQ_LEN = 512
# If this is too small for worldsize/model, then the following assertion error
# is raised within the model code: rope_cache.shape != (seqlen, head_dim * 2)
MAX_SEQ_LEN = 512

# Constants for training parameters
TEST_TRAINING_STEPS = 3
TEST_LR = 0.01

# Constants for model parameters
# To use OVERRIDEABLE SDP on TPU, head dimension (which is dim / n_heads)
# should be either < 128 OR divisible by 128.
MODEL_DIM = 128
MODEL_N_HEADS = 8


def _get_qwen3_model_args(
    vocab_size=128, dim=MODEL_DIM, n_layers=2, n_heads=MODEL_N_HEADS, hidden_dim=128
) -> Qwen3ModelArgs:
  """Returns model arguments for Qwen3 model for testing."""
  return Qwen3ModelArgs(
      dim=dim,
      n_layers=n_layers,
      n_heads=n_heads,
      n_kv_heads=n_heads,
      vocab_size=vocab_size,
      hidden_dim=hidden_dim,
      max_seq_len=MAX_SEQ_LEN,
      moe_enabled=False,  # Ensure we are testing a dense model
  )


def _verify_dtensor_qwen3_non_moe_tp_forward_worker(
    device: torch.device, rank: int, world_size: int
):
  """Verifies numerical equivalence of forward pass of Qwen3 model with DTensor TP."""

  def apply_tp_wrapper(model):
    tp_mesh = DeviceMesh(device_type=device.type, mesh=torch.arange(world_size))
    qwen3_dtensor_parallelize.apply_non_moe_tp(
        model, tp_mesh,
        loss_parallel=False,
        enable_float8_tensorwise_tp=False,
        enable_async_tp=False,
        cp_enabled=False,
    )

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=qwen3_model.Qwen3Model,
      model_args=_get_qwen3_model_args(),
      parallelism_func=apply_tp_wrapper,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=True,
  )

  runner.run_forward_parity(TEST_BATCH_SIZE, TEST_SEQ_LEN, atol=5e-1, rtol=5e-1)


def _verify_dtensor_qwen3_tp_backward_worker(
    device: torch.device, rank: int, world_size: int
):
  """Worker function to run forward andbackward pass on model after apply_tp is called."""

  def apply_tp_wrapper(model):
    tp_mesh = DeviceMesh(device_type=device.type, mesh=torch.arange(world_size))
    qwen3_dtensor_parallelize.apply_non_moe_tp(
        model,
        tp_mesh,
        loss_parallel=False,
        enable_float8_tensorwise_tp=False,
        enable_async_tp=False,
        cp_enabled=False,
    )

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=qwen3_model.Qwen3Model,
      model_args=_get_qwen3_model_args(),
      parallelism_func=apply_tp_wrapper,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=True,
  )
  runner.run_backward_parity(
      num_steps=1, batch_size=TEST_BATCH_SIZE, seq_len=TEST_SEQ_LEN
  )


def _verify_dtensor_qwen3_fsdp_training_loop_worker(
    device: torch.device,
    rank: int,
    world_size: int,
):
  """Worker function to call the FSDP numerical equivalence helper on Qwen3 model."""

  def apply_fsdp_wrapper(model):
    dp_mesh = DeviceMesh(device_type=device.type, mesh=torch.arange(world_size))
    qwen3_dtensor_parallelize.apply_fsdp(
        model,
        dp_mesh,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        pp_enabled=False,
        cpu_offload=False,
        reshard_after_forward_policy="default",
    )

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=qwen3_model.Qwen3Model,
      model_args=_get_qwen3_model_args(),
      parallelism_func=apply_fsdp_wrapper,
      input_distribution=InputDistribution.SPLIT_BATCH,
      use_meta_init=True,
  )

  runner.run_backward_parity(
      num_steps=TEST_TRAINING_STEPS,
      batch_size=TEST_BATCH_SIZE,
      seq_len=TEST_SEQ_LEN,
      atol_loss=3,  # TODO(tbajpai): investigate tolerance issues b/495494788
      rtol_loss=3,
      atol_grad=9e-1,
      rtol_grad=9e-1,
  )


class Qwen3DTensorParallelizeTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests the DTensor-based `apply_non_moe_tp` in a distributed environment."""

  def test_apply_non_moe_tp_forward_equivalence_distributed(self):
    """Launches test to verify numerical equivalence of the DTensor parallel model."""
    logging.info(
        "Launching DTensor TP forward numerical equivalence test with %d"
        " devices on Qwen3 model.",
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_dtensor_qwen3_non_moe_tp_forward_worker,
    )  # pytype: disable=wrong-arg-types
    logging.info(
        "Distributed DTensor TP forward numerical equivalence test finished."
    )

  def test_apply_non_moe_tp_backward_equivalence_distributed(self):
    """Tests if the backward pass runs on model after apply_tp."""
    logging.info(
        "Launching DTensor TP backward numerical equivalence test with %d"
        " devices on Qwen3 model.",
    )
    distributed.run_distributed(
        self.num_devices,
        self.accelerator_device_type,
        _verify_dtensor_qwen3_tp_backward_worker,
    )
    logging.info(
        "Distributed DTensor TP backward numerical equivalence test finished."
    )

  def test_apply_fsdp_full_training_loop_equivalence_distributed(self):
    """Verifies numerical equivalence of a full training loop with FSDP."""
    logging.info(
        "Launching FSDP training equivalence test with %d devices on Qwen3"
        " model.",
        self.num_devices,
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_dtensor_qwen3_fsdp_training_loop_worker,
    )  # pytype: disable=wrong-arg-types
    logging.info("Distributed FSDP training equivalence test finished.")


if __name__ == "__main__":
  absltest.main()
