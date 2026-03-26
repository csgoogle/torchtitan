"""Distributed tests for the Qwen3 Fairscale TP."""

from absl import logging
from absl.testing import absltest
from fairscale.nn import model_parallel
import torch
import torch.distributed as dist
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu.base_distributed_device_test import InputDistribution
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu.qwen3.infra import fairscale_parallelize as qwen3_fairscale_parallelize
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from torchtitan.models.qwen3.model import model as qwen3_model

from internal.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


# Constants for test parameters
TEST_BATCH_SIZE = 8
TEST_SEQ_LEN = 64

# Constants for training parameters
TEST_TRAINING_STEPS = 3
TEST_LR = 0.01


def _get_qwen3_model_args(
    vocab_size=128, dim=64, n_layers=2, n_heads=8, hidden_dim=128
) -> Qwen3ModelArgs:
  """Returns model arguments for Qwen3 model for testing."""
  return Qwen3ModelArgs(
      dim=dim,
      n_layers=n_layers,
      n_heads=n_heads,
      n_kv_heads=n_heads,
      vocab_size=vocab_size,
      hidden_dim=hidden_dim,
      max_seq_len=256,
      moe_enabled=False,  # Ensure we are testing a dense model
  )


def _verify_fairscale_qwen3_non_moe_tp_forward_worker(
    device: torch.device, rank: int, world_size: int
):
  """Verifies numerical equivalence of forward pass of Qwen3 model with TP."""
  logging.info(
      "Equivalence worker started: Rank %d, Device %s, World Size %d",
      rank,
      device,
      world_size,
  )
  if not model_parallel.initialize.model_parallel_is_initialized():
    model_parallel.initialize.initialize_model_parallel(world_size)

  def apply_tp_wrapper(model):
    # Ensure consistent init before splitting so it matches Reference
    torch.manual_seed(0)
    model.init_weights()
    qwen3_fairscale_parallelize.apply_non_moe_tp(
        model, world_size=world_size, rank=rank
    )

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=qwen3_model.Qwen3Model,
      model_args=_get_qwen3_model_args(),
      input_distribution=InputDistribution.REPLICATE,
      parallelism_func=apply_tp_wrapper,
      use_meta_init=False,
      use_fairscale=True,
  )

  runner.run_forward_parity(TEST_BATCH_SIZE, TEST_SEQ_LEN, atol=5e-2, rtol=5e-2)

  model_parallel.initialize.destroy_model_parallel()


def _verify_fairscale_qwen3_fsdp_training_loop_worker(
    device: torch.device, rank: int, world_size: int
):
  """Verifies training loop equivalence for Qwen3 Fairscale FSDP."""
  logging.info(
      "FSDP Equivalence worker started: Rank %d, Device %s, World Size %d",
      rank,
      device,
      world_size,
  )

  def apply_fsdp_wrapper(model):
    torch.manual_seed(0)
    model.init_weights()
    dp_group = dist.group.WORLD
    qwen3_fairscale_parallelize.apply_fsdp(
        model,
        dp_group,
        pp_enabled=False,
        cpu_offload=False,
    )

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=qwen3_model.Qwen3Model,
      model_args=_get_qwen3_model_args(),
      input_distribution=InputDistribution.SPLIT_BATCH,
      parallelism_func=apply_fsdp_wrapper,
      use_meta_init=False,
      use_fairscale=True,
  )

  runner.run_backward_parity(
      num_steps=TEST_TRAINING_STEPS,
      batch_size=TEST_BATCH_SIZE,
      seq_len=TEST_SEQ_LEN,
      atol_loss=5e-3,
      rtol_loss=5e-3,
  )


class Qwen3FairscaleParallelizeTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests fairscale_parallelize_qwen3 in a distributed environment."""

  def test_apply_non_moe_tp_forward_equivalence_distributed(self):
    """Launches test to verify numerical equivalence."""
    logging.info(
        "Launching numerical equivalence test with %d devices on Qwen3 model.",
        self.num_devices,
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_fairscale_qwen3_non_moe_tp_forward_worker,
    )  # pytype: disable=wrong-arg-types
    logging.info("Distributed numerical equivalence test finished.")

  @test_utils.skip_if_tpu(
      reason="FairScale FSDP is built specifically for the CUDA backend",
  )
  @test_utils.skip_if_cpu(
      reason="FairScale FSDP is built specifically for the CUDA backend",
  )
  def test_apply_fsdp_numerical_equivalence_distributed_gpu_only(self):
    """Verifies numerical equivalence of a full training loop with FSDP."""
    logging.info(
        "Launching FSDP numerical equivalence test with %d devices on Qwen3"
        " model.",
        self.num_devices,
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_fairscale_qwen3_fsdp_training_loop_worker,
    )  # pytype: disable=wrong-arg-types
    logging.info("Distributed FSDP numerical equivalence test finished.")

if __name__ == "__main__":
  g3_multiprocessing.handle_test_main(absltest.main)
