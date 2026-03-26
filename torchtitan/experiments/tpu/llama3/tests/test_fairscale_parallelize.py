"""Distributed tests for fairscale_parallelize.py.

Verifies the model parameter sharding logic of the apply_tp function
using the Llama3 model architecture. Also verifies numerical equivalence
of the parallel model after a forward and backward pass.
"""

from typing import Dict

from absl import logging
from absl.testing import absltest
from fairscale.nn import model_parallel
from fairscale.nn.model_parallel import layers as fairscale_layers
import torch
from torch import nn
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu.base_distributed_device_test import InputDistribution
from torchtitan.experiments.tpu.llama3.infra import fairscale_parallelize as llama3_fairscale_parallelize
from torchtitan.models.llama3.model import model as llama3_model

from internal.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


# Constants for test parameters
TEST_BATCH_SIZE = 4
TEST_SEQ_LEN = 64


def _get_llama3_model_args(
    vocab_size=128, dim=64, n_layers=2, n_heads=8, multiple_of=16,
) -> llama3_model.TransformerModelArgs:
  """Returns model arguments for Llama3 model for testing."""
  return llama3_model.TransformerModelArgs(
      dim=dim,
      n_layers=n_layers,
      n_heads=n_heads,
      n_kv_heads=n_heads,
      vocab_size=vocab_size,
      multiple_of=multiple_of,
      max_seq_len=256,  # Not directly used in sharding test
  )


def _verify_fairscale_llama3_tp_forward_worker(
    device: torch.device, rank: int, world_size: int
):
  """Verifies forward pass equivalence using DistributedUnitTestRunner."""
  if not model_parallel.initialize.model_parallel_is_initialized():
    model_parallel.initialize.initialize_model_parallel(world_size)

  def apply_tp_wrapper(model):
    # Ensure consistent init before splitting
    torch.manual_seed(0)
    model.init_weights()
    llama3_fairscale_parallelize.apply_tp(
        model, world_size=world_size, rank=rank
    )
    # Fairscale TP is in-place, implicitly returns None

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=llama3_model.Transformer,
      model_args=_get_llama3_model_args(),
      parallelism_func=apply_tp_wrapper,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=False,
      use_fairscale=True,
  )

  runner.run_forward_parity(TEST_BATCH_SIZE, TEST_SEQ_LEN, atol=5e-2, rtol=5e-2)

  model_parallel.initialize.destroy_model_parallel()


def _verify_fairscale_llama3_tp_backward_worker(
    device: torch.device, rank: int, world_size: int
):
  """Verifies backward pass equivalence using DistributedUnitTestRunner."""
  if not model_parallel.initialize.model_parallel_is_initialized():
    model_parallel.initialize.initialize_model_parallel(world_size)

  def apply_tp_wrapper(model):
    torch.manual_seed(0)
    model.init_weights()
    llama3_fairscale_parallelize.apply_tp(
        model, world_size=world_size, rank=rank
    )

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=llama3_model.Transformer,
      model_args=_get_llama3_model_args(),
      parallelism_func=apply_tp_wrapper,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=False,
      use_fairscale=True,
  )

  runner.run_backward_parity(
      num_steps=1, batch_size=TEST_BATCH_SIZE, seq_len=TEST_SEQ_LEN
  )

  model_parallel.initialize.destroy_model_parallel()


class Llama3FairscaleParallelizeTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):

  def test_apply_tp_forward_equivalence_distributed(self):
    """Launches test to verify numerical equivalence of forward and backward pass."""
    logging.info(
        "Launching numerical equivalence test with %d devices on Llama3 model.",
        self.num_devices,
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_fairscale_llama3_tp_forward_worker,
    )  # pytype: disable=wrong-arg-types
    logging.info("Distributed numerical equivalence test finished.")

  def test_apply_tp_backward_equivalence_distributed(self):
    """Launches test to verify numerical equivalence of forward and backward pass."""
    logging.info(
        "Launching numerical equivalence test with %d devices on Llama3 model.",
        self.num_devices,
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_fairscale_llama3_tp_backward_worker,
    )  # pytype: disable=wrong-arg-types
    logging.info("Distributed numerical equivalence test finished.")


if __name__ == "__main__":
  g3_multiprocessing.handle_test_main(absltest.main)
