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


# TODO(tbajpai): Delete sharding logic tests once 'aten::equal' is implemented
# for TPU (for consistency and since its overkill).
def _assert_parallel_embedding(
    layer: fairscale_layers.ParallelEmbedding,
    orig_weight: torch.Tensor,
    vocab_size: int,
    embedding_dim: int,
    rank: int,
    world_size: int,
    device: torch.device,
    msg_prefix: str = "",
):
  """Asserts that a layer is a correctly sharded ParallelEmbedding."""
  assert isinstance(
      layer, fairscale_layers.ParallelEmbedding
  ), f"{msg_prefix}: Not a ParallelEmbedding"

  chunk_size = embedding_dim // world_size
  start_col = rank * chunk_size
  end_col = start_col + chunk_size
  expected_shape = (vocab_size, chunk_size)
  assert layer.weight.shape == expected_shape, (
      f"{msg_prefix}: Shape mismatch: Expected {expected_shape}, got"
      f" {layer.weight.shape}"
  )

  expected_weight = orig_weight[:, start_col:end_col].detach().clone()
  assert torch.equal(
      layer.weight.data, expected_weight.to(device)
  ), f"{msg_prefix}: ParallelEmbedding weights do not match"


def _assert_column_parallel(
    layer,
    orig_weight,
    orig_bias,
    out_features,
    in_features,
    rank,
    world_size,
    device,
    gather_output=None,
    msg_prefix="",
):
  """Asserts that a layer is a correctly sharded ColumnParallelLinear."""
  assert isinstance(
      layer, fairscale_layers.ColumnParallelLinear
  ), f"{msg_prefix}: Not a ColumnParallelLinear"
  if gather_output is not None:
    assert layer.gather_output == gather_output, (
        f"{msg_prefix}: gather_output mismatch: Expected {gather_output}, got"
        f" {layer.gather_output}"
    )

  chunk_size = out_features // world_size
  start_row = rank * chunk_size
  end_row = start_row + chunk_size
  expected_weight_shape = (chunk_size, in_features)
  assert layer.weight.shape == expected_weight_shape, (
      f"{msg_prefix}: Weight shape mismatch: Expected {expected_weight_shape},"
      f" got {layer.weight.shape}"
  )

  expected_weight = (
      orig_weight[start_row:end_row, :].detach().clone().to(device)
  )
  assert torch.equal(
      layer.weight.data, expected_weight
  ), f"{msg_prefix}: ColumnParallelLinear weights do not match"

  if orig_bias is not None:
    assert layer.bias is not None, f"{msg_prefix}: Expected bias, got None"

    expected_bias_shape = (chunk_size,)
    assert layer.bias.shape == expected_bias_shape, (
        f"{msg_prefix}: Bias shape mismatch: Expected {expected_bias_shape},"
        f" got {layer.bias.shape}"
    )

    expected_bias = orig_bias[start_row:end_row].detach().clone().to(device)
    assert torch.equal(
        layer.bias.data, expected_bias
    ), f"{msg_prefix}: ColumnParallelLinear biases do not match"

  else:
    assert (
        layer.bias is None
    ), f"{msg_prefix}: Expected no bias, got {layer.bias}"


def _assert_row_parallel(
    layer,
    orig_weight,
    orig_bias,
    out_features,
    in_features,
    rank,
    world_size,
    device,
    msg_prefix="",
):
  """Asserts that a layer is a correctly sharded RowParallelLinear."""
  assert isinstance(
      layer, fairscale_layers.RowParallelLinear
  ), f"{msg_prefix}: Not a RowParallelLinear"

  chunk_size = in_features // world_size
  start_col = rank * chunk_size
  end_col = start_col + chunk_size
  expected_weight_shape = (out_features, chunk_size)
  assert layer.weight.shape == expected_weight_shape, (
      f"{msg_prefix}: Weight shape mismatch: Expected {expected_weight_shape},"
      f" got {layer.weight.shape}"
  )

  expected_weight = (
      orig_weight[:, start_col:end_col].detach().clone().to(device)
  )
  assert torch.equal(
      layer.weight.data, expected_weight
  ), f"{msg_prefix}: RowParallelLinear weights do not match"

  if orig_bias is not None:
    assert layer.bias is not None, f"{msg_prefix}: Expected bias, got None"
    assert torch.equal(
        layer.bias.data, orig_bias.detach().clone().to(device)
    ), f"{msg_prefix}: RowParallelLinear biases do not match"
  else:
    assert (
        layer.bias is None
    ), f"{msg_prefix}: Expected no bias, got {layer.bias}"


def _run_assertions(
    model: llama3_model.Transformer,
    original_weights: Dict[str, torch.Tensor],
    rank: int,
    world_size: int,
    device: torch.device,
):
  """Contains all the sharding and weight assertions for Llama3 model."""

  args = _get_llama3_model_args()
  dim = args.dim
  vocab_size = args.vocab_size

  _assert_parallel_embedding(
      model.tok_embeddings,
      original_weights["tok_embeddings.weight"],
      vocab_size,
      dim,
      rank,
      world_size,
      device,
      msg_prefix="tok_embeddings",
  )
  _assert_column_parallel(
      model.output,
      original_weights["output.weight"],
      None,
      vocab_size,
      dim,
      rank,
      world_size,
      device,
      gather_output=True,
      msg_prefix="output",
  )
  for i, layer in enumerate(model.layers.values()):
    layer_prefix = f"layers.{i}"
    attn = layer.attention
    attn_prefix = f"{layer_prefix}.attention"
    _assert_column_parallel(
        attn.wq,
        original_weights[f"{attn_prefix}.wq.weight"],
        None,
        dim,
        dim,
        rank,
        world_size,
        device,
        gather_output=False,
        msg_prefix=f"{attn_prefix}.wq",
    )
    _assert_column_parallel(
        attn.wk,
        original_weights[f"{attn_prefix}.wk.weight"],
        None,
        dim,
        dim,
        rank,
        world_size,
        device,
        gather_output=False,
        msg_prefix=f"{attn_prefix}.wk",
    )
    _assert_column_parallel(
        attn.wv,
        original_weights[f"{attn_prefix}.wv.weight"],
        None,
        dim,
        dim,
        rank,
        world_size,
        device,
        gather_output=False,
        msg_prefix=f"{attn_prefix}.wv",
    )
    _assert_row_parallel(
        attn.wo,
        original_weights[f"{attn_prefix}.wo.weight"],
        None,
        dim,
        dim,
        rank,
        world_size,
        device,
        msg_prefix=f"{attn_prefix}.wo",
    )
    ffn = layer.feed_forward
    ffn_prefix = f"{layer_prefix}.feed_forward"
    hidden_dim = ffn.w1.weight.shape[0] * world_size
    _assert_column_parallel(
        ffn.w1,
        original_weights[f"{ffn_prefix}.w1.weight"],
        None,
        hidden_dim,
        dim,
        rank,
        world_size,
        device,
        gather_output=False,
        msg_prefix=f"{ffn_prefix}.w1",
    )
    _assert_column_parallel(
        ffn.w3,
        original_weights[f"{ffn_prefix}.w3.weight"],
        None,
        hidden_dim,
        dim,
        rank,
        world_size,
        device,
        gather_output=False,
        msg_prefix=f"{ffn_prefix}.w3",
    )
    _assert_row_parallel(
        ffn.w2,
        original_weights[f"{ffn_prefix}.w2.weight"],
        None,
        dim,
        hidden_dim,
        rank,
        world_size,
        device,
        msg_prefix=f"{ffn_prefix}.w2",
    )
    assert isinstance(
        layer.attention_norm, nn.RMSNorm
    ), f"{layer_prefix}.attention_norm is not RMSNorm"
    assert isinstance(
        layer.ffn_norm, nn.RMSNorm
    ), f"{layer_prefix}.ffn_norm is not RMSNorm"
  assert isinstance(model.norm, nn.RMSNorm), "model.norm is not RMSNorm"


def sharding_test_worker(device: torch.device, rank: int, world_size: int):
  """Main worker function to test sharding logic of apply_tp."""

  logging.info(
      "Worker started: Rank %d, Device %s, World Size %d",
      rank,
      device,
      world_size,
  )

  if not model_parallel.initialize.model_parallel_is_initialized():
    model_parallel.initialize.initialize_model_parallel(world_size)

  args = _get_llama3_model_args()
  torch.manual_seed(0)  # Ensure consistent init
  model = llama3_model.Transformer(args)
  model.init_weights()
  model = model.to(device)

  original_weights = {
      name: p.detach().clone() for name, p in model.named_parameters()
  }

  llama3_fairscale_parallelize.apply_tp(
      model,
      world_size=world_size,
      rank=rank)

  _run_assertions(model, original_weights, rank, world_size, device)
  logging.info("Rank %d: All assertions passed.", rank)

  model_parallel.initialize.destroy_model_parallel()


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

  runner.run_backward_parity(TEST_BATCH_SIZE, TEST_SEQ_LEN)

  model_parallel.initialize.destroy_model_parallel()


class Llama3FairscaleParallelizeTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests fairscale_parallelize.apply_tp in a distributed environment.

  This test verifies the tensor parallelism sharding logic of apply_tp
  on a Llama3 model across multiple devices, as well as the numerical
  equivalence of the parallel model after a forward and backward pass.
  """

  def test_apply_tp_sharding_distributed(self):
    logging.info(
        "Launching distributed test with %d devices on Llama3 model.",
        self.num_devices,
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=sharding_test_worker,
    )  # pytype: disable=wrong-arg-types
    logging.info("Distributed test run_distributed call finished.")

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
