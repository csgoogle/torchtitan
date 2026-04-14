"""Distributed tests for the DeepSeekV3 DTensor TP."""

from absl import logging
from absl.testing import absltest
from torch.distributed.device_mesh import DeviceMesh
import torch
import torchtitan.models.moe
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu.base_distributed_device_test import InputDistribution
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu.deepseek_v3.model.args import DeepSeekV3ModelArgs
from torchtitan.models.deepseek_v3.infra import parallelize as deepseek_v3_dtensor_parallelize
from torchtitan.models.deepseek_v3.model import model as deepseek_v3_model


from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch


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


def _get_deepseek_v3_model_args(
    vocab_size=128, dim=MODEL_DIM, n_layers=2, n_heads=MODEL_N_HEADS
) -> DeepSeekV3ModelArgs:
  """Returns model arguments for DeepSeekV3 model for testing."""
  args = DeepSeekV3ModelArgs(
      dim=dim,
      n_layers=n_layers,
      n_dense_layers=1,
      n_heads=n_heads,
      vocab_size=vocab_size,
      max_seq_len=MAX_SEQ_LEN,
      inter_dim=128,
      moe_inter_dim=64,
      moe_args=torchtitan.models.moe.MoEArgs(
          num_experts=4,
          top_k=2,
          use_grouped_mm=False,
      ),
      q_lora_rank=0,  # For testing dense q_proj
      kv_lora_rank=32,
      qk_nope_head_dim=8,
      qk_rope_head_dim=8,
      v_head_dim=16,
  )
  args.moe_impl = "standard"
  return args


def _verify_dtensor_deepseek_v3_non_moe_tp_forward_worker(
    device: torch.device, rank: int, world_size: int
):
  """Verifies numerical equivalence of forward pass of DeepSeekV3 model with DTensor TP."""
  # Apply histc CPU workaround in worker process. Doesn't propagate from parent.
  use_cpu_safe_histc_patch()

  def apply_tp_wrapper(model):
    tp_mesh = DeviceMesh(device_type=device.type, mesh=torch.arange(world_size))
    deepseek_v3_dtensor_parallelize.apply_non_moe_tp(
        model,
        tp_mesh,
        loss_parallel=False,
        enable_float8_tensorwise_tp=False,
        cp_enabled=False,
    )

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=deepseek_v3_model.DeepSeekV3Model,
      model_args=_get_deepseek_v3_model_args(),
      parallelism_func=apply_tp_wrapper,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=True,
  )

  runner.run_forward_parity(TEST_BATCH_SIZE, TEST_SEQ_LEN, atol=2e-1, rtol=1e-1)


def _verify_dtensor_deepseek_v3_tp_backward_worker(
    device: torch.device, rank: int, world_size: int
):
  """Worker function to run forward andbackward pass on model after apply_tp is called."""
  # Apply histc CPU workaround in worker process. Doesn't propagate from parent.
  use_cpu_safe_histc_patch()

  def apply_tp_wrapper(model):
    tp_mesh = DeviceMesh(device_type=device.type, mesh=torch.arange(world_size))
    deepseek_v3_dtensor_parallelize.apply_non_moe_tp(
        model,
        tp_mesh,
        loss_parallel=False,
        enable_float8_tensorwise_tp=False,
        cp_enabled=False,
    )

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=deepseek_v3_model.DeepSeekV3Model,
      model_args=_get_deepseek_v3_model_args(),
      parallelism_func=apply_tp_wrapper,
      input_distribution=InputDistribution.REPLICATE,
      use_meta_init=True,
  )
  runner.run_backward_parity(
      num_steps=TEST_TRAINING_STEPS,
      batch_size=TEST_BATCH_SIZE,
      seq_len=TEST_SEQ_LEN,
  )


def _verify_dtensor_deepseek_v3_fsdp_training_loop_worker(
    device: torch.device,
    rank: int,
    world_size: int,
):
  """Worker function to call the FSDP numerical equivalence helper on DeepSeekV3 model."""
  # Apply histc CPU workaround in worker process. Doesn't propagate from parent.
  use_cpu_safe_histc_patch()

  def apply_fsdp_wrapper(model):
    dp_mesh = DeviceMesh(device_type=device.type, mesh=torch.arange(world_size))
    deepseek_v3_dtensor_parallelize.apply_fsdp(
        model,
        dp_mesh,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        pp_enabled=False,
        cpu_offload=False,
    )

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=deepseek_v3_model.DeepSeekV3Model,
      model_args=_get_deepseek_v3_model_args(),
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


class DeepSeekV3DTensorParallelizeTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests the DTensor-based `apply_non_moe_tp` in a distributed environment."""

  # TP Tests are run with loss_parallel=False.
  def test_apply_non_moe_tp_forward_equivalence_distributed(self):
    """Launches test to verify numerical equivalence of the DTensor parallel model."""
    logging.info(
        "Launching DTensor TP forward numerical equivalence test with %d"
        " devices on DeepSeekV3 model.",
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_dtensor_deepseek_v3_non_moe_tp_forward_worker,
    )  # pytype: disable=wrong-arg-types
    logging.info(
        "Distributed DTensor TP forward numerical equivalence test finished."
    )

  def test_apply_non_moe_tp_backward_equivalence_distributed(self):
    """Tests if the backward pass runs on model after apply_tp."""
    logging.info(
        "Launching DTensor TP backward numerical equivalence test with %d"
        " devices on DeepSeekV3 model.",
    )
    distributed.run_distributed(
        self.num_devices,
        self.accelerator_device_type,
        _verify_dtensor_deepseek_v3_tp_backward_worker,
    )
    logging.info(
        "Distributed DTensor TP backward numerical equivalence test finished."
    )

  def test_apply_fsdp_full_training_loop_equivalence_distributed(self):
    """Verifies numerical equivalence of a full training loop with FSDP."""
    logging.info(
        "Launching FSDP training equivalence test with %d devices on DeepSeekV3"
        " model.",
        self.num_devices,
    )
    distributed.run_distributed(
        num_devices=self.num_devices,
        accelerator_device_type=self.accelerator_device_type,
        func=_verify_dtensor_deepseek_v3_fsdp_training_loop_worker,
    )  # pytype: disable=wrong-arg-types
    logging.info("Distributed FSDP training equivalence test finished.")


if __name__ == "__main__":
  absltest.main()
