"""Tests for Qwen3 Dense model components on GPU and TPU devices.

This file contains tests to ensure parity between CPU and GPU/TPU device
implementations of various Qwen3 Dense model layers (Embedding, Attention,
FeedForward, and TransformerBlock) and full model training.

NOTE: This file currently verifies the Qwen3 Dense and MoE implementation.
"""

import copy
from typing import Any, Dict, Tuple

from absl.testing import absltest
from absl.testing import parameterized

import torch
from torch import nn
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import test_utils
from torchtitan.models.qwen3.model import model as qwen3_model
import torchtitan.models.moe


# Base arguments for a small Qwen3 test setup.
BASE_TEST_ARGS = {
    "dim": 64,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 4,
    "vocab_size": 32,
    "max_seq_len": 16,
    "head_dim": 16,
    "hidden_dim": 256,
    "rope_theta": 100000.0,
}

DENSE_TEST_CONFIG = {
    **BASE_TEST_ARGS,
    "moe_enabled": False,
    "moe_inter_dim": 0,
    "moe_args": None,
}

MOE_TEST_CONFIG = {
    **BASE_TEST_ARGS,
    "moe_enabled": True,
    "moe_inter_dim": 768,
    # Use for-loop experts for device compatibility on CPU.
    "moe_args": torchtitan.models.moe.MoEArgs(
        num_experts=4, top_k=2, use_grouped_mm=False),
}

FULL_MODEL_TEST_CONFIGS = [
    dict(
        testcase_name="_DENSE",
        test_config=DENSE_TEST_CONFIG,
        loss_atol=2e-2,
        loss_rtol=1e-2,
        grad_atol=0.5,  # Grad noise builds up over multiple steps
        grad_rtol=1e-2,
        param_atol=5e-2,
        param_rtol=1e-2,
        is_moe=False,
    ),
    dict(
        testcase_name="_MOE",
        test_config=MOE_TEST_CONFIG,
        # May need looser tolerances for MOE
        loss_atol=2e-2,
        loss_rtol=1e-2,
        grad_atol=0.5,  # Grad noise builds up over multiple steps
        grad_rtol=1e-2,
        param_atol=5e-2,
        param_rtol=1e-2,
        is_moe=True,   # Used for skipping MOE TPU tests.
    ),
]


class Qwen3Test(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for Qwen3 model components (Dense and MoE) on CPU and accelerator devices.

  This class contains tests to ensure parity between CPU and accelerator device
  implementations of various Qwen3 model layers and full model training.
  """

  def _get_model_args(
      self, test_config: Dict[str, Any]
  ) -> qwen3_model.Qwen3ModelArgs:
    """Returns default model args for Qwen3 model."""
    model_args_dict = test_config.copy()
    return qwen3_model.Qwen3ModelArgs(**model_args_dict)

  def _create_rope_input_tensors_cpu_device(
      self,
      args: qwen3_model.Qwen3ModelArgs,
      batch: int = 2,
      seq_len: int = 8,
      requires_grad: bool = True,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates input tensors (x, rope_cache) necessary for the Attention module."""
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch, seq_len, args.dim),
        requires_grad=requires_grad)

    rope_cache_cpu = qwen3_model.precompute_rope_cache(
        args.head_dim,
        args.max_seq_len,
        base=args.rope_theta)
    rope_cache_device = rope_cache_cpu.to(self.accelerator_device)

    return x_cpu, x_device, rope_cache_cpu, rope_cache_device

  def test_qwen_embedding_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the Embedding layer."""
    args = self._get_model_args(DENSE_TEST_CONFIG)
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    embedding_cpu, embedding_device = self._setup_module_cpu_device(
        nn.Embedding,
        init_fn=lambda m: nn.init.normal_(m.weight),
        num_embeddings=args.vocab_size,
        embedding_dim=args.dim
    )

    tokens_cpu = torch.randint(
        0, args.vocab_size,
        (batch, seq_len),
        device="cpu")
    tokens_device = tokens_cpu.to(self.accelerator_device)

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = embedding_cpu(tokens_cpu)
      out_device = embedding_device(tokens_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=1e-5,
          rtol=1e-5,
          check_name="Embedding Forward",
      )

      with self.subTest(name="BackwardPass"):
        # Backward pass
        # Add requires_grad=True for output to calculate gradients.
        out_cpu.requires_grad_(True)
        out_device.requires_grad_(True)
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()
        test_utils.check_equivalence(
            embedding_device.weight.grad.cpu(),
            embedding_cpu.weight.grad,
            atol=1e-5,
            rtol=1e-5,
            check_name="Embedding Weight Grad",
        )

  def test_qwen_attention_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the Attention layer."""
    args = self._get_model_args(DENSE_TEST_CONFIG)
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    attention_cpu, attention_device = self._setup_module_cpu_device(
        qwen3_model.Attention,
        lambda m: m.init_weights(init_std=0.02),
        args
    )

    x_cpu, x_device, rope_cache_cpu, rope_cache_device = (
        self._create_rope_input_tensors_cpu_device(args, batch, seq_len)
    )

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = attention_cpu(x_cpu, rope_cache_cpu, None)
      out_device = attention_device(x_device, rope_cache_device, None)
      # Note looser tolerance.
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=1e-3,
          rtol=1e-3,
          check_name="Attention Forward",
      )
      with self.subTest(name="BackwardPass"):
        # Backward pass
        # requires_grad=True per _create_rope_input_tensors_cpu_device().
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()
        # Gradient check (Note looser tolerance).
        test_utils.check_equivalence(
            x_device.grad.cpu(),
            x_cpu.grad,
            atol=1e-3,
            rtol=1e-3,
            check_name="Attention Input Grad",
        )
        for (name, p_cpu), p_device in zip(
            attention_cpu.named_parameters(), attention_device.parameters()
        ):
          test_utils.check_equivalence(
              p_device.grad.cpu(),
              p_cpu.grad,
              atol=5e-2,
              rtol=5e-2,
              check_name=f"Attention Param Grad: {name}"
          )

  def test_qwen_rmsnorm_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the RMSNorm layer."""
    args = self._get_model_args(DENSE_TEST_CONFIG)
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    # Manual setup kept ad-hoc for simplicity.
    norm_cpu = nn.RMSNorm(args.dim, eps=args.norm_eps).cpu()
    norm_device = copy.deepcopy(norm_cpu).to(self.accelerator_device)
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch, seq_len, args.dim), requires_grad=True
    )

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = norm_cpu(x_cpu)
      out_device = norm_device(x_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=1e-5,
          rtol=1e-5,
          check_name="RMSNorm Forward",
      )
      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()
        # Gradient check (Note looser tolerance).
        test_utils.check_equivalence(
            x_device.grad.cpu(),
            x_cpu.grad,
            atol=5e-4,
            rtol=5e-4,
            check_name="RMSNorm Input Grad",
        )
        for (name, p_cpu), p_device in zip(
            norm_cpu.named_parameters(), norm_device.parameters()
        ):
          test_utils.check_equivalence(
              p_device.grad.cpu(),
              p_cpu.grad,
              atol=5e-4,
              rtol=5e-4,
              check_name=f"RMSNorm Param Grad: {name}",
          )

  def test_qwen_dense_feedforward_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the Dense FeedForward layer."""
    args = self._get_model_args(DENSE_TEST_CONFIG)
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    feedforward_cpu, feedforward_device = self._setup_module_cpu_device(
        qwen3_model.FeedForward,
        init_fn=lambda m: m.init_weights(init_std=0.02),
        dim=args.dim,
        hidden_dim=args.hidden_dim)
    x_cpu, x_device = self._create_input_tensor_cpu_device((
        batch, seq_len, args.dim), requires_grad=True)

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = feedforward_cpu(x_cpu)
      out_device = feedforward_device(x_device)
      # Note looser tolerance.
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=1e-4,
          rtol=1e-4,
          check_name="Dense FFN Forward",
      )
      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()
        # Gradient check (Note looser tolerance).
        test_utils.check_equivalence(
            x_device.grad.cpu(),
            x_cpu.grad,
            atol=5e-4,
            rtol=5e-4,
            check_name="Dense FFN Input Grad",
        )
        for (name, p_cpu), p_device in zip(
            feedforward_cpu.named_parameters(),
            feedforward_device.parameters(),
        ):
          test_utils.check_equivalence(
              p_device.grad.cpu(),
              p_cpu.grad,
              atol=5e-3,
              rtol=5e-3,
              check_name=f"Dense FFN Param Grad: {name}",
          )

  def test_qwen_moe_feedforward_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the raw MoE layer."""
    args = self._get_model_args(MOE_TEST_CONFIG)
    batch, seq_len = 2, 8

    # CPU setup (uses for-loop impl)
    cpu_moe_args = copy.deepcopy(args.moe_args)
    cpu_moe_args.use_grouped_mm = False 
    moe_cpu = torchtitan.models.moe.MoE(
        moe_args=cpu_moe_args,
        dim=args.dim,
        hidden_dim=args.moe_inter_dim,
    )
    moe_cpu.init_weights(init_std=0.02, buffer_device=torch.device("cpu"))
    moe_cpu = moe_cpu.cpu()

    # Device setup (uses grouped_mm default)
    device_moe_args = copy.deepcopy(args.moe_args)
    moe_device = torchtitan.models.moe.MoE(
        moe_args=device_moe_args,
        dim=args.dim,
        hidden_dim=args.moe_inter_dim,
    )
    moe_device.load_state_dict(moe_cpu.state_dict())
    moe_device = moe_device.to(self.accelerator_device)
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch, seq_len, args.dim), requires_grad=True
    )

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = moe_cpu(x_cpu)
      out_device = moe_device(x_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=1e-3,
          rtol=1e-3,
          check_name="MoE Forward",
      )

      with self.subTest(name="BackwardPass"):
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()

        test_utils.check_equivalence(
            x_device.grad.cpu(),
            x_cpu.grad,
            atol=5e-3,
            rtol=5e-3,
            check_name="MoE Input Grad",
        )
        for (name, p_cpu), p_device in zip(
            moe_cpu.named_parameters(), moe_device.parameters()
        ):
          test_utils.check_equivalence(
              p_device.grad.cpu(),
              p_cpu.grad,
              atol=5e-3,
              rtol=5e-3,
              check_name=f"MoE Param Grad: {name}",
          )

  @parameterized.named_parameters(*FULL_MODEL_TEST_CONFIGS)
  def test_qwen_transformer_block_cpu_device_parity(
      self,
      test_config,
      loss_atol, loss_rtol, grad_atol, grad_rtol, param_atol, param_rtol, is_moe,  # pylint: disable=unused-argument
  ):
    """Tests the CPU vs. DEVICE parity of the TransformerBlock layer."""

    args = self._get_model_args(test_config)
    batch, seq_len, layer_id = 2, 8, 0
    transformer_block_cpu, transformer_block_device = (
        self._setup_module_cpu_device(
            qwen3_model.TransformerBlock,
            lambda m: m.init_weights(buffer_device=torch.device("cpu")),
            layer_id,
            args,
        )
    )
    x_cpu, x_device, rope_cache_cpu, rope_cache_device = (
        self._create_rope_input_tensors_cpu_device(args, batch, seq_len)
    )

    out_cpu = transformer_block_cpu(x_cpu, rope_cache_cpu, None)
    out_device = transformer_block_device(x_device, rope_cache_device, None)
    test_utils.check_equivalence(
        out_device.cpu(),
        out_cpu,
        atol=5e-3,
        rtol=5e-3,
        check_name="TransformerBlock Forward",
    )

    out_cpu.sum().backward()
    out_device.sum().backward()
    test_utils.check_equivalence(
        x_device.grad.cpu(),
        x_cpu.grad,
        atol=5e-4,
        rtol=5e-4,
        check_name="TransformerBlock Input Grad",
    )
    for (name, p_cpu), p_device in zip(
        transformer_block_cpu.named_parameters(),
        transformer_block_device.parameters(),
    ):
      param_atol = 5e-3
      param_rtol = 5e-3
      # Relax tolerances for Attention layers to cover noise from new kernel.
      if "attention" in name:
        param_atol = 5e-2
        param_rtol = 5e-2
      test_utils.check_equivalence(
          p_device.grad.cpu(),
          p_cpu.grad,
          atol=param_atol,
          rtol=param_rtol,
          check_name=f"TransformerBlock Param Grad: {name}",
      )

  @parameterized.named_parameters(*FULL_MODEL_TEST_CONFIGS)
  def test_qwen_full_model_training_steps_cpu_device_parity(
      self,
      test_config,
      loss_atol,
      loss_rtol,
      grad_atol,
      grad_rtol,
      param_atol,
      param_rtol,
      is_moe,  # pylint: disable=unused-argument
  ):
    """Tests CPU vs. DEVICE parity of full model after 3 training steps."""

    args = self._get_model_args(test_config)
    batch, seq_len = 2, 8

    # For MoE's it is important to initialize model before passing to test
    # helper.
    model_cpu = qwen3_model.Qwen3Model(args)
    model_cpu.init_weights(buffer_device=torch.device("cpu"))
    model_cpu = model_cpu.cpu()

    self._run_full_model_training_steps_parity_test(
        model_cpu=model_cpu,
        # Tolerances for Qwen3 are looser than Llama3.
        loss_atol=loss_atol,
        loss_rtol=loss_rtol,
        grad_atol=grad_atol,  # Grad noise builds up over multiple steps
        grad_rtol=grad_rtol,
        param_atol=param_atol,
        param_rtol=param_rtol,
        vocab_size=args.vocab_size,
        batch=batch,
        seq_len=seq_len,
    )

if __name__ == "__main__":
  absltest.main()
