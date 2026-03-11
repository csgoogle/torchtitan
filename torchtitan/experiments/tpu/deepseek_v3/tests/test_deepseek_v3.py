"""Tests for DeepSeekV3 model components on GPU and TPU devices.

This file contains tests to ensure parity between CPU and GPU/TPU device
implementations of various DeepSeekV3 model layers (including Embedding, and
Attention) as well as full model testing.
"""

from absl.testing import absltest
import torch
from torch import nn
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import test_utils
from torchtitan.models.deepseek_v3.model import model as deepseek_v3_model
from torchtitan.models import moe


class DeepseekV3Test(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for DeepSeekV3 model components on CPU and accelerator devices.

  This class contains tests to ensure parity between CPU and accelerator device
  implementations of various DeepSeekV3 model layers and full model testing.
  """

  def _get_model_args(
      self,
      n_layers: int = 2,
      vocab_size: int = 32,
      max_seq_len: int = 8,
  ) -> deepseek_v3_model.DeepSeekV3ModelArgs:
    """Returns reduced model arguments for DeepSeek V3 for debugging."""
    args = deepseek_v3_model.DeepSeekV3ModelArgs(
        vocab_size=vocab_size,
        dim=64,
        inter_dim=128,
        moe_inter_dim=64,
        n_layers=n_layers,
        n_dense_layers=1,
        n_heads=4,
        moe_args=moe.MoEArgs(
            use_grouped_mm=False,
        ),
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        mscale=0.70,
        max_seq_len=max_seq_len,
        max_batch_size=4,
        original_seq_len=max_seq_len,
    )
    args.moe_impl = "standard"
    return args

  def test_deepseek_embedding_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the Embedding layer."""
    args = self._get_model_args()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    embedding_cpu, embedding_device = self._setup_module_cpu_device(
        nn.Embedding,
        init_fn=lambda m: nn.init.normal_(m.weight),
        num_embeddings=args.vocab_size,
        embedding_dim=args.dim,
    )

    tokens_cpu = torch.randint(
        0,
        args.vocab_size,
        (batch, seq_len),
        device="cpu",
        requires_grad=False,
    )
    tokens_device = tokens_cpu.to(self.accelerator_device)

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = embedding_cpu(tokens_cpu)
      out_device = embedding_device(tokens_device)
      test_utils.check_equivalence(
          out_cpu, out_device.cpu(), atol=1e-5, rtol=1e-5
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
            embedding_cpu.weight.grad,
            embedding_device.weight.grad.cpu(),
            atol=1e-9,
            rtol=1e-9,
        )

  def test_apply_rotary_emb_cpu_device_parity(self):
    """Tests CPU vs. DEVICE parity for apply_rotary_emb function."""

    batch, seq_len, n_heads, head_dim = 2, 8, 4, 16
    # Instantiate args just to trigger the patch in __post_init__
    _ = self._get_model_args()

    # CPU setup
    x_cpu = torch.randn(
        batch, seq_len, n_heads, head_dim, device="cpu", requires_grad=True
    )
    theta = torch.randn(seq_len, head_dim // 2, device="cpu")
    freqs_cis_cpu = torch.exp(1j * theta)

    # DEVICE setup
    x_device = (
        x_cpu.detach().clone().to(self.accelerator_device).requires_grad_(True)
    )
    freqs_cis_device = freqs_cis_cpu.to(self.accelerator_device)

    with self.subTest(name="ForwardPass"):
      # Forward pass
      y_cpu = deepseek_v3_model.apply_rotary_emb(x_cpu, freqs_cis_cpu)
      y_device = deepseek_v3_model.apply_rotary_emb(x_device, freqs_cis_device)

      test_utils.check_equivalence(y_cpu, y_device.cpu(), atol=1e-5, rtol=1e-5)

      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = y_cpu.sum()
        loss_device = y_device.sum()

        loss_cpu.backward()
        loss_device.backward()

        test_utils.check_equivalence(
            x_cpu.grad, x_device.grad.cpu(), atol=1e-5, rtol=1e-5
        )

  def test_deepseek_attention_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the Attention layer."""
    args = self._get_model_args()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    attention_cpu, attention_device = self._setup_module_cpu_device(
        deepseek_v3_model.Attention,
        lambda m: m.init_weights(init_std=0.02),
        args,
    )
    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch, seq_len, args.dim), requires_grad=True
    )
    freqs_cis_cpu = deepseek_v3_model.precompute_freqs_cis(args)
    freqs_cis_device = freqs_cis_cpu.to(self.accelerator_device)

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = attention_cpu(x_cpu, freqs_cis_cpu, None)
      out_device = attention_device(x_device, freqs_cis_device, None)
      test_utils.check_equivalence(out_cpu, out_device.cpu(), atol=1e-3, rtol=1e-3)

      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()

        test_utils.check_equivalence(
            x_cpu.grad, x_device.grad.cpu(), atol=1e-3, rtol=1e-3
        )

        for p_cpu, p_device in zip(
            attention_cpu.parameters(), attention_device.parameters()
        ):
          test_utils.check_equivalence(
              p_cpu.grad, p_device.grad.cpu(), atol=1e-1, rtol=1e-1
          )

  def test_deepseek_has_expected_layers(self):
    """Tests that the DeepSeek model contains MoE and Dense FeedForward layers."""
    args = self._get_model_args()
    model = deepseek_v3_model.DeepSeekV3Model(args)
    model.init_weights()

    contains_moe = False
    contains_dense = False
    for layer in model.modules():
      if isinstance(layer, moe.MoE):
        contains_moe = True
      elif isinstance(layer, deepseek_v3_model.FeedForward):
        contains_dense = True
    self.assertTrue(contains_moe, "Model does not contain MoE layers.")
    self.assertTrue(
        contains_dense, "Model does not contain Dense FeedForward layers."
    )

  def test_deepseek_complete_model_training_steps_cpu_device_parity(self):
    """Tests CPU vs DEVICE parity of full DeepSeek model after 3 training steps.

    Uses baseclass function wrapper for full model training steps parity test.
    """
    args = self._get_model_args()
    batch, seq_len = 2, 8

    model = deepseek_v3_model.DeepSeekV3Model(args)
    model.init_weights()

    self._run_full_model_training_steps_parity_test(
        model_cpu=model.cpu(),
        batch=batch,
        seq_len=seq_len,
        loss_atol=1e-1,
        loss_rtol=1e-1,
        grad_atol=1e-1,
        grad_rtol=1e-1,
        param_atol=1e-1,
        param_rtol=1e-1,
    )

  @test_utils.skip_if_cpu(
      reason="CPU results should be equal",
      bug_id=None,
  )
  @absltest.expectedFailure
  def test_deepseek_parity_test_raises_assertion_error_with_zero_tol(self):
    """Tests that the parity test raises an AssertionError with zero tolerance."""
    args = self._get_model_args()
    batch, seq_len = 2, 8

    model = deepseek_v3_model.DeepSeekV3Model(args)
    model.init_weights()

    self._run_full_model_training_steps_parity_test(
        model_cpu=model.cpu(),
        batch=batch,
        seq_len=seq_len,
        loss_atol=0.0,
        loss_rtol=0.0,
        grad_atol=0.0,
        grad_rtol=0.0,
        param_atol=0.0,
        param_rtol=0.0,
    )


if __name__ == "__main__":
  absltest.main()
