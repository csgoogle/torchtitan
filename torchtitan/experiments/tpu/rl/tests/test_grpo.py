# Copyright 2026 The TorchTitan Authors. All Rights Reserved.

"""Tests for GRPO logic."""

import unittest
import unittest.mock
import torch
from torchtitan.experiments.tpu.rl import grpo_utils


class TestGRPO(unittest.TestCase):
  """Tests for GRPO logic."""

  def test_compute_grpo_advantages(self):
    """Test GRPO advantage computation."""
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

    group_size = 2
    # Group 1: [1.0, 2.0], Mean = 1.5, Std = 0.5
    # Group 2: [3.0, 4.0], Mean = 3.5, Std = 0.5
    advantages, _ = grpo_utils.compute_grpo_advantages(
        rewards, group_size=group_size
    )

    expected = torch.tensor([-1.0, 1.0, -1.0, 1.0])

    torch.testing.assert_close(advantages, expected, atol=1e-3, rtol=1e-3)

  @unittest.mock.patch("torch.distributed.tensor.distribute_tensor")
  def test_sync_model_weights(self, mock_distribute):
    """Test model weights synchronization."""
    mock_distribute.side_effect = lambda t, device_mesh, placements: t

    class SimpleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    model1 = SimpleModel()
    model2 = SimpleModel()

    with torch.no_grad():
      model1.linear.weight.copy_(torch.randn(10, 10))
      model1.linear.bias.copy_(torch.randn(10))

    self.assertFalse(torch.allclose(model1.linear.weight, model2.linear.weight))

    model1.device_mesh = None
    grpo_utils.sync_model_weights(model1, model2, parallel_dims=None)

    self.assertTrue(torch.allclose(model1.linear.weight, model2.linear.weight))
    self.assertTrue(torch.allclose(model1.linear.bias, model2.linear.bias))

  def test_compute_grpo_loss(self):
    """Test GRPO loss computation."""

    class MockModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

      def forward(self, x):
        # Return dummy logits [batch, seq_len, vocab_size]
        # Fill with non-zero to avoid log(0)
        # F.log_softmax handles zeros fine, but let's make it realistic
        return torch.ones(x.shape[0], x.shape[1], 100)

    model = MockModel()
    prompt_ids = torch.zeros(2, 5, dtype=torch.long)
    completed_ids = torch.zeros(2, 10, dtype=torch.long)
    ref_log_probs = torch.zeros(2, 5)

    advantages = torch.zeros(2)

    loss = grpo_utils.compute_grpo_loss(
        model,
        prompt_ids,
        completed_ids,
        ref_log_probs,
        advantages,
        grpo_beta=0.0,
    )

    # Since advantages are 0 and beta=0, loss should be 0.
    self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
  unittest.main()
