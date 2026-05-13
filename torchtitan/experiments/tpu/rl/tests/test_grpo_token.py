# Copyright 2026 The TorchTitan Authors. All Rights Reserved.

"""Tests for token-level GRPO loss."""

import unittest
import torch
from torchtitan.experiments.tpu.rl import grpo_utils


class TestGRPOToken(unittest.TestCase):
  """Tests for token-level GRPO loss."""

  def test_compute_grpo_loss_token(self):
    """Test token-level GRPO loss computation."""

    class MockModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 100)

      def forward(self, x):
        # Return ones [batch, seq_len, vocab_size]
        return torch.ones(x.shape[0], x.shape[1], 100)

    model = MockModel()
    prompt_ids = torch.zeros(2, 5, dtype=torch.long)
    completed_ids = torch.zeros(2, 10, dtype=torch.long)
    # gen_len = 5
    ref_log_probs = torch.zeros(2, 5)
    advantages = torch.ones(2)  # Use 1.0 to see effect!

    # token_log_probs will be log(1/100) = -4.605
    # log_ratio = -4.605 - 0 = -4.605
    # ratio = exp(-4.605) = 0.01
    # unclipped_loss = 0.01 * 1.0 = 0.01
    # clipped_ratio = clamp(0.01, 0.8, 1.2) = 0.8
    # clipped_loss = 0.8 * 1.0 = 0.8
    # pg_loss = -min(0.01, 0.8) = -0.01
    # kl = exp(4.605) + (-4.605) - 1.0 = 100 - 4.605 - 1 = 94.395
    # Total loss = -0.01 + 0.1 * 94.395 = -0.01 + 9.4395 = 9.4295

    loss = grpo_utils.compute_grpo_loss(
        model,
        prompt_ids,
        completed_ids,
        ref_log_probs,
        advantages,
        old_log_probs=ref_log_probs,
        ppo_clip_eps=0.2,
        grpo_beta=0.1,
    )

    self.assertAlmostEqual(loss.item(), 9.4295, places=3)


if __name__ == "__main__":
  unittest.main()
