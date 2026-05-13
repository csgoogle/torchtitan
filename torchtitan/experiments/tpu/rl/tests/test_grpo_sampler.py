# Copyright 2026 The TorchTitan Authors. All Rights Reserved.

import unittest
import torch
from torchtitan.experiments.tpu.rl import grpo_sampler


class TestGRPOSampler(unittest.TestCase):

  def test_logits_to_probs_temperature(self):
    """Test temperature scaling in logits_to_probs."""
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    # High temperature should make it more uniform
    probs_high = grpo_sampler.logits_to_probs(logits, temperature=10.0)
    # Low temperature should make it more spiked
    probs_low = grpo_sampler.logits_to_probs(logits, temperature=0.1)

    self.assertTrue(torch.all(probs_high > 0))
    self.assertGreater(probs_low[0, 2], 0.9)  # 3.0 is the largest logit

  def test_logits_to_probs_top_k(self):
    """Test top-k filtering in logits_to_probs."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    probs = grpo_sampler.logits_to_probs(logits, temperature=1.0, top_k=2)
    # Only top 2 should have non-zero probability
    self.assertEqual(probs[0, 0], 0.0)
    self.assertEqual(probs[0, 1], 0.0)
    self.assertGreater(probs[0, 2], 0)
    self.assertGreater(probs[0, 3], 0)

  def test_generate_fake(self):
    """Test generate_fake returns correct shapes and values."""
    batch_size = 2

    max_new_tokens = 5
    vocab_size = 100
    input_ids = torch.zeros((batch_size, 3), dtype=torch.long)

    generated_tokens, token_log_probs = grpo_sampler.generate_fake(
        model=None,  # Not used in generate_fake
        input_ids=input_ids,
        max_seq_len=10,
        max_new_tokens=max_new_tokens,
        vocab_size=vocab_size,
    )

    self.assertEqual(generated_tokens.shape, (batch_size, 3 + max_new_tokens))
    self.assertEqual(token_log_probs.shape, (batch_size, max_new_tokens))
    self.assertTrue(torch.all(token_log_probs < 0))
    self.assertTrue(torch.all(generated_tokens >= 0))
    self.assertTrue(torch.all(generated_tokens < vocab_size))


if __name__ == "__main__":
  unittest.main()
