"""Utility functions for torch titan experiments."""

import torch


def fake_dataloader(size, seqlen, batch_size, vocab_size=32000):
  """Generates synthetic data for training."""
  for _ in range(size):
    x = torch.randint(0, vocab_size, (batch_size, seqlen), device="cpu")
    yield {"input": x}, (x + 1) % vocab_size
