"""Utility functions for torch titan experiments."""

import torch


def fake_dataloader(size, seqlen, batch_size):
  """Generates synthetic data for training."""
  for _ in range(size):
    x = torch.randint(0, 32000, (batch_size, seqlen), device='cpu')
    yield {"input": x}, (x + 1) % 32000
