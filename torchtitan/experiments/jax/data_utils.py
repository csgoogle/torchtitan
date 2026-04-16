"""Data utilities for JAX experiment."""

import numpy as np


def fake_dataloader(size, seqlen, batch_size):
    """Generates synthetic data (numpy arrays, no PyTorch dependency)."""
    for _ in range(size):
        x = np.random.randint(0, 32000, (batch_size, seqlen), dtype=np.int32)
        yield x, (x + 1) % 32000


def torch_loader_to_jax(train_loader):
    """Wrap a torchtitan dataloader to yield numpy arrays."""
    for batch in train_loader:
        inputs, labels = batch[0]['input'], batch[1]
        yield inputs.numpy().astype(np.int32), labels.numpy().astype(np.int32)
