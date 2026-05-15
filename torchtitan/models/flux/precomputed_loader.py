import os
import glob
import torch
from typing import Iterator

class PrecomputedDataloader:
    """
    A dataloader that loads precomputed batches from a directory.

    This dataloader looks for files named `batch_*_rank_{rank}.pt` or `batch_*.pt`
    in the given data directory. It loads them in numerical order of batch index
    and yields each batch and its image encodings up to `max_steps`.
    """
    def __init__(self, data_dir: str, max_steps: int, rank: int = 0):
        self.data_dir = data_dir
        self.max_steps = max_steps
        self.rank = rank

        # Look for rank-specific files first
        self.files = glob.glob(os.path.join(data_dir, f"batch_*_rank_{rank}.pt"))
        if len(self.files) == 0:
            # Fall back to standalone files if rank specific don't exist
            self.files = glob.glob(os.path.join(data_dir, "batch_*.pt"))
            self.files = [f for f in self.files if "rank" not in f]

        if len(self.files) == 0:
            raise ValueError(f"No precomputed batches found in {data_dir} for rank {rank}")

        # Sort numerically by batch index
        self.files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

    def __iter__(self) -> Iterator:
        step = 0
        while True:
            for file_path in self.files:
                if step >= self.max_steps:
                    break
                data = torch.load(file_path, weights_only=False)
                yield data, data["img_encodings"]
                step += 1
            if step >= self.max_steps:
                raise RuntimeError(f"Reached max_steps={self.max_steps} while iterating through precomputed batches. Please ensure that the number of precomputed batches is at least max_steps.")
