import torch
from typing import Iterator
from torchtitan.models.flux.utils import IMAGE_LATENT_SIZE_RATIO, LATENT_CHANNELS

class RandomDataloader:
    """
    A dataloader that generates random encodings.
    Pre-generates num_precomputed_batches batches and loops them round-robin to avoid generation bottlenecks during training.
    """
    def __init__(self, bsz: int, max_steps: int, img_size: int, max_t5_len: int, t5_dim: int, clip_dim: int):
        self.bsz = bsz
        self.max_steps = max_steps
        self.img_size = img_size
        self.max_t5_len = max_t5_len
        self.t5_dim = t5_dim
        self.clip_dim = clip_dim

        self.batches = []
        h_latent = self.img_size // IMAGE_LATENT_SIZE_RATIO
        w_latent = self.img_size // IMAGE_LATENT_SIZE_RATIO

        for _ in range(self.max_steps):
            img_encodings = torch.randn((self.bsz, LATENT_CHANNELS, h_latent, w_latent))
            t5_encodings = torch.randn((self.bsz, self.max_t5_len, self.t5_dim))
            clip_encodings = torch.randn((self.bsz, self.clip_dim))

            data = {
                "img_encodings": img_encodings,
                "t5_encodings": t5_encodings,
                "clip_encodings": clip_encodings,
            }
            self.batches.append((data, img_encodings))

    def __iter__(self) -> Iterator:
        step = 0
        num_batches = len(self.batches)

        while step < self.max_steps:
            yield self.batches[step % num_batches]
            step += 1

        raise RuntimeError(f"Reached max_steps={self.max_steps} while generating random batches.")
