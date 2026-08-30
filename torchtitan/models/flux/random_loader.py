import logging
import torch
from typing import Iterator
from torchtitan.models.flux.utils import IMAGE_LATENT_SIZE_RATIO, LATENT_CHANNELS

logger = logging.getLogger(__name__)

class RandomDataloader:
    """
    A dataloader that generates random encodings.
    Pre-generates a configurable pool of batches and loops them round-robin
    to avoid generation bottlenecks and memory bloat during training.
    """
    def __init__(
        self,
        bsz: int,
        max_steps: int,
        img_size: int,
        max_t5_len: int,
        t5_dim: int,
        clip_dim: int,
        max_cached_batches: int = 100
    ):
        self.bsz = bsz
        self.max_steps = max_steps
        self.img_size = img_size
        self.max_t5_len = max_t5_len
        self.t5_dim = t5_dim
        self.clip_dim = clip_dim

        self.batches = []
        h_latent = self.img_size // IMAGE_LATENT_SIZE_RATIO
        w_latent = self.img_size // IMAGE_LATENT_SIZE_RATIO

        # Cap the pool size based on the parameter.
        num_samples = min(self.max_steps, max_cached_batches)

        for _ in range(num_samples):
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

        while True:
            if step > 0 and step % num_batches == 0:
                logger.warning("RandomDataset is being re-looped.")

            data, img_enc = self.batches[step % num_batches]

            # Yield a dictionary copy so inplace `.to(device)` calls in
            # trainer.py don't overwrite the cached CPU tensors in self.batches,
            # which grows memory usage and can cause TPU OOM.
            data_copy = {
                "img_encodings": data["img_encodings"],
                "t5_encodings": data["t5_encodings"],
                "clip_encodings": data["clip_encodings"],
            }
            yield data_copy, img_enc
            step += 1
