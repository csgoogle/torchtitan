"""Utilities for initializing distributed training on cluster/TAP and Cloud."""

import os
from absl import logging

from torchtitan.experiments.tpu import cloud_distributed_utils


def maybe_init_distributed(num_devices: int | None = None) -> bool:
  """Initializes the TPU environment variables based on the platform."""
  # Use provided num_devices, or fall back to WORLD_SIZE env var
  if num_devices is None:
    num_devices = int(os.environ.get("WORLD_SIZE", "1"))

  # Note on differences between cluster and Cloud/GKE initialization:
  # - Cloud/GKE: Initialization must be called from within each worker process.
  #   We derive configuration from JAX env vars and initialize `TORCH_TPU_...`
  #   vars globally since we have a global view and `torchrun` does not
  #   support per-process env vars.
  # - cluster: Initialization must be called from the host process before worker
  #   processes begin. This is because ports need to be fixed beforehand,
  #   requiring external orchestration (e.g., via launcher) at the host level.
  if "TORCH_TPU_SLICEBUILDER_ADDRESSES" in os.environ:
    return True  # Already initialzed

  logging.info("Platform: Cloud. Initializing via cloud_distributed_utils.")
  return cloud_distributed_utils.maybe_init_distributed()


