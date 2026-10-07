"""SPMD utilities for TPU."""

import torch
from torchtitan.tools.logging import logger

try:
  spmd_safe = torch.tpu.distributed.spmd_safe
except AttributeError:
  try:
    # torch_tpu releases before the public API (e.g. 0.1.2) only have it here.
    from torch_tpu._internal.distributed.spmd_util import spmd_safe  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    logger.warning(
        "Unable to find spmd_safe decorator in torch_tpu. Compiled modules "
        f"will fall back to skinny collective behavior.\nDetails: {e}"
    )
    def spmd_safe(func):
      return func


def apply_spmd_safe(mod: torch.nn.Module) -> torch.nn.Module:
  """Monkey patches a module's forward method with torch_tpu spmd_safe decorator."""
  orig_forward = mod.forward
  mod.forward = spmd_safe(orig_forward)
  return mod
