# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Glimmer on the TorchTPU lane.

Thin wrapper over ``torchtitan.models.muse_glimmer``: the same model configs
(text-only flavors), with every flex inner attention swapped for the TPU
splash kernel (``SplashAttentionConverter``) and the TPU parallelizer
installed. The per-layer sliding-window / global pattern reaches the kernel
through ``Attention.Config.sliding_window_size`` (window incl. self), which
the converter maps to splash's ``LocalMask``; NoPE layers need nothing
special (``rope=None`` is honoured by ``GQAttention``).

Multimodal flavors (``*_mm``) are not wired: the vision encoder is built on
flex block-diagonal masks and per-image dynamic shapes that have no TPU path
yet.
"""

from torchtitan.experiments.tpu.attention import SplashAttentionConverter
import torchtitan.models.muse_glimmer as muse_models
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .infra.parallelize import parallelize_muse_glimmer

__all__ = [
    "model_registry",
    "parallelize_muse_glimmer",
    "tpu_converters",
]

_TEXT_FLAVORS = ("debugmodel", "30B")


def tpu_converters(
    converters: list[ModelConfigConverter.Config] | None = None,
) -> list[ModelConfigConverter.Config]:
  """The splash converter first, then whatever the recipe asked for."""
  return [SplashAttentionConverter.Config(), *(converters or [])]


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
  """Upstream's muse_glimmer registry with splash attention and the TPU
  parallelizer.

  ``attn_backend`` names the upstream inner attention the converter replaces;
  it never survives to the built model.
  """
  if flavor not in _TEXT_FLAVORS:
    raise ValueError(
        f"Muse Glimmer flavor {flavor!r} is not supported on the TPU lane;"
        f" text-only flavors are {_TEXT_FLAVORS}."
    )
  model_spec = muse_models.model_registry(
      flavor,
      seq_len=seq_len,
      attn_backend=attn_backend,
      converters=tpu_converters(converters),
  )
  model_spec.parallelize_fn = parallelize_muse_glimmer
  return model_spec
