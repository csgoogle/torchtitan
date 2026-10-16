"""Muse Spark Open on TPU (text-only).

Wraps ``torchtitan.models.muse_spark`` for the TPU lane, the same way
``experiments/tpu/muse_glimmer`` wraps Muse Glimmer:

* Every layer's inner attention becomes Muse Glimmer's
  ``SlidingWindowScaledDotProductInnerAttention``, so sliding-window layers get
  an SDPA window mask (or Splash ``LocalMask`` under
  ``--splash_attention_kernel.use_splash_attention_kernel``) and global layers
  a causal one, both restricted to each packed document. The ported
  ``MuseSparkAttention`` already forwards ``positions`` and ``segment_ids``.
* ``infra.parallelize.parallelize_muse_spark`` replaces the core FSDP2-only
  parallelize with the ``partial_dtensor`` + SimpleFSDP path.
"""

from torchtitan.experiments.tpu.muse_glimmer.attention import (
    SlidingWindowScaledDotProductInnerAttention,
)
import torchtitan.models.muse_spark as upstream_muse_spark
from torchtitan.models.muse_spark.model import MuseSparkModel
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .infra.parallelize import parallelize_muse_spark

__all__ = [
    "model_registry",
    "parallelize_muse_spark",
    "use_tpu_attention",
]

# The 816B flavor is listed for config inspection; it does not fit one host.
_SUPPORTED_FLAVORS = ("debugmodel", "6B-A1.4B", "220B-A44B", "816B-A42B")


def use_tpu_attention(config: MuseSparkModel.Config) -> MuseSparkModel.Config:
  """Installs the window-aware SDPA inner attention on every layer.

  The window is read from ``MuseSparkAttention.Config.window_size`` (None for
  global layers). Modifies ``config`` in place and returns it.
  """
  for layer in config.layers:
    layer.attention.inner_attention = (
        SlidingWindowScaledDotProductInnerAttention.Config(
            window_size=layer.attention.window_size
        )
    )
  return config


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
  """Returns a TPU ``ModelSpec`` for a text-only Muse Spark flavor."""
  if flavor not in _SUPPORTED_FLAVORS:
    raise ValueError(
        f"Unsupported Muse Spark flavor on TPU: {flavor!r}. Supported:"
        f" {_SUPPORTED_FLAVORS}"
    )
  model_spec = upstream_muse_spark.model_registry(
      flavor, seq_len=seq_len, attn_backend="sdpa", converters=converters
  )
  use_tpu_attention(model_spec.model)
  model_spec.parallelize_fn = parallelize_muse_spark
  return model_spec


from .config_registry import *  # pylint: disable=wildcard-import,g-import-not-at-top,g-bad-import-order
