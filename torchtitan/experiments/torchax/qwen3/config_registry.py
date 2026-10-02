"""Config registry for the torchax-lane Qwen3 model."""

from functools import partial

import torch.nn as nn

from torchtitan.experiments.torchax import qwen3 as torchax_qwen3
from torchtitan.experiments.torchax.torchax_job_config import TorchaxJobConfig
from torchtitan.experiments.tpu.qwen3 import config_registry as tpu_qwen3
from torchtitan.protocols.model_spec import ModelSpec


def _model_spec(flavor: str, *, max_context_length: int) -> ModelSpec:
  """Torchax-lane ``ModelSpec``. ``parallelize_fn`` is ``None`` — the torchax
  trainer performs sharding via the JAX named mesh + the per-flavor sharding
  maps under ``experiments/torchax/qwen3/sharding.py``."""
  # Upstream d771abb0 made ModelSpec.max_context_length required. This helper
  # only swaps lane-specific fields, so callers thread through the value the
  # derived-from config already resolved.
  return ModelSpec(
      name="qwen3",
      flavor=flavor,
      model=torchax_qwen3.args[flavor],
      max_context_length=max_context_length,
      parallelize_fn=None,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def qwen3_debugmodel() -> TorchaxJobConfig:
  """Qwen3 debug model on torchax (smoke test)."""
  cfg = TorchaxJobConfig.derive_from(tpu_qwen3.qwen3_debugmodel())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(
      cfg.model_spec.flavor,
      max_context_length=cfg.model_spec.max_context_length,
  )
  # tpu's qwen3_debugmodel disables weight tying on the model config — preserve
  # that mutation on the torchax-side model args as well. Untying needs the
  # embedding initializer too: the tied flavor skips it (see the tpu config).
  cfg.model_spec.model.enable_weight_tying = False
  cfg.model_spec.model.tok_embeddings.param_init = {
      "weight": partial(nn.init.normal_, std=1.0)
  }
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = False
  return cfg
