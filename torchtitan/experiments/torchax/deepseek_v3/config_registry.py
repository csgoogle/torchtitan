"""Config registry for the torchax-lane DeepSeek-v3 model.

Thin delegator on top of ``torchtitan/experiments/tpu/deepseek_v3/config_registry.py``
— calls the tpu registry, then derives a ``TorchaxJobConfig`` from the
resulting ``TPUTrainerConfig`` and mutates lane-specific bits in place.
``flavor`` is inherited via ``cfg.model_spec.flavor``; the string is stated
only in the tpu lane.

Called by ``ConfigManager.parse_args`` via ``--module=torchtitan.experiments.\
torchax.deepseek_v3 --config=<name>``.
"""

from torchtitan.experiments.torchax import deepseek_v3 as torchax_deepseek_v3
from torchtitan.experiments.torchax.torchax_job_config import TorchaxJobConfig
from torchtitan.experiments.tpu.deepseek_v3 import config_registry as tpu_deepseek_v3
from torchtitan.protocols.model_spec import ModelSpec


def _model_spec(flavor: str, *, max_context_length: int) -> ModelSpec:
  """Torchax-lane ``ModelSpec``. ``parallelize_fn`` is ``None`` — the torchax
  trainer performs sharding via the JAX named mesh + the per-flavor sharding
  maps under ``experiments/torchax/deepseek_v3/sharding.py``."""
  # Upstream d771abb0 made ModelSpec.max_context_length required. This helper
  # only swaps lane-specific fields, so callers thread through the value the
  # derived-from config already resolved.
  return ModelSpec(
      name="deepseek_v3",
      flavor=flavor,
      model=torchax_deepseek_v3.args[flavor],
      max_context_length=max_context_length,
      parallelize_fn=None,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def deepseek_v3_debugmodel() -> TorchaxJobConfig:
  """DeepSeek-v3 debug model on torchax (smoke test)."""
  cfg = TorchaxJobConfig.derive_from(tpu_deepseek_v3.deepseek_v3_debugmodel())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(
      cfg.model_spec.flavor,
      max_context_length=cfg.model_spec.max_context_length,
  )
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = False
  return cfg
