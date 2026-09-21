"""Config registry for the torchax-lane Llama3 model."""

from torchtitan.experiments.torchax import llama3 as torchax_llama3
from torchtitan.experiments.torchax.torchax_job_config import TorchaxJobConfig
from torchtitan.experiments.tpu.llama3 import config_registry as tpu_llama3
from torchtitan.models.llama3 import config_registry as models_llama3
from torchtitan.protocols.model_spec import ModelSpec


def _model_spec(flavor: str, *, max_context_length: int) -> ModelSpec:
  """Torchax-lane ``ModelSpec``. ``parallelize_fn`` is ``None`` — the torchax
  trainer performs sharding via the JAX named mesh + the per-flavor sharding
  maps under ``experiments/torchax/llama3/sharding.py``."""
  # Upstream d771abb0 made ModelSpec.max_context_length required. This helper
  # only swaps lane-specific fields, so callers thread through the value the
  # derived-from config already resolved.
  return ModelSpec(
      name="llama3",
      flavor=flavor,
      model=torchax_llama3.args[flavor],
      max_context_length=max_context_length,
      parallelize_fn=None,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def llama3_debugmodel() -> TorchaxJobConfig:
  """Llama3 debug model on torchax (smoke test)."""
  cfg = TorchaxJobConfig.derive_from(tpu_llama3.llama3_debugmodel())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(
      cfg.model_spec.flavor,
      max_context_length=cfg.model_spec.max_context_length,
  )
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = False
  return cfg


def llama3_1b() -> TorchaxJobConfig:
  """Llama3 1B on torchax."""
  cfg = TorchaxJobConfig.derive_from(tpu_llama3.llama3_1b())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(
      cfg.model_spec.flavor,
      max_context_length=cfg.model_spec.max_context_length,
  )
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = True
  return cfg


def llama3_8b() -> TorchaxJobConfig:
  """Llama3 8B on torchax."""
  cfg = TorchaxJobConfig.derive_from(tpu_llama3.llama3_8b())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(
      cfg.model_spec.flavor,
      max_context_length=cfg.model_spec.max_context_length,
  )
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = True
  return cfg


def llama3_70b() -> TorchaxJobConfig:
  """Llama3 70B on torchax."""
  cfg = TorchaxJobConfig.derive_from(models_llama3.llama3_70b())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(
      cfg.model_spec.flavor,
      max_context_length=cfg.model_spec.max_context_length,
  )
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = True
  return cfg


def llama3_405b() -> TorchaxJobConfig:
  """Llama3 405B on torchax."""
  cfg = TorchaxJobConfig.derive_from(models_llama3.llama3_405b())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(
      cfg.model_spec.flavor,
      max_context_length=cfg.model_spec.max_context_length,
  )
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = True
  return cfg
