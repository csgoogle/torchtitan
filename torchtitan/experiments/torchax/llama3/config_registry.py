"""Config registry for the torchax-lane Llama3 model."""

from torchtitan.experiments.torchax import llama3 as torchax_llama3
from torchtitan.experiments.torchax.torchax_job_config import TorchaxJobConfig
from torchtitan.experiments.tpu.llama3 import config_registry as tpu_llama3
from torchtitan.models.llama3 import config_registry as models_llama3
from torchtitan.protocols.model_spec import ModelSpec


def _model_spec(flavor: str) -> ModelSpec:
  """Torchax-lane ``ModelSpec``. ``parallelize_fn`` is ``None`` — the torchax
  trainer performs sharding via the JAX named mesh + the per-flavor sharding
  maps under ``experiments/torchax/llama3/sharding.py``."""
  return ModelSpec(
      name="llama3",
      flavor=flavor,
      model=torchax_llama3.args[flavor],
      parallelize_fn=None,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def llama3_debugmodel() -> TorchaxJobConfig:
  """Llama3 debug model on torchax (smoke test)."""
  cfg = TorchaxJobConfig.derive_from(tpu_llama3.llama3_debugmodel())
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = False
  return cfg


def llama3_1b() -> TorchaxJobConfig:
  """Llama3 1B on torchax."""
  cfg = TorchaxJobConfig.derive_from(tpu_llama3.llama3_1b())
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = True
  return cfg


def llama3_8b() -> TorchaxJobConfig:
  """Llama3 8B on torchax."""
  cfg = TorchaxJobConfig.derive_from(tpu_llama3.llama3_8b())
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = True
  return cfg


def llama3_70b() -> TorchaxJobConfig:
  """Llama3 70B on torchax."""
  cfg = TorchaxJobConfig.derive_from(models_llama3.llama3_70b())
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = True
  return cfg


def llama3_405b() -> TorchaxJobConfig:
  """Llama3 405B on torchax."""
  cfg = TorchaxJobConfig.derive_from(models_llama3.llama3_405b())
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = True
  return cfg
