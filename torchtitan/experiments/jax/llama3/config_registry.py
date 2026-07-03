"""Config registry for the JAX-lane Llama3 model."""

from torchtitan.experiments.jax import llama3 as jax_llama3
from torchtitan.experiments.jax.jax_job_config import JaxJobConfig
from torchtitan.experiments.tpu.llama3 import config_registry as tpu_llama3
from torchtitan.protocols.model_spec import ModelSpec


def _model_spec(flavor: str) -> ModelSpec:
  """JAX-lane ``ModelSpec``. ``parallelize_fn`` is ``None`` — the JAX trainer
  performs sharding via the JAX named mesh + ``llama3.sharding_map_*``."""
  return ModelSpec(
      name="llama3",
      flavor=flavor,
      model=jax_llama3.args[flavor],  # pyrefly: ignore[bad-argument-type]
      parallelize_fn=lambda *args, **kwargs: None,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def llama3_debugmodel() -> JaxJobConfig:
  """Llama3 debug model on JAX (smoke test)."""
  cfg = JaxJobConfig.derive_from(tpu_llama3.llama3_debugmodel())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.jax_config.use_scan = False
  return cfg
