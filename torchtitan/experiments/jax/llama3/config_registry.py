"""Config registry for the JAX-lane Llama3 model."""

from torchtitan.experiments.jax import llama3 as jax_llama3
from torchtitan.experiments.jax.jax_job_config import JaxJobConfig
from torchtitan.experiments.tpu.llama3 import config_registry as tpu_llama3
from torchtitan.protocols.model_spec import ModelSpec


def _model_spec(flavor: str, *, max_context_length: int) -> ModelSpec:
  """JAX-lane ``ModelSpec``. ``parallelize_fn`` is ``None`` — the JAX trainer
  performs sharding via the JAX named mesh + ``llama3.sharding_map_*``."""
  # Upstream d771abb0 made ModelSpec.max_context_length required. This helper
  # only swaps lane-specific fields, so callers thread through the value the
  # derived-from config already resolved.
  return ModelSpec(
      name="llama3",
      flavor=flavor,
      model=jax_llama3.args[flavor],  # pyrefly: ignore[bad-argument-type]
      max_context_length=max_context_length,
      parallelize_fn=lambda *args, **kwargs: None,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def llama3_debugmodel() -> JaxJobConfig:
  """Llama3 debug model on JAX (smoke test)."""
  cfg = JaxJobConfig.derive_from(tpu_llama3.llama3_debugmodel())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(
      cfg.model_spec.flavor,
      max_context_length=cfg.model_spec.max_context_length,
  )
  cfg.jax_config.use_scan = False
  return cfg
