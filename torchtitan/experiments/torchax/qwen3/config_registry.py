"""Config registry for the torchax-lane Qwen3 model."""

from torchtitan.experiments.torchax import qwen3 as torchax_qwen3
from torchtitan.experiments.torchax.torchax_job_config import TorchaxJobConfig
from torchtitan.experiments.tpu.qwen3 import config_registry as tpu_qwen3
from torchtitan.protocols.model_spec import ModelSpec


def _model_spec(flavor: str) -> ModelSpec:
  """Torchax-lane ``ModelSpec``. ``parallelize_fn`` is ``None`` — the torchax
  trainer performs sharding via the JAX named mesh + the per-flavor sharding
  maps under ``experiments/torchax/qwen3/sharding.py``."""
  return ModelSpec(
      name="qwen3",
      flavor=flavor,
      model=torchax_qwen3.args[flavor],
      parallelize_fn=None,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def qwen3_debugmodel() -> TorchaxJobConfig:
  """Qwen3 debug model on torchax (smoke test)."""
  cfg = TorchaxJobConfig.derive_from(tpu_qwen3.qwen3_debugmodel())
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  # tpu's qwen3_debugmodel disables weight tying on the model config — preserve
  # that mutation on the torchax-side model args as well.
  cfg.model_spec.model.enable_weight_tying = False
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = False
  return cfg
