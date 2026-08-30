"""Config registry for Conformer on Torchax."""

import torchtitan.experiments.torchax.torchax_job_config as torchax_job_config_module
import torchtitan.experiments.tpu.conformer
import torchtitan.experiments.tpu.conformer.config_registry as tpu_conformer
import torchtitan.protocols.model_spec as protocols_model_spec


def _model_spec(flavor: str) -> protocols_model_spec.ModelSpec:
  return protocols_model_spec.ModelSpec(
      name="conformer",
      flavor=flavor,
      model=torchtitan.experiments.tpu.conformer.conformer_args[flavor],
      parallelize_fn=None,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def conformer_test() -> torchax_job_config_module.TorchaxJobConfig:
  """Conformer test model on torchax."""
  cfg = torchax_job_config_module.TorchaxJobConfig.derive_from(
      tpu_conformer.conformer_test()
  )
  assert cfg.model_spec is not None, (
      "tpu conformer_test config must have a model_spec"
  )
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = True
  return cfg


def conformer_debugmodel() -> torchax_job_config_module.TorchaxJobConfig:
  """Conformer debug model on torchax."""
  cfg = torchax_job_config_module.TorchaxJobConfig.derive_from(
      tpu_conformer.conformer_debugmodel()
  )
  assert cfg.model_spec is not None, (
      "tpu conformer_debugmodel config must have a model_spec"
  )
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = True
  return cfg


