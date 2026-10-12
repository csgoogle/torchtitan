# Copyright 2026 Google LLC

"""Mamba-3 TPU model definitions, training specifications, and configs."""

# pylint: disable=g-statement-before-imports,g-import-not-at-top

try:
  from . import config_registry

  mamba3_configs = config_registry.mamba3_configs
  model_registry = config_registry.model_registry
  mamba3_2_7b_256k = config_registry.mamba3_2_7b_256k
  mamba3_tpu_configs = {
      "mamba3_2_7b_256k": mamba3_2_7b_256k(),
  }
except (ImportError, ModuleNotFoundError):
  config_registry = None
  mamba3_configs = {}
  model_registry = {}
  mamba3_2_7b_256k = None
  mamba3_tpu_configs = {}

from . import model

try:
  from . import parallelize
  parallelize_mamba3 = parallelize.parallelize_mamba3
except (ImportError, ModuleNotFoundError):
  parallelize = None
  parallelize_mamba3 = None

Mamba3Block = model.Mamba3Block
Mamba3Config = model.Mamba3Config
Mamba3LM = model.Mamba3LM
Mamba3Mixer = model.Mamba3Mixer

__all__ = [
    "Mamba3Block",
    "Mamba3Config",
    "Mamba3LM",
    "Mamba3Mixer",
    "mamba3_configs",
    "model_registry",
    "mamba3_tpu_configs",
    "parallelize_mamba3",
    "mamba3_2_7b_256k",
]
