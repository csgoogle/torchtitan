# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU GraphTrainer config conversion."""

import dataclasses
from collections.abc import Callable
from typing import TypeVar

from torchtitan.experiments.graph_trainer.configs import to_graph_trainer_config
from torchtitan.experiments.tpu.scan import resolve_peel_first_layer
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.protocols.model_spec import ModelSpec


ConfigT = TypeVar("ConfigT")


def _scan_parallelize_fn(train_config: TPUTrainerConfig) -> Callable:
  """llama3's scan-aware `parallelize_fn` for this lane, bound to the config."""
  from torchtitan.experiments.tpu.graph_trainer.parallelize_scan import (
      parallelize_llama_scan,
  )

  peel_first_layer = resolve_peel_first_layer(train_config)

  def parallelize(model, **kwargs):
    return parallelize_llama_scan(
        model, peel_first_layer=peel_first_layer, **kwargs
    )

  return parallelize


def to_tpu_graph_trainer_config(
    train_config: TPUTrainerConfig,
    model_registry: Callable[[str], ModelSpec],
    target_cls: type[ConfigT],
) -> ConfigT:
  """Convert a TPU recipe and propagate its GraphTrainer memory policy."""
  memory_policy = train_config.tpu_config.graph_trainer_memory_policy
  if memory_policy == "sac_and_offload":
    raise ValueError(
        "TPU GraphTrainer does not support memory_policy='sac_and_offload'"
    )
  graph_config = to_graph_trainer_config(
      train_config, model_registry, target_cls=target_cls
  )
  graph_config.compile.memory_policy = memory_policy
  # The spec `to_graph_trainer_config` returns is a copy, made before `train.py`
  # installs `patched_parallelize` on the original -- so `tpu_use_scan` cannot
  # reach the lane through that patch, and the scan wiring has to be part of
  # building the copy rather than a second edit to it afterwards.
  if (
      train_config.model_spec is not None
      and train_config.model_spec.name == "llama3"
      and train_config.tpu_config.use_scan
  ):
    graph_config.model_spec = dataclasses.replace(
        graph_config.model_spec,
        parallelize_fn=_scan_parallelize_fn(train_config),
    )
  return graph_config


__all__ = ["to_tpu_graph_trainer_config"]
