# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU GraphTrainer config conversion."""

from collections.abc import Callable
from typing import TypeVar

from torchtitan.experiments.graph_trainer.configs import to_graph_trainer_config
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.protocols.model_spec import ModelSpec


ConfigT = TypeVar("ConfigT")


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
  return graph_config


__all__ = ["to_tpu_graph_trainer_config"]
