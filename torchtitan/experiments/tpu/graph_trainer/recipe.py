# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recipe construction helpers for TPU GraphTrainer."""

from collections.abc import Callable

from torchtitan.experiments.graph_trainer.configs import to_graph_trainer_config
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.protocols.model_spec import ModelSpec


def to_tpu_graph_trainer_config(
    base_config: TPUTrainerConfig,
    model_registry: Callable[[str], ModelSpec],
) -> TpuGraphTrainer.Config:
  """Convert a TPU recipe directly to its final GraphTrainer config type."""
  return to_graph_trainer_config(
      base_config,
      model_registry,
      target_cls=TpuGraphTrainer.Config,
  )


__all__ = ["to_tpu_graph_trainer_config"]
