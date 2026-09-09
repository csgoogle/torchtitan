# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU GraphTrainer config conversion."""

import dataclasses
import functools
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
  """Convert a TPU recipe and propagate its TPU-only GraphTrainer settings."""
  memory_policy = train_config.tpu_config.graph_trainer_memory_policy
  if memory_policy == "sac_and_offload":
    raise ValueError(
        "TPU GraphTrainer does not support memory_policy='sac_and_offload'"
    )
  graph_config = to_graph_trainer_config(
      train_config, model_registry, target_cls=target_cls
  )
  graph_config.compile.memory_policy = memory_policy
  scan_layers = train_config.tpu_config.use_scan_layers
  stack_layers = train_config.tpu_config.use_stacked_layers or scan_layers
  if stack_layers:
    # Layer stacking is TPU-only, so the flags live on tpu_config. The code
    # that acts on them is inside the shared `apply_simple_fsdp`, which only
    # takes bools -- bind them here rather than teach shared code about
    # tpu_config.
    graph_config.model_spec = dataclasses.replace(
        graph_config.model_spec,
        parallelize_fn=functools.partial(
            graph_config.model_spec.parallelize_fn,
            stack_layers=True,
            scan_layers=scan_layers,
        ),
    )
  return graph_config


__all__ = ["to_tpu_graph_trainer_config"]
