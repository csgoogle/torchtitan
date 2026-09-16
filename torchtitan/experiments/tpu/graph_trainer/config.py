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
from torchtitan.experiments.tpu.graph_trainer.layer_stack import (
    with_tpu_stacked_layers,
)
from torchtitan.experiments.tpu.graph_trainer.scan_guards import (
    check_scan_parallelism,
)
from torchtitan.experiments.tpu.layer_loop import builds_layer_loop
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
  budget = train_config.tpu_config.graph_trainer_cpu_offload_budget_gb
  if budget < 0:
    raise ValueError("CPU-offload budget must be nonnegative")
  graph_config = to_graph_trainer_config(
      train_config, model_registry, target_cls=target_cls
  )
  graph_config.compile.memory_policy = memory_policy
  graph_config.compile.cpu_offload_budget_gb = budget
  if train_config.tpu_config.use_scan_layers:
    check_scan_parallelism(train_config.parallelism)
    # Expert parallelism shards `moe.experts` on a mesh of its own. A stack is
    # one tensor sharded once, so there is nowhere to put that. Refusing MoE
    # already covers every model that can set this; the check is here so the
    # flag gets named instead of failing somewhere further in.
    if train_config.parallelism.expert_parallel_degree > 1:
      raise ValueError(
          "--tpu-config.use-scan-layers cannot be combined with "
          "--parallelism.expert-parallel-degree > 1 "
          "(https://github.com/google-pytorch/torchtitan/issues/395)."
      )
    # Scanning is a property of the model, so it is expressed as a config
    # rewrite: the recipe gets put on a model class that stacks its layer
    # weights on construction and runs them as one scan, and nothing in the
    # parallelize path has to know about it.
    #
    # The rewrite happens here rather than in the recipe because the
    # GraphTrainer model spec built above comes from a fixed class in the
    # graph-trainer registry, which discards whatever the recipe had set.
    graph_config.model_spec = dataclasses.replace(
        graph_config.model_spec,
        model=with_tpu_stacked_layers(
            graph_config.model_spec.model, use_scan=True
        ),
    )
    # The rewrite is only a config edit, so anything added after it that
    # rebuilds `model_spec` from a fixed class drops it again -- and an
    # unstacked run looks entirely normal. Check the promise this function
    # makes, rather than leaving it to the ordering comment above.
    if not builds_layer_loop(graph_config.model_spec.model):
      raise AssertionError(
          "--tpu-config.use-scan-layers did not survive config conversion"
      )
  return graph_config


__all__ = ["to_tpu_graph_trainer_config"]
