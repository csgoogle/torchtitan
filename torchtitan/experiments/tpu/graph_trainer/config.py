# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU GraphTrainer config conversion."""

from collections.abc import Callable
from typing import TypeVar

from torchtitan.experiments.graph_trainer.configs import to_graph_trainer_config
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.protocols.model_spec import ModelSpec


ConfigT = TypeVar("ConfigT")


def _validate_bf16_rounded_paired_dweight_elision(
    train_config: TPUTrainerConfig,
) -> None:
  if not train_config.tpu_config.elide_bf16_rounded_paired_dweight_casts:
    return

  requirements = {
      "GraphTrainer": train_config.tpu_config.use_graph_trainer,
      "SimpleFSDP": train_config.parallelism.use_simple_fsdp,
      "expert_parallel_degree=1": (
          train_config.parallelism.expert_parallel_degree == 1
      ),
      "pipeline_parallel_degree=1": (
          train_config.parallelism.pipeline_parallel_degree == 1
      ),
      "BF16 FSDP parameters": (
          train_config.training.mixed_precision_param == "bfloat16"
      ),
      "FP32 FSDP reduction": (
          train_config.training.mixed_precision_reduce == "float32"
      ),
      "optimizer_in_graph=False": (
          not train_config.tpu_config.optimizer_in_graph
      ),
  }
  missing = [name for name, enabled in requirements.items() if not enabled]
  if missing:
    raise ValueError(
        "BF16-rounded paired dWeight cast elision requires "
        + ", ".join(missing)
    )

  if train_config.model_spec is None:
    raise ValueError("BF16-rounded paired dWeight cast elision requires a model")
  grouped_experts = list(
      train_config.model_spec.model.traverse(TpuGroupedExperts.Config)
  )
  if not grouped_experts or not all(
      config.emit_bf16_rounded_fp32_paired_dweight
      and config.use_pallas_paired_grouped_mm
      and config.use_kernel_native_weight_layout
      for _, config, _, _ in grouped_experts
  ):
    raise ValueError(
        "BF16-rounded paired dWeight cast elision requires every TPU grouped "
        "expert to emit BF16-rounded FP32 paired dWeights"
    )


def to_tpu_graph_trainer_config(
    train_config: TPUTrainerConfig,
    model_registry: Callable[[str], ModelSpec],
    target_cls: type[ConfigT],
) -> ConfigT:
  """Convert a TPU recipe and propagate its GraphTrainer memory policy."""
  _validate_bf16_rounded_paired_dweight_elision(train_config)
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
