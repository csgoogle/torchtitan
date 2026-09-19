# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 sharding helpers shared by regular Trainer and GraphTrainer."""

import torch.nn as nn

import torchtitan.distributed
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.tools.logging import logger


def configure_qwen_simple_fsdp_shard_dims(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
) -> None:
  """Use divisible hidden axes for uneven Qwen parameters under SimpleFSDP."""
  fsdp_degree = parallel_dims.dp_shard * parallel_dims.cp
  if fsdp_degree <= 1:
    return

  hidden_axis_modules: list[str] = []
  for module_fqn in ("tok_embeddings", "lm_head"):
    module = getattr(model, module_fqn)
    weight = module.weight
    if weight.shape[0] % fsdp_degree == 0:
      continue
    if weight.ndim != 2 or weight.shape[1] % fsdp_degree != 0:
      raise ValueError(
          f"Cannot FSDP-shard {module_fqn}.weight with shape "
          f"{tuple(weight.shape)} over {fsdp_degree} ranks"
      )
    # This is apply_simple_fsdp's documented per-submodule shard-axis hook,
    # also used by LayerStackHolder for stacked layer weights.
    setattr(module, "_fsdp_shard_dim", 1)
    hidden_axis_modules.append(module_fqn)

  # With EP, apply_simple_fsdp separately wraps each expert bank on the EDP
  # mesh and owns its shard-axis selection.
  if not parallel_dims.ep_enabled:
    qwen_expert_modules = [
        (module_fqn, module)
        for module_fqn, module in model.named_modules()
        if isinstance(module, TpuGroupedExperts)
    ]
    for module_fqn, module in qwen_expert_modules:
      if module.num_experts % fsdp_degree == 0:
        continue

      parameters = list(module.named_parameters(recurse=False))
      if not parameters:
        raise ValueError(f"{module_fqn} has no direct parameters to FSDP-shard")
      incompatible = [
          f"{name}{tuple(parameter.shape)}"
          for name, parameter in parameters
          if (
              parameter.ndim != 3
              or parameter.shape[0] != module.num_experts
              or parameter.shape[1] % fsdp_degree != 0
          )
      ]
      if incompatible:
        raise ValueError(
            f"Cannot FSDP-shard {module_fqn} on Qwen's hidden dimension 1 "
            f"over {fsdp_degree} ranks; incompatible expert parameters: "
            f"{incompatible}"
        )
      # See the documented apply_simple_fsdp hook above.
      setattr(module, "_fsdp_shard_dim", 1)
      hidden_axis_modules.append(module_fqn)

  if hidden_axis_modules:
    logger.info(
        "Using SimpleFSDP dimension 1 for %d uneven Qwen modules",
        len(hidden_axis_modules),
    )


__all__ = ["configure_qwen_simple_fsdp_shard_dims"]
