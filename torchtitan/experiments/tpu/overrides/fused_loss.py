# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Output-projection override for fused linear cross entropy on TPU."""

from dataclasses import dataclass

import torch

from torchtitan.config import derive, override
from torchtitan.models.common import Linear


class FusedLossLinearWrapper(Linear):
  """LM-head replacement that supplies operands to a fused loss.

  This module owns the same ``lm_head.weight`` as an ordinary ``Linear``, but
  its forward does not execute the projection or the loss. Instead, it returns
  ``(hidden_states, weight.T)``. ``PallasCrossEntropyLoss`` consumes that pair
  and performs the deferred linear projection, softmax, and cross entropy in a
  single fused kernel without materializing the full logits tensor.

  Constructing the wrapper from ``Linear.Config`` preserves the normal weight
  initialization, sharding, optimizer ownership, and checkpoint key.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Linear.Config):
    pass

  def __init__(self, config: Config):
    if config.bias:
      raise ValueError("Fused linear loss does not support a biased LM head")
    super().__init__(config)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x, self.weight.t()


@override(
    "tpu_fused_loss_lm_head",
    target=Linear.Config,
    fqns=["model_spec.model.lm_head"],
    exact=True,
    description="Defer the Qwen LM-head projection to TPU fused cross entropy.",
)
def fused_loss_lm_head(cfg: Linear.Config) -> FusedLossLinearWrapper.Config:
  return derive(cfg, FusedLossLinearWrapper.Config)


__all__ = ["FusedLossLinearWrapper", "fused_loss_lm_head"]
