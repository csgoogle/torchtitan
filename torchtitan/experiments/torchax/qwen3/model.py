# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


import torch
from torchtitan.experiments.torchax.moe import MoE
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from torchtitan.models.qwen3.model.model import Qwen3Model as Qwen3ModelBase
from torchtitan.models.qwen3.model.model import TransformerBlock as Qwen3TransformerBlockBase


class TransformerBlock(Qwen3TransformerBlockBase):
  """TransformerBlock Module.

  Overrides the MoE module to use the Torchax MoE module.
  """

  def __init__(self, layer_id: int, model_args: Qwen3ModelArgs):
    super().__init__(layer_id, model_args)

    self.moe_enabled = model_args.moe_enabled
    if self.moe_enabled:
      self.moe = MoE(
          model_args.moe_args,
          dim=model_args.dim,
          hidden_dim=model_args.moe_inter_dim,
      )


class Qwen3Model(Qwen3ModelBase):
  """Qwen3Model Module."""

  def __init__(self, model_args: Qwen3ModelArgs):
    super().__init__(model_args)
    self.layers = torch.nn.ModuleDict()
    for layer_id in range(model_args.n_layers):
      self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)
