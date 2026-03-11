# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torchtitan.experiments.torchax.moe import MoE
from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
from torchtitan.models.deepseek_v3.model.model import DeepSeekV3Model as DeepSeekV3ModelBase
from torchtitan.models.deepseek_v3.model.model import TransformerBlock as DeepSeekV3TransformerBlockBase


class TransformerBlock(DeepSeekV3TransformerBlockBase):
  """TransformerBlock Module.

  Overrides the MoE module to use the Torchax MoE module.
  """

  def __init__(self, layer_id: int, model_args: DeepSeekV3ModelArgs):

    super().__init__(layer_id, model_args)

    self.moe_enabled = layer_id >= model_args.n_dense_layers
    if self.moe_enabled:
      self.moe = MoE(
          model_args.moe_args,
          dim=model_args.dim,
          hidden_dim=model_args.moe_inter_dim,
      )


class DeepSeekV3Model(DeepSeekV3ModelBase):
  """DeepSeek-V3 Transformer model with attention and feed-forward layers."""

  def __init__(self, model_args: DeepSeekV3ModelArgs):
    super().__init__(model_args)

    self.layers = torch.nn.ModuleDict()
    for layer_id in range(model_args.n_layers):
      self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)
