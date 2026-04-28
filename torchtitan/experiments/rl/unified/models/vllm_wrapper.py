# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torchtitan.models.qwen3.model.model import Qwen3Model
from torchtitan.experiments.rl.vllm_compat.models.attention import (
    VLLMCompatibleFlashAttention,
)

class TorchTitanVLLMModelWrapper(nn.Module):
    """
    Minimal wrapper for TorchTitan models on TPU.
    """
    def __init__(self, model_args):
        super().__init__()
        self.model = Qwen3Model(model_args)
        
        # Optimized attention path for TPU
        for layer in self.model.layers.values():
            if hasattr(layer, "attention"):
                layer.attention.inner_attention = VLLMCompatibleFlashAttention()
                
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
