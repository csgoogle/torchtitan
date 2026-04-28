# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified model folder for RL experiments.
"""

from .infra.parallelism_utils import create_parallel_dims_from_vllm_config
from .models.vllm_wrapper import TorchTitanVLLMModelWrapper

__all__ = [
    "TorchTitanVLLMModelWrapper",
    "create_parallel_dims_from_vllm_config",
]
