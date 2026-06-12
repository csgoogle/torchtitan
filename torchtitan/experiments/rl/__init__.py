# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified approach for running TorchTitan models with vLLM inference.

To register TorchTitan models with vLLM:
    from torchtitan.experiments.rl.models.vllm_registry import register_model_to_vllm_model_registry
    register_model_to_vllm_model_registry(model_spec)
"""

from torchtitan.tools.utils import get_device_type
from torchtitan.experiments.rl.models.vllm_registry import (
    register_model_to_vllm_model_registry,
)
if get_device_type() == "tpu":
    # With TPU, if we are NOT using vllm sampler, we skip vllm import
    try:
        from torchtitan.experiments.rl.models.vllm_wrapper import TorchTitanVLLMModelWrapper
    except:
        pass
else:
    from torchtitan.experiments.rl.models.vllm_wrapper import TorchTitanVLLMModelWrapper


__all__ = [
    "TorchTitanVLLMModelWrapper",
    "register_model_to_vllm_model_registry",  # Export register function for manual use
]
