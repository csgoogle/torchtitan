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

from torchtitan.experiments.rl.models.vllm_registry import (
    register_model_to_vllm_model_registry,
)

# To prevent importing the heavy 'vllm' library (and hitting ImportErrors on
# hardware environments like standard TPUs where 'vllm' is not installed),
# we lazily load 'TorchTitanVLLMModelWrapper' only when explicitly requested.
# Importing other RL submodules (e.g., sum_digits.py) is therefore completely safe.
def __getattr__(name: str):
    if name == "TorchTitanVLLMModelWrapper":
        from torchtitan.experiments.rl.models.vllm_wrapper import (
            TorchTitanVLLMModelWrapper,
        )

        return TorchTitanVLLMModelWrapper
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "TorchTitanVLLMModelWrapper",
    "register_model_to_vllm_model_registry",  # Export register function for manual use
]
