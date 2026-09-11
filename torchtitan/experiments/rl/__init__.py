# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified approach for running TorchTitan models with vLLM inference.

To register TorchTitan models with vLLM:
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.experiments.rl.models.vllm_registry import register_to_vllm

    # Standalone inference (loads HF weights):
    register_to_vllm(
        model_spec,
        parallelism=parallelism_config,
        compile_config=compile_config,
        checkpoint_config=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_path="/path/to/hf/checkpoint",
        ),
    )

    # RL loop (skip HF loading, weights from TorchStore):
    register_to_vllm(
        model_spec,
        parallelism=parallelism_config,
        compile_config=compile_config,
        checkpoint_config=CheckpointManager.Config(enable=False),
    )
"""

from torchtitan.experiments.rl.runtime_env import apply_env_defaults


# ``python -m torchtitan.experiments.rl.train`` executes this package initializer
# before train.py. Apply the defaults before anything below imports
# vllm_wrapper, which imports torch.
apply_env_defaults()


# To prevent importing the heavy 'vllm' library (and hitting ImportErrors on
# hardware environments like standard TPUs where 'vllm' is not installed),
# the re-exports below are resolved lazily, only when explicitly requested.
# Importing other RL submodules is therefore safe without vllm.
def __getattr__(name: str):
    if name == "VLLMModelWrapper":
        from torchtitan.experiments.rl.models.vllm_wrapper import VLLMModelWrapper

        return VLLMModelWrapper
    if name == "register_to_vllm":
        from torchtitan.experiments.rl.models.vllm_registry import register_to_vllm

        return register_to_vllm
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "VLLMModelWrapper",
    "register_to_vllm",  # Export register function for manual use
]
