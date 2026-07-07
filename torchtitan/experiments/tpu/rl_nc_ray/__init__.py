"""
TPU/Ray specific initialization and dependency mocking.

Dynamically mocks `monarch` and `torchstore` in `sys.modules` to allow inheriting from 
core torchtitan RL classes (`RLTrainer`, `PolicyTrainer`, `VLLMGenerator`) without 
pulling in Monarch's networking or RDMA dependencies, which conflict with Ray and TPU.
"""

import os
import sys
from unittest.mock import MagicMock

# =====================================================================
# torch_tpu CPU Mocking
# =====================================================================
# [TPU HACK]: If running on CPU (e.g., PolicyTrainer in local non-colocated RL),
# importing torch_tpu can trigger PJRT initialization and grab TPU resources,
# causing conflicts with the active TPU generator (vLLM).
# What didn't work: Having a top-level "import torch_tpu" in trainer.py and 
# weight_transfer_utils.py. Because those processes execute on CPU, importing
# torch_tpu initializes the runtime, attempting to lock /dev/vfio/0 which was 
# already held by vLLM. This resulted in "Device or resource busy" errors.
# We dynamically mock torch_tpu with a no-op synchronize if device is cpu.
# TODO: remove HACK once stable torch_tpu/PJRT releases do not initialize 
# hardware on import, or once imports of torch_tpu are lazily/conditionally 
# deferred across the entire torchtitan trainer/utils modules.
# =====================================================================
if 'torch_tpu' not in sys.modules:
    target_device = os.environ.get("TORCHTITAN_DEVICE_TYPE", "cpu")
    if target_device == "cpu" or os.environ.get("TORCH_TPU_MOCK_CPU", "0") == "1":
        # [TPU HACK]: Disable device backend autoloading to prevent PyTorch from trying to load
        # the real torch_tpu via entrypoints, which initializes PJRT and attempts to lock TPU hardware
        # already held by the vLLM generator. We delete it immediately after importing torch so that
        # spawned Ray workers do not inherit this and fail to load torch_tpu.
        # TODO: Remove HACK once stable torch_tpu does not initialize hardware on import.
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        import torch
        del os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"]
        
        import types
        mock_torch_tpu = types.ModuleType('torch_tpu')
        mock_torch_tpu.__path__ = []
        sys.modules['torch_tpu'] = mock_torch_tpu
        
        mock_internal = types.ModuleType('torch_tpu._internal')
        mock_internal.__path__ = []
        class MockSync:
            @staticmethod
            def synchronize(wait=True):
                pass
        mock_internal.sync = MockSync()
        sys.modules['torch_tpu._internal'] = mock_internal
        
        print("@@@ Mock torch_tpu installed for CPU run")


# =====================================================================
# Monarch / Torchstore Mocking for Ray
# =====================================================================
# The PolicyTrainer and VLLMGenerator from torchtitan.experiments.rl.actors 
# inherit from monarch.actor.Actor and use torchstore. 
# Since we are using Ray and do not want Monarch dependencies or its 
# Actor metaclass/endpoints to interfere with Ray, we dynamically mock 
# them out before importing the RL actors.
#
# This allows us to cleanly inherit from the torchtitan classes without 
# requiring Monarch or torchstore to be installed, and it completely 
# disarms the monarch Actor base class.
# =====================================================================

if 'monarch' not in sys.modules:
    sys.modules['torchstore'] = MagicMock()
    sys.modules['monarch'] = MagicMock()
    sys.modules['monarch.actor'] = MagicMock()
    sys.modules['monarch.spmd'] = MagicMock()
    sys.modules['monarch.rdma'] = MagicMock()

    class DummyActor:
        pass

    def dummy_endpoint(func):
        return func

    sys.modules['monarch.actor'].Actor = DummyActor
    sys.modules['monarch.actor'].endpoint = dummy_endpoint
    sys.modules['monarch.actor'].this_host = MagicMock()

