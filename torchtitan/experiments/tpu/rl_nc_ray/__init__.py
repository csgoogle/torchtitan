"""
TPU/Ray specific initialization and dependency mocking.

Dynamically mocks `monarch` and `torchstore` in `sys.modules` to allow inheriting from 
core torchtitan RL classes (`RLTrainer`, `PolicyTrainer`, `VLLMGenerator`) without 
pulling in Monarch's networking or RDMA dependencies, which conflict with Ray and TPU.
"""

import sys
from unittest.mock import MagicMock

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
