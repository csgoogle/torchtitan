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

class DummyActor:
    pass

if 'monarch' not in sys.modules:
    import types
    sys.modules['torchstore'] = MagicMock()
    
    import os
    real_file = os.path.abspath(__file__)
    # Create real module objects for monarch instead of MagicMocks so they are serializable by reference
    monarch = types.ModuleType("monarch")
    monarch.__file__ = real_file
    monarch_actor = types.ModuleType("monarch.actor")
    monarch_actor.__file__ = real_file
    monarch_spmd = types.ModuleType("monarch.spmd")
    monarch_spmd.__file__ = real_file
    monarch_rdma = types.ModuleType("monarch.rdma")
    monarch_rdma.__file__ = real_file
    
    sys.modules['monarch'] = monarch
    sys.modules['monarch.actor'] = monarch_actor
    sys.modules['monarch.spmd'] = monarch_spmd
    sys.modules['monarch.rdma'] = monarch_rdma

    def dummy_endpoint(func):
        return func

    async def dummy_setup_torch_elastic_env_async(*args, **kwargs):
        pass

    monarch_actor.Actor = DummyActor
    monarch_actor.endpoint = dummy_endpoint
    monarch_actor.this_host = MagicMock()
    
    monarch_rdma.is_rdma_available = lambda: False
    monarch_spmd.setup_torch_elastic_env_async = dummy_setup_torch_elastic_env_async




