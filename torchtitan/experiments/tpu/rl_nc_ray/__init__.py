"""
TPU/Ray specific initialization and dependency mocking.

Dynamically mocks `monarch` and `torchstore` in `sys.modules` to allow inheriting from 
core torchtitan RL classes (`RLTrainer`, `PolicyTrainer`, `VLLMGenerator`) without 
pulling in Monarch's networking or RDMA dependencies, which conflict with Ray and TPU.
"""

import sys
import copyreg
import torch.utils._config_module
from unittest.mock import MagicMock
import logging

import importlib

# Register custom reducer on PyTorch ConfigModule base class so that cloudpickle / Ray 
# can serialize RayTPUPolicyTrainer without failing on dynamically modified PyTorch config modules.
torch.utils._config_module.ConfigModule.__reduce__ = lambda self: (
    importlib.import_module,
    (self.__module__,),
)

logger = logging.getLogger(__name__)

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


# =====================================================================
# vLLM GPU Codegen Mocking for TPU/CPU
# =====================================================================
# When torchtitan's vllm_wrapper.py imports vllm.compilation.codegen, it will
# crash on TPU or CPU-only container environments because Triton or GPU-specific
# codegen tools are absent. This shim preemptively injects codegen as None into
# the vLLM package hierarchy if it fails to import.
# =====================================================================
try:
    import sys
    import types
    import vllm

    # Try importing the real vllm.compilation.codegen first.
    # Only fallback to a mock shim if vllm codegen fails to import.
    try:
        import vllm.compilation.codegen
    except (ImportError, AttributeError):
        vllm_codegen = types.ModuleType("vllm.compilation.codegen")
        vllm_codegen._node_ref = lambda arg: arg
        try:
            import vllm.compilation
        except (ImportError, AttributeError) as e:
            logger.debug("Failed to import vllm.compilation, fallback mocking: %s", e)
            vllm_compilation = types.ModuleType("vllm.compilation")
            vllm_compilation.__path__ = []
            sys.modules["vllm.compilation"] = vllm_compilation
            setattr(vllm, "compilation", vllm_compilation)
        vllm.compilation.codegen = vllm_codegen
        sys.modules["vllm.compilation.codegen"] = vllm_codegen

    # ---------------------------------------------------------------------------
    # vLLM support_torch_compile Decorator Override for TPU
    # ---------------------------------------------------------------------------
    # The support_torch_compile decorator in vllm is GPU-specific and fails on
    # TPU due to Triton/GPU compilation assumptions. We intercept it dynamically
    # and replace it with a dummy decorator that returns the decorated class untouched.
    # ---------------------------------------------------------------------------
    def dummy_support_torch_compile(*args, **kwargs):
        """A mock decorator replacing vllm's support_torch_compile.
        
        Supports two modes of usage:
        1. Used directly: @support_torch_compile -> returns the class as-is.
        2. Used with arguments: @support_torch_compile(flag=True) -> returns a lambda 
           that returns the class as-is.
        """
        if len(args) == 1 and isinstance(args[0], type):
            return args[0]
        return lambda cls: cls

    try:
        import vllm.compilation.decorators as decorators
        decorators.support_torch_compile = dummy_support_torch_compile
    except ImportError as e:
        logger.debug("Failed to import vllm.compilation.decorators, fallback mocking: %s", e)
        # Create a mock decorators module if not found on the system
        vllm_decorators = types.ModuleType("vllm.compilation.decorators")
        vllm_decorators.support_torch_compile = dummy_support_torch_compile
        sys.modules["vllm.compilation.decorators"] = vllm_decorators
        if hasattr(vllm, "compilation"):
            vllm.compilation.decorators = vllm_decorators

    # ---------------------------------------------------------------------------
    # create_torchtitan_config_from_vllm_config TPU tp=1 override
    # ---------------------------------------------------------------------------
    # On TPU with tp=1, each rank runs an independent single-device vLLM instance. 
    # The generator processes are initialized within a shared distributed group. 
    # Thus, Torch's dist.get_world_size() returns the group's total 
    # world size (e.g., 4 or 8), triggering an assertion check failure in ParallelDims 
    # (dp * tp * pp != world_size). We dynamically intercept the config creation and 
    # temporarily mock dist.get_world_size() to return 1 if tp=1.
    # ---------------------------------------------------------------------------
    try:
        import torchtitan.experiments.rl.models.vllm_wrapper as vllm_wrapper
        original_create_config = vllm_wrapper.create_torchtitan_config_from_vllm_config

        def patched_create_config(vllm_config):
            """Intercepts and modifies config translation to prevent world size assertion errors."""
            if vllm_config.parallel_config.tensor_parallel_size == 1:
                # If tp=1, bypass ParallelDims' constraint check that expects the 
                # total world size to match (dp * tp * pp). We fake get_world_size()
                # to return 1 during config parsing.
                import torch.distributed as dist
                orig_get_world_size = dist.get_world_size
                dist.get_world_size = lambda: 1
                try:
                    parallel_dims, parallelism = original_create_config(vllm_config)
                finally:
                    # Always restore the original get_world_size function to avoid
                    # corrupting downstream training logic.
                    dist.get_world_size = orig_get_world_size
            else:
                parallel_dims, parallelism = original_create_config(vllm_config)

            # If TP > 1 on TPU, initialize the distributed communication environment
            # and build the device mesh grid before the model parallelization runs.
            if parallel_dims.tp_enabled:
                import torch.distributed as dist
                if not dist.is_initialized():
                    import os
                    # Initialize PyTorch distributed with XLA's specialized TPU backend.
                    dist.init_process_group(
                        backend="tpu_dist",
                        rank=int(os.environ.get("RANK", 0)),
                        world_size=int(os.environ.get("WORLD_SIZE", 1))
                    )
                # Build the distributed device mesh (e.g., 2D/3D grids) on the TPU slice.
                parallel_dims.build_mesh()

            return parallel_dims, parallelism

        vllm_wrapper.create_torchtitan_config_from_vllm_config = patched_create_config

        # Stub out _initial_load_weights since TPU weights will be initialized/loaded
        # through alternative TPU-native paths or random initialization.
        vllm_wrapper.TorchTitanVLLMModelWrapper._initial_load_weights = lambda self, *args, **kwargs: None

        # Patched __init__ of TorchTitanVLLMModelWrapper to support meta device initialization
        # and on-device materialization dynamically on TPU if TP > 1.
        original_init = vllm_wrapper.TorchTitanVLLMModelWrapper.__init__

        def patched_init(self, *, model_spec, vllm_config, **kwargs):
            """Modified constructor supporting meta device instantiation on TPUs.
            
            This prevents initial OOM (Out Of Memory) issues by initializing the
            unsharded model on PyTorch's 'meta' device, parallelizing it (sharding),
            and then materializing the resulting sharded layers onto physical TPU devices.
            """
            
            # Bump Dynamo's recompile limit to 100 on TPU to accommodate
            # dynamic shape changes during RL generation/training.
            compile_config = kwargs.get("compile_config")
            if compile_config and compile_config.enable:
                import torch
                torch._dynamo.config.recompile_limit = 100

            tp_enabled = vllm_config.parallel_config.tensor_parallel_size > 1
            if tp_enabled:
                import torch
                model_class = model_spec.model.__class__
                original_build = model_class.build
                
                # 1. Intercept the build method to instantiate model parameters on "meta" device.
                def patched_build(*b_args, **b_kwargs):
                    with torch.device("meta"):
                        return original_build(*b_args, **b_kwargs)
                
                model_class.build = patched_build

                # 2. Intercept the parallelization step to shard first, then materialize.
                original_parallelize = model_spec.parallelize_fn
                def patched_parallelize(*p_args, **p_kwargs):
                    # Shard the meta-device model representation.
                    sharded_model = original_parallelize(*p_args, **p_kwargs)
                    device = vllm_config.device_config.device
                    # Materialize empty (uninitialized) tensor storage on the actual TPU device.
                    sharded_model = sharded_model.to_empty(device=device)
                    # Build and instantiate Rotary Position Embeddings (RoPE) for TPU.
                    extended_rope = model_spec.model.rope.build()
                    sharded_model.freqs_cis = extended_rope.cache.to(
                        device=device,
                        dtype=sharded_model.tok_embeddings.weight.dtype
                    )
                    return sharded_model

                model_spec.parallelize_fn = patched_parallelize

                try:
                    return original_init(self, model_spec=model_spec, vllm_config=vllm_config, **kwargs)
                finally:
                    model_class.build = original_build
                    model_spec.parallelize_fn = original_parallelize
            else:
                return original_init(self, model_spec=model_spec, vllm_config=vllm_config, **kwargs)

        vllm_wrapper.TorchTitanVLLMModelWrapper.__init__ = patched_init

        # ---------------------------------------------------------------------------
        # [TPU HACK 11]: Intercept and flatten 3D attention inputs to 2D for Pallas kernel
        # compatibility. This completely decouples TPU-specific tensor shaping from attention.py!
        # ---------------------------------------------------------------------------
        from vllm.model_executor.layers.attention import Attention
        original_attention_forward = Attention.forward
        def patched_attention_forward(self, query, key, value, *args, **kwargs):
            if len(query.shape) == 3:
                query = query.reshape(query.shape[0], -1)
            if len(key.shape) == 3:
                key = key.reshape(key.shape[0], -1)
            if len(value.shape) == 3:
                value = value.reshape(value.shape[0], -1)
            return original_attention_forward(self, query, key, value, *args, **kwargs)
        Attention.forward = patched_attention_forward

    except Exception as e:
        logger.debug("Failed to apply vllm model wrapper patches: %s", e)
except ImportError as e:
    logger.debug("vllm is not installed on this environment: %s", e)

