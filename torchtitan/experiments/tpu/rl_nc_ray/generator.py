"""
TPU-specific Ray Actor for vLLM Generation.

Uses composition rather than inheriting from `VLLMGenerator` to defer `vllm` imports until 
the Ray worker initializes, preventing the orchestrator from eagerly locking the TPU PJRT client.
Replaces Monarch/Torchstore weight sync with `load_model_state_dict()`, receiving weights 
via Ray's object store and manually copying them into the vLLM model's `DTensor` parameters.

TODO (jialei): use the latest torchtpu-vllm, and confirm each of those TPU hacks is still needed. 
"""

import asyncio
import json
import logging
import os
import pickle
import sys
import types

import ray
import torch
import torch_tpu
import torchtitan.experiments.tpu.rl_nc_ray  # Ensure mock runs first

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate
from torchtitan.experiments.rl.actors.generator import VLLMGenerator
from torchtitan.experiments.rl.models.vllm_registry import VLLM_MODEL_NAME
from torchtitan.experiments.rl.models.vllm_registry import register_model_to_vllm_model_registry
from torchtitan.tools.utils import get_device_type

logger = logging.getLogger(__name__)

# TPU Hack Constants
VLLM_ARGS_PATH = "/tmp/torchtitan_vllm_args.pkl" if os.environ.get("LOCAL_VM_RUN") == "1" else "/data/jialei/torchtitan_vllm_args.pkl"

@ray.remote
class RayVLLMGenerator:
    def __init__(self, config, *, model_spec, model_path, compile_config, max_num_seqs):
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")  # different port from trainer just in case
        
        import vllm

        try:
            # [TPU HACK 10]: `torchtpu-vllm`'s worker blindly imports `init_cached_hf_modules`,
            # which was removed/moved in newer vLLM codebase versions, instantly crashing workers.
            if "vllm.utils.import_utils" not in sys.modules:
                sys.modules["vllm.utils.import_utils"] = types.ModuleType("import_utils")
            sys.modules["vllm.utils.import_utils"].init_cached_hf_modules = lambda: None
            
            from tpu_inference.executors import ray_distributed_executor
            
            # [TPU HACK 1]: Force MATH attention backend on driver.
            # TPU v6e (v6e) has limited VMEM. If the Pallas attention kernel requests 
            # exactly 128MB, XLA compilation crashes. Forcing the MATH backend avoids this.
            try:
                import tpu_inference.platforms.tpu_platform as tpu_platform
                from vllm.config import AttentionConfig
                from vllm.v1.attention.backends.registry import AttentionBackendEnum
                
                orig_wrap = tpu_platform.TpuPlatform.wrap_engine_kwargs
                def patched_wrap(self, engine_kwargs):
                    orig_wrap(self, engine_kwargs)
                    if "attention_config" in engine_kwargs:
                        engine_kwargs["attention_config"].backend = AttentionBackendEnum.MATH
                tpu_platform.TpuPlatform.wrap_engine_kwargs = patched_wrap
            except Exception as e:
                logger.warning(f"Failed to patch TPUPlatform.wrap_engine_kwargs: {e}")

            # [TPU HACK 2]: Add missing 4-chip and 8-chip topologies to torchtpu-vllm.
            # The library hardcodes 16-chip multi-host topologies. We must inject our
            # smaller slice topologies for it to correctly dispatch PJRT clients.
            ray_distributed_executor.TPU_TOPOLOGY_MAP[4] = "2,2,1"
            ray_distributed_executor.TPU_TOPOLOGY_MAP[8] = "2,4,1"

            # [TPU HACK 3]: Prevent torchtpu-vllm from claiming the entire Ray cluster.
            # torchtpu-vllm assumes it is the only application running on the Ray cluster.
            # It queries `ray.nodes()` and overwrites `TORCH_TPU_TOPOLOGY` and slice builder 
            # addresses with ALL available TPUs (16 chips), stealing the Trainer's 8 chips.
            # We freeze the env vars to the 8 chips provided by grpo.py.
            original_driver_environ_setitem = os.environ.__class__.__setitem__
            def patched_driver_environ_setitem(self, key, value):
                if key in ("TORCH_TPU_SLICEBUILDER_ADDRESSES", "TPU_PROCESS_ADDRESSES", 
                           "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS", "TORCH_TPU_TOPOLOGY"):
                    value = os.environ.get(key, value)
                elif key == "LIBTPU_INIT_ARGS":
                    value = value.replace("--deepsea_chip_config_name=megachip_tccontrol", "")
                original_driver_environ_setitem(self, key, value)
            os.environ.__class__.__setitem__ = patched_driver_environ_setitem
            
            # [TPU HACK 4]: Hide the Trainer's TPU nodes from vLLM's RayExecutor.
            # vLLM blindly spawns Ray workers on all `ray.nodes()` that have "TPU" resources.
            # We patch `ray.nodes()` to only return the 8 IPs allocated to the Sampler.
            allowed_ips = set(addr.split(':')[0] for addr in os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES", "").split(',') if addr)
            if allowed_ips:
                original_ray_nodes = ray.nodes
                def patched_ray_nodes(*args, **kwargs):
                    nodes = original_ray_nodes(*args, **kwargs)
                    return [n for n in nodes if n.get("NodeManagerAddress") in allowed_ips or "TPU" not in n.get("Resources", {})]
                ray.nodes = patched_ray_nodes
                
                original_cluster_resources = ray.cluster_resources
                def patched_cluster_resources(*args, **kwargs):
                    res = original_cluster_resources(*args, **kwargs)
                    res["TPU"] = int(os.environ.get("WORLD_SIZE", "8"))
                    return res
                ray.cluster_resources = patched_cluster_resources

            OriginalRayWorkerWrapper = ray_distributed_executor.RayWorkerWrapper
            original_init_worker = OriginalRayWorkerWrapper.init_worker
            
            # [TPU HACK 5]: Prevent vLLM from attempting to set CUDA devices on TPU workers.
            # vLLM's `RayWorkerWrapper.setup_device_if_necessary` explicitly accesses 
            # `self.worker.device`, which throws an AttributeError on `TPUWorker` and crashes the job.
            def patched_setup_device_if_necessary(self):
                if not getattr(self, "compiled_dag_cuda_device_set", False):
                    try:
                        from vllm.platforms import current_platform
                        if current_platform.is_cuda() and hasattr(self.worker, "device") and self.worker.device is not None:
                            current_platform.set_device(self.worker.device)
                    except Exception:
                        pass
                    self.compiled_dag_cuda_device_set = True
            OriginalRayWorkerWrapper.setup_device_if_necessary = patched_setup_device_if_necessary
            
            # This method is executed in the isolated Ray worker processes spawned by vLLM.
            def patched_init_worker(self, *args, **kwargs):
                # [TPU HACK 6]: `torchtpu-vllm`'s worker blindly imports `init_cached_hf_modules`,
                # which was removed/moved in newer vLLM codebase versions, instantly crashing workers.
                if "vllm.utils.import_utils" not in sys.modules:
                    sys.modules["vllm.utils.import_utils"] = types.ModuleType("import_utils")
                sys.modules["vllm.utils.import_utils"].init_cached_hf_modules = lambda: None

                # [TPU HACK 7]: When loading dummy formats to save memory, vLLM attempts to randomize 
                # weights in Python logic to prevent NaN loss. On TPU, running eager seq randomization 
                # over millions of parameters triggers thousands of un-fused XLA compilations.
                # This causes a 20+ minute hang! We mock out the dummy initializer here.
                import vllm.model_executor.model_loader.weight_utils
                vllm.model_executor.model_loader.weight_utils.initialize_dummy_weights = lambda *args, **kwargs: None
                import vllm.model_executor.model_loader.dummy_loader
                vllm.model_executor.model_loader.dummy_loader.initialize_dummy_weights = lambda *args, **kwargs: None
                
                # [TPU HACK 8]: Implement multi-host remote weight injection.
                # The orchestrator cannot use `driver_worker.get_model()` to sync weights because 
                # the driver worker does not exist when using `TPU_MULTIHOST_BACKEND="ray"`.
                # We inject this method directly into the Ray workers.
                def load_weights_from_state_dict_on_worker(self, state_dict):
                    vllm_model = self.worker.model_runner.model
                    
                    clean_state_dict = {}
                    for name, source_p in state_dict.items():
                        clean_name = name.replace("_fsdp_wrapped_module.", "") \
                                         .replace("_checkpoint_wrapped_module.", "") \
                                         .replace("module.", "")
                        
                        if hasattr(source_p, "full_tensor"):
                            val = source_p.full_tensor()
                        else:
                            val = source_p
                        clean_state_dict[clean_name] = val
                    
                    model_sd = vllm_model.model.state_dict()
                    wrapped_sd = {}
                    missing_keys = []
                    
                    for k, v in clean_state_dict.items():
                        # If torchtitan model keys do not have 'model.' prefix but the state dict does
                        alt_k = k
                        if k.startswith("model.") and k not in model_sd and k[6:] in model_sd:
                            alt_k = k[6:]
                        elif not k.startswith("model.") and k not in model_sd and f"model.{k}" in model_sd:
                            alt_k = f"model.{k}"
                            
                        if alt_k in model_sd:
                            if isinstance(model_sd[alt_k], DTensor):
                                target_dtensor = model_sd[alt_k]
                                device_mesh = target_dtensor.device_mesh
                                wrapped_sd[alt_k] = DTensor.from_local(
                                    v.to(device_mesh.device_type),
                                    device_mesh=device_mesh,
                                    placements=[Replicate()],
                                )
                            else:
                                wrapped_sd[alt_k] = v
                        else:
                            wrapped_sd[k] = v
                            
                    vllm_model.load_weights_from_state_dict(wrapped_sd)
                    
                    torch_tpu._internal.sync.synchronize(wait=True)
                    print(f"@@@ Worker: Loaded {len(wrapped_sd)} model state dicts via vllm_model.load_weights_from_state_dict")
                    return len(wrapped_sd)
                    
                self.__class__.load_weights_from_state_dict_on_worker = load_weights_from_state_dict_on_worker
                
                if os.path.exists(VLLM_ARGS_PATH):
                    try:
                        with open(VLLM_ARGS_PATH, "rb") as f:
                            ms_spec, c_config = pickle.load(f)
                        register_model_to_vllm_model_registry(ms_spec, c_config)
                    except Exception as e:
                        logger.error(f"Failed to load TorchTitan args: {e}")
                
                # Prevent Dynamo recompilations due to grad_mode toggling 
                torch.set_grad_enabled(False)
                
                return original_init_worker(self, *args, **kwargs)
                
            OriginalRayWorkerWrapper.init_worker = patched_init_worker
        except ImportError:
            pass

        # [TPU PATCH 9]: Teleport configs to Ray workers
        # vLLM spawns isolated Ray worker processes that don't inherit the main process's
        # model registry. We dump `model_spec` and `compile_config` here so the workers can
        # load them and run `register_model_to_vllm_model_registry` inside their own processes.
        # Note: We use a local file `VLLM_ARGS_PATH` for simplicity.
        # For multi-host TPU execution, this should be updated to use Ray's object store or KV store.
        with open(VLLM_ARGS_PATH, "wb") as f:
            pickle.dump((model_spec, compile_config), f)

        self.generator = VLLMGenerator(
            config,
            model_spec=model_spec,
            model_path=model_path,
            compile_config=compile_config,
            max_num_seqs=max_num_seqs
        )

    def generate(self, *args, **kwargs):
        return asyncio.run(self.generator.generate(*args, **kwargs))

    def load_model_state_dict(self, version: int, state_dict: dict) -> None:
        """Ray-native weight injection mirroring reference script."""
        print(f"@@@ Generator: Received state dict for version {version}. Injecting...")
        
        # If weight tying is enabled, explicitly add lm_head for syncing
        tok_key = next((k for k in state_dict.keys() if "tok_embeddings.weight" in k), None)
        lm_key = next((k for k in state_dict.keys() if "lm_head.weight" in k), None)
        if tok_key and not lm_key:
            state_dict[tok_key.replace("tok_embeddings", "lm_head")] = state_dict[tok_key]

        executor = self.generator._engine.model_executor
        if hasattr(executor, "workers"):
            sd_ref = ray.put(state_dict)
            refs = []
            for worker in executor.workers:
                refs.append(worker.execute_method.remote("load_weights_from_state_dict_on_worker", sd_ref))
            ray.get(refs)
        else:
            vllm_model = executor.driver_worker.get_model()
            target_sd = dict(vllm_model.named_parameters())
            with torch.no_grad():
                for name, source_p in state_dict.items():
                    clean_name = name.replace("_fsdp_wrapped_module.", "") \
                                     .replace("_checkpoint_wrapped_module.", "") \
                                     .replace("module.", "")
                    target_name = f"model.{clean_name}"
                    
                    if target_name in target_sd:
                        target_t = target_sd[target_name]
                        val = source_p
                        if hasattr(val, "full_tensor"):
                            val = val.full_tensor()
                        if target_t.shape == val.shape:
                            target_t.copy_(val.to(target_t.device))
                        else:
                            logger.warning(f"Shape mismatch for {target_name}: {target_t.shape} != {val.shape}")
                    elif clean_name in target_sd:
                        target_name = clean_name
                        target_t = target_sd[target_name]
                        val = source_p
                        if hasattr(val, "full_tensor"):
                            val = val.full_tensor()
                        if target_t.shape == val.shape:
                            target_t.copy_(val.to(target_t.device))
                        else:
                            logger.warning(f"Shape mismatch for {target_name}: {target_t.shape} != {val.shape}")
                        
        self.generator.policy_version = version
        self.generator._engine.reset_prefix_cache()
        
        print("@@@ Generator: Calling torch_tpu._internal.sync.synchronize()...")
        torch_tpu._internal.sync.synchronize(wait=True)
        print(
            f"@@@ Generator: Loaded model state dicts for policy v{version}"
        )
