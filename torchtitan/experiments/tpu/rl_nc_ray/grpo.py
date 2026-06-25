# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Ray-based RL training orchestrator for TPU.

Implements `RayRLTrainer`, which inherits core GRPO logic from the base GPU `RLTrainer` 
but overrides `setup()` to spawn Ray actors instead of Monarch procs. Uses proxy classes 
(`RayTrainerProxy`, `RayGeneratorProxy`) to transparently translate Monarch `.call().get()` 
RPCs into Ray `.remote()` and `ray.get()` futures. Pins the trainer to CPU and the 
generator to TPU to bypass PJRT client deadlock limitations.

TODO (jialei): move the ray hacks to a separate file.
"""

import asyncio
import os
import sys
import typing
from dataclasses import dataclass

import ray
import ray.util.scheduling_strategies

import torchtitan.experiments.tpu.rl_nc_ray  # Ensure mock runs first

from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.rl.grpo import RLTrainer
from torchtitan.experiments.tpu.rl_nc_ray.trainer import RayTPUPolicyTrainer
from torchtitan.experiments.tpu.rl_nc_ray.generator import RayVLLMGenerator
from torchtitan.tools.logging import init_logger


class FutureProxy:
    """Mocks the Monarch Future returned by .call()"""
    def __init__(self, obj_refs):
        self.obj_refs = obj_refs if isinstance(obj_refs, list) else [obj_refs]

    def get(self):
        vals = ray.get(self.obj_refs)
        val = vals[0] if vals else None
        
        class ItemProxy:
            def item(self, **kwargs):
                return val
            # Some operations (like pull_model_state_dict) just return None
            def __getattr__(self, name):
                return getattr(val, name)
        
        # If the value is a dict/list/tuple, we still might need to support .item() for metrics
        if isinstance(val, (dict, list, tuple, float, int, str, bool)) or val is None:
            class Wrapper:
                def __init__(self, v):
                    self.v = v
                def item(self, **kwargs):
                    return self.v
            return Wrapper(val)
        return ItemProxy()


class RayTrainerProxy:
    """Mocks the Monarch Actor wrapper for the Trainer"""
    def __init__(self, ray_trainers):
        self.ray_trainers = ray_trainers if isinstance(ray_trainers, list) else [ray_trainers]

    def __getattr__(self, name):
        class MethodProxy:
            def __init__(self, proxy):
                self.proxy = proxy
            def call(self, *args, **kwargs):
                if name == "push_model_state_dict":
                    # We skip torchstore, so pushing model state dict just means
                    # returning an empty future (the actual state dict is pulled later)
                    return FutureProxy([ray.put(None)] * len(self.proxy.ray_trainers))
                
                futures = []
                if name == "forward_backward":
                    batches = args[0]
                    for i, trainer in enumerate(self.proxy.ray_trainers):
                        method = getattr(trainer, name)
                        # Pass the ENTIRE batches list so the trainer can slice it via self.dp_rank
                        futures.append(method.remote(batches))
                else:
                    for trainer in self.proxy.ray_trainers:
                        method = getattr(trainer, name)
                        futures.append(method.remote(*args, **kwargs))
                return FutureProxy(futures)
        return MethodProxy(self)


class RayGeneratorProxy:
    """Mocks the Monarch Actor wrapper for the Generator"""
    def __init__(self, ray_generator, ray_trainers):
        self.ray_generator = ray_generator
        self.ray_trainers = ray_trainers if isinstance(ray_trainers, list) else [ray_trainers]

    def __getattr__(self, name):
        class MethodProxy:
            def __init__(self, generator, trainers):
                self.generator = generator
                self.trainers = trainers
            def call(self, *args, **kwargs):
                if name == "pull_model_state_dict":
                    version = args[0]
                    # We bypass torchstore and use Ray object store directly
                    # Call on all trainers to participate in collective operation!
                    refs = []
                    for trainer in self.trainers:
                        refs.append(getattr(trainer, "get_model_state_dict").remote())
                    
                    return FutureProxy([self.generator.load_model_state_dict.remote(version, refs[0])])
                method = getattr(self.generator, name)
                return FutureProxy([method.remote(*args, **kwargs)])
        return MethodProxy(self.ray_generator, self.ray_trainers)


class RayRLTrainer(RLTrainer):
    """Subclasses the GPU RLTrainer, overriding ONLY setup() to inject Ray."""

    async def setup(self, **kwargs):
        config = self.config
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        all_tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
        
        if all_tpu_nodes:
            slices = {}
            for n in all_tpu_nodes:
                slice_name = n.get("Labels", {}).get("ray.io/tpu-slice-name", "default")
                slices.setdefault(slice_name, []).append(n)
            
            slice_names = list(slices.keys())
            trainer_nodes_info = slices[slice_names[0]]
            
            if len(slice_names) > 1:
                sampler_nodes_info = slices[slice_names[1]]
            else:
                sampler_nodes_info = trainer_nodes_info
            
            tpu_type = trainer_nodes_info[0].get("Labels", {}).get("ray.io/tpu-pod-type")
            trainer_tpu_count = int(sum([n.get("Resources", {}).get("TPU", 0) for n in trainer_nodes_info]))
            sampler_tpu_count = int(sum([n.get("Resources", {}).get("TPU", 0) for n in sampler_nodes_info]))
        else:
            trainer_nodes_info = []
            sampler_nodes_info = []
            trainer_tpu_count = 0
            sampler_tpu_count = 0
            tpu_type = None

        is_local_run = os.environ.get("LOCAL_VM_RUN") == "1"

        self.trainer_world_size = 1 if is_local_run else (trainer_tpu_count if trainer_tpu_count > 0 else 4)
        self.generator_world_size = config.generator.parallelism.tensor_parallel_degree
        self.trainer_dp_degree = self.trainer_world_size
        self._multi_node = False
        
        trainer_device_type = "cpu" if is_local_run else "tpu"
        generator_device_type = "tpu"
        
        trainer_nodes_info.sort(key=lambda n: int(n["Labels"].get("ray.io/tpu-worker-id", "0")))
        trainer_node_ips = [node["NodeManagerAddress"] for node in trainer_nodes_info] if trainer_nodes_info else ["127.0.0.1"]
        sampler_nodes_info.sort(key=lambda n: int(n["Labels"].get("ray.io/tpu-worker-id", "0")))
        sampler_node_ips = [node["NodeManagerAddress"] for node in sampler_nodes_info] if sampler_nodes_info else ["127.0.0.1"]

        chips_per_host = int(trainer_nodes_info[0]["Resources"].get("TPU", 4)) if trainer_nodes_info else 4
        
        trainer_sb_addresses = []
        if trainer_nodes_info:
            for node in trainer_nodes_info:
                dns_name = node["NodeManagerAddress"]
                for chip_idx in range(chips_per_host):
                    trainer_sb_addresses.append(f"{dns_name}:{8471 + chip_idx}")
        else:
            for chip_idx in range(chips_per_host):
                trainer_sb_addresses.append(f"127.0.0.1:{8471 + chip_idx}")

        trainer_actors = []
        
        # Compute absolute paths for the worker environment
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        dummy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "dummy_modules"))
        worker_pythonpath = f"{dummy_dir}:{base_dir}"

        for rank in range(self.trainer_world_size):
            local_chip_index = rank % chips_per_host
            host_index = rank // chips_per_host
            host_ip = trainer_node_ips[host_index]
            
            env_vars = {
                "TORCHTITAN_DEVICE_TYPE": trainer_device_type,
                "RANK": str(rank),
                "LOCAL_RANK": str(local_chip_index),
                "WORLD_SIZE": str(self.trainer_world_size),
                "MASTER_ADDR": trainer_node_ips[0],
                "MASTER_PORT": "8450",
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(trainer_sb_addresses),
                "TPU_PROCESS_ADDRESSES": ",".join(trainer_sb_addresses),
                "TPU_PROCESS_PORT": str(8471 + local_chip_index),
                "CLOUD_TPU_TASK_ID": str(rank),
                "TPU_WORKER_HOSTNAMES": ",".join(trainer_node_ips),
                "JAX_MEM_FRACTION": "0.45",
                "JAX_THREE_G_MEM_ALLOC_ON_FREE": "true",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.45",
                "TORCH_DYNAMO_RECOMPILE_LIMIT": "100",
                "PYTHONPATH": worker_pythonpath,
            }
            if tpu_type == "v6e-8":
                env_vars.update({
                    "TORCH_TPU_TOPOLOGY": "2,4,1" if self.trainer_world_size == 8 else ("2,2,1" if self.trainer_world_size == 4 else "1,1,1"),
                    "TPU_HOST_BOUNDS": "2,4,1" if self.trainer_world_size == 8 else ("2,2,1" if self.trainer_world_size == 4 else "1,1,1"),
                    "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
                    "CHIPS_PER_HOST": "4",
                })
            else:
                env_vars.update({
                    "TORCH_TPU_TOPOLOGY": "2,2,1" if self.trainer_world_size == 4 else "1,1,1",
                    "TPU_HOST_BOUNDS": "1,1,1",
                    "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1" if self.trainer_world_size == 4 else "1,1,1",
                    "CHIPS_PER_HOST": "4",
                })
                
            trainer_resources = {f"node:{host_ip}": 0.01}
            if trainer_device_type == "tpu":
                trainer_resources["TPU"] = 1
                
            trainer_resources = {f"node:{host_ip}": 0.01}
            if trainer_device_type == "tpu":
                trainer_resources["TPU"] = 1
                
            trainer_actor = RayTPUPolicyTrainer.options(
                resources=trainer_resources,
                runtime_env={"env_vars": env_vars}
            ).remote(
                config.trainer,
                model_spec=config.model_spec,
                hf_assets_path=config.hf_assets_path,
                generator_dtype=config.generator.model_dtype,
                compile_config=config.compile,
            )
            trainer_actors.append(trainer_actor)

        generator_sb_addresses = []
        if sampler_nodes_info:
            for node in sampler_nodes_info:
                dns_name = node["NodeManagerAddress"]
                for chip_idx in range(chips_per_host):
                    generator_sb_addresses.append(f"{dns_name}:{8070 + chip_idx}")
        else:
            for chip_idx in range(chips_per_host):
                generator_sb_addresses.append(f"127.0.0.1:{8070 + chip_idx}")

        generator_env_vars = {
            "TORCHTITAN_DEVICE_TYPE": generator_device_type,
            "SKIP_JAX_PRECOMPILE": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            # Pre-emptively set TPU env vars to prevent torchtpu-vllm from guessing wrong bounds
            "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
            "TPU_HOST_BOUNDS": "2,4,1" if self.generator_world_size == 8 else ("2,2,1" if self.generator_world_size == 4 else "1,1,1"),
            "TORCH_TPU_TOPOLOGY": "2,4,1" if self.generator_world_size == 8 else ("2,2,1" if self.generator_world_size == 4 else "1,1,1"),
            "CHIPS_PER_HOST": "4",
            # Pass correct slicebuilder addresses for the generator slice only
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(generator_sb_addresses),
            "TPU_PROCESS_ADDRESSES": ",".join(generator_sb_addresses),
            # Set LIBTPU_INIT_ARGS to prevent torchtpu-vllm from appending megachip_tccontrol
            "LIBTPU_INIT_ARGS": "--xla_tpu_use_enhanced_launch_barrier=false",
            # Dynamo recompile limit needs to be increased because vLLM passes layer_name strings
            "TORCH_DYNAMO_RECOMPILE_LIMIT": "100",
            "PYTHONPATH": worker_pythonpath,
        }
        if config.generator.parallelism.tensor_parallel_degree > 1:
            generator_env_vars["TPU_MULTIHOST_BACKEND"] = "ray"
            
        generator_resources = {}
        if not is_local_run and self.generator_world_size <= chips_per_host:
            generator_resources["TPU"] = self.generator_world_size

        generator_actor = RayVLLMGenerator.options(
            resources=generator_resources,
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=sampler_nodes_info[0]["NodeID"],
                soft=False
            ) if sampler_nodes_info else "DEFAULT",
            runtime_env={
                "env_vars": generator_env_vars
            }
        ).remote(
            config.generator,
            model_spec=config.model_spec,
            model_path=config.hf_assets_path,
            compile_config=config.compile,
            max_num_seqs=max(
                config.num_prompts_per_step * config.generator.sampling.n,
                config.num_validation_samples,
            ),
        )

        # Wrap them in Monarch-compatible proxies
        self.trainer = RayTrainerProxy(trainer_actors)
        self.generator = RayGeneratorProxy(generator_actor, trainer_actors)
        
        self._proc_meshes = [] # Dummy out Monarch meshes for cleanup

        print("@@@ Setup: Pushing initial weights...")
        self.trainer.push_model_state_dict.call().get()
        print("@@@ Setup: Pulling initial weights to vLLM...")
        self.generator.pull_model_state_dict.call(0).get()
        print("@@@ Setup: Completed!")


def main():
    init_logger()
    config = typing.cast(RLTrainer.Config, ConfigManager().parse_args())
    rl_trainer = RayRLTrainer(config)
    asyncio.run(rl_trainer.setup())
    asyncio.run(rl_trainer.train())

if __name__ == "__main__":
    main()
