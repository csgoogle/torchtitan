# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Ray-based RL training orchestrator for TPU.

Implements `RayRLTrainer`, which inherits core GRPO logic from the base GPU `RLTrainer` 
but overrides `setup()` to spawn Ray actors instead of Monarch procs. Uses proxy classes 
(`RayTrainerProxy`, `RayGeneratorProxy`) to transparently translate Monarch `.call().get()` 
RPCs into Ray `.remote()` and `ray.get()` futures. Pins the trainer to CPU and the 
generator to TPU to bypass PJRT client deadlock limitations.
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

from torchtitan.experiments.tpu.rl_nc_ray.tpu_env_utils import (
    discover_tpu_cluster_layout,
    build_trainer_env_vars,
    build_generator_env_vars,
)


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
                    
                    # [TPU HACK / OPTIMIZATION]
                    # Why: Passing refs[0] directly to the remote actor auto-resolves/deserializes it on the generator driver.
                    # This double-serialization on the generator driver CPU causes massive CPU/memory bottlenecks.
                    # Solution: Wrap refs[0] in a list [refs[0]] to bypass auto-resolution, passing raw ObjectRef.
                    # TODO: Remove this container-wrapping hack once Ray supports a cleaner way to pass
                    # unresolved ObjectRefs directly.
                    return FutureProxy([self.generator.load_model_state_dict.remote(version, [refs[0]])])
                method = getattr(self.generator, name)
                return FutureProxy([method.remote(*args, **kwargs)])
        return MethodProxy(self.ray_generator, self.ray_trainers)


class RayRLTrainer(RLTrainer):
    """Subclasses the GPU RLTrainer, overriding ONLY setup() to inject Ray."""

    async def setup(self, **kwargs):
        config = self.config
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # 1. Discover physical TPU cluster nodes, topology layout, bounds, and process counts
        layout = discover_tpu_cluster_layout(config)
        
        is_local_run = layout["is_local_run"]
        sampler_nodes_info = layout["sampler_nodes_info"]
        chips_per_host = layout["chips_per_host"]

        self.trainer_world_size = layout["trainer_world_size"]
        self.generator_world_size = layout["generator_world_size"]
        self.trainer_dp_degree = self.trainer_world_size
        self._multi_node = False
        
        trainer_device_type = layout["trainer_device_type"]
        generator_device_type = layout["generator_device_type"]

        # Compute absolute paths for the worker runtime environment
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        dummy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "dummy_modules"))
        worker_pythonpath = f"{dummy_dir}:{base_dir}"

        # 2. Launch Ray Trainer actors with worker-specific layout environments
        trainer_actors = []
        for rank in range(self.trainer_world_size):
            env_vars, host_ip = build_trainer_env_vars(rank, layout, worker_pythonpath)
                
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

        # 3. Launch Ray Generator actor
        generator_env_vars = build_generator_env_vars(layout, worker_pythonpath)
            
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

        # 4. Wrap actors in Monarch-compatible proxies
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
