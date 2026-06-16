import os
import sys
import types
import asyncio
import functools
import ray
import torch
import contextlib
from typing import Dict, Tuple

# ===============================================================================
# Monarch Actor Interception Hack
# ===============================================================================
# We intercept monarch.actor so that PolicyTrainer and VLLMGenerator do not inherit
# from Monarch's RPC Actor class or wrap methods in Monarch endpoints. This turns
# them into plain Python classes that run natively inside Ray workers.
mock_monarch = types.ModuleType("monarch.actor")

class DummyActor:
    def __init__(self, *args, **kwargs):
        pass

def dummy_endpoint(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # The original methods are async def, so we run them synchronously in Ray workers
        coro = func(*args, **kwargs)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    return wrapper

mock_monarch.Actor = DummyActor
mock_monarch.endpoint = dummy_endpoint
sys.modules["monarch.actor"] = mock_monarch

from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.actors.generator import VLLMGenerator
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu import utils as tpu_utils

try:
    from torch_tpu._internal.sync import synchronize
except ImportError:
    def synchronize(*args, **kwargs):
        pass


class TPUPolicyTrainer(PolicyTrainer):
    """Subclass of PolicyTrainer implementing the TPU graph split optimization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'parallel_dims') and self.parallel_dims.dp_replicate_enabled and getattr(self.device, 'type', '') == "tpu":
            self.maybe_enable_amp = contextlib.nullcontext()

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )
        with self.train_context():
            with self.maybe_enable_amp:
                pred = self.model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                loss_sum = self.loss_fn(pred, labels)
                loss = loss_sum / global_valid_tokens
            del pred

            synchronize(loss, wait=False)
            loss.backward()

        return loss


@ray.remote(resources={"TPU": 1})  # 1 Actor per TPU chip/core
class RayPolicyTrainerWorker:
    """Ray Worker representing a single Trainer rank on 1 TPU."""
    
    def __init__(self, rank, world_size, master_addr, master_port, config, model_spec, hf_assets_path, generator_dtype, compile_config):
        # Set environment variables so dist.init_process_group("tpu_dist") works inside Ray
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        
        # Configure torch_tpu slicebuilder addresses and topology for PyTorch XLA / tpu_dist
        if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
            ports = [master_port + i for i in range(world_size)]
            os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join([f"{master_addr}:{p}" for p in ports])
        if "TORCH_TPU_TOPOLOGY" not in os.environ:
            topo_map = {1: "1,1,1", 2: "2,1,1", 4: "2,2,1", 8: "2,2,2"}
            os.environ["TORCH_TPU_TOPOLOGY"] = topo_map.get(world_size, "2,2,1")

        # Apply SFT hacks
        if tpu_utils.get_device_type() == "tpu" and config.optimizer.implementation == "fused":
            config.optimizer.implementation = "foreach"
        if config.debug.seed is None:
            config.debug.seed = 42

        # Patch parallelize to inject TPU patches
        orig_parallelize = model_spec.parallelize_fn
        def patched_parallelize(model, *args, **kwargs):
            workarounds.apply_patches(model, config)
            return orig_parallelize(model, *args, **kwargs)
        model_spec.parallelize_fn = patched_parallelize

        self.trainer = TPUPolicyTrainer(
            config,
            model_spec=model_spec,
            hf_assets_path=hf_assets_path,
            generator_dtype=generator_dtype,
            compile_config=compile_config
        )

    def forward_backward(self, batches):
        return self.trainer.forward_backward(batches)

    def optim_step(self):
        return self.trainer.optim_step()

    def get_sharded_weights(self) -> Tuple[Dict[str, torch.Tensor], int]:
        """Return the local sharded weights and policy version directly."""
        weights = {k: v.cpu() for k, v in self.trainer.model.state_dict().items()}
        return weights, self.trainer.policy_version


@ray.remote(resources={"TPU": 1})  # 1 Actor per TPU chip/core
class RayVLLMGeneratorWorker:
    """Ray Worker representing a single Generator rank on 1 TPU."""
    
    def __init__(self, rank, world_size, master_addr, master_port, config, model_spec, model_path, compile_config, max_num_seqs):
        os.environ["VLLM_TARGET_DEVICE"] = "tpu"
        os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
        os.environ["VLLM_TPU_MULTIHOST_BACKEND"] = "ray"
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        # Configure torch_tpu slicebuilder addresses and topology for vLLM TPU backend
        if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
            ports = [master_port + i for i in range(world_size)]
            os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join([f"{master_addr}:{p}" for p in ports])
        if "TORCH_TPU_TOPOLOGY" not in os.environ:
            topo_map = {1: "1,1,1", 2: "2,1,1", 4: "2,2,1", 8: "2,2,2"}
            os.environ["TORCH_TPU_TOPOLOGY"] = topo_map.get(world_size, "2,2,1")

        self.generator = VLLMGenerator(
            config,
            model_spec=model_spec,
            model_path=model_path,
            compile_config=compile_config,
            max_num_seqs=max_num_seqs
        )

    def generate(self, prompts, sampling_config=None):
        # In external_launcher mode, all ranks must run generate in sync
        return self.generator.generate(prompts, sampling_config=sampling_config)

    def load_sharded_weights(self, weights_and_version: Tuple[Dict[str, torch.Tensor], int]):
        """Load the local sharded weights, update policy version, and reset prefix cache."""
        state_dict, version = weights_and_version
        self.generator._get_model().model.load_state_dict(state_dict)
        self.generator.policy_version = version
        # CRITICAL: Invalidate the KV prefix cache so stale values from old weights are not reused
        self.generator._engine.reset_prefix_cache()
