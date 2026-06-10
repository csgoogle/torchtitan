# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RL training loop using Monarch Actors.

This demonstrates:
1. Distributed actor architecture with VLLMGenerator (vLLM) and PolicyTrainer (TorchTitan)
   running on separate GPU meshes
2. Weight synchronization across meshes: trainer gathers full (unsharded) weights,
   generator reshards to match its own parallelism layout via distribute_tensor
3. Envs driven rollouts; reward and advantage computation live inline
   in the controller.

Command to run:
python3 torchtitan/experiments/rl/grpo.py \
    --module rl --config rl_grpo_qwen3_0_6b \
    --hf_assets_path=<path_to_model_checkpoint>
"""

import multiprocessing
# Force 'spawn' start method to prevent deadlocks with shared GPU/TPU driver contexts.
# Wrap in try-except in case the context has already been initialized by other modules.
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
import asyncio
import logging
import math
import os as os
# Configure Monarch transport BEFORE importing any monarch module to ensure
# that all local agents/mailboxes are created with TCP transport instead of IPC.
if os.getenv("MONARCH_WORKERS"):
    from monarch.config import configure, ChannelTransport
    configure(default_transport=ChannelTransport.TcpWithHostname)
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any
import tyro

import torch
from torchtitan.config import Configurable, ParallelismConfig
from torchtitan.config.configs import CompileConfig
from torchtitan.config.manager import ConfigManager
from torchtitan.tools.utils import get_device_type
# TPU is not support torchstore. We skip it for now.
try:
    import torchstore as ts
except ImportError:
    ts = None
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async
from monarch.tools.network import AddrType
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.types import (
    Completion,
    Episode,
    Step,
    TrainBatch,
    Trajectory,
)
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


class GRPOLoss(Configurable):
    """Clipped GRPO surrogate loss.

    Takes per-sample response logprobs (already extracted from whatever
    packing or padding format the trainer uses).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_eps: float = 0.2
        """PPO clipping epsilon for the probability ratio."""

    def __init__(self, config: Config):
        self.clip_eps = config.clip_eps

    def __call__(
        self,
        policy_logprobs: list[torch.Tensor],
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        per_sample_mean_lps = []
        for policy_lps in policy_logprobs:
            per_sample_mean_lps.append(policy_lps.mean())

        mean_log_ratio = torch.stack(per_sample_mean_lps)
        ratio = torch.exp(mean_log_ratio)

        unclipped_loss = ratio * advantages
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        clipped_loss = clipped_ratio * advantages
        pg_loss = -torch.min(unclipped_loss, clipped_loss).mean()

        metrics = {
            "pg_loss": pg_loss.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_clipped_frac": (torch.abs(ratio - clipped_ratio) > 1e-6)
            .float()
            .mean()
            .item(),
        }
        return pg_loss, metrics


def _default_python_bootstrap() -> None:
    import sys
    # Enforce loading modified codebase from /app and remove /data to prevent stale file handles
    sys.path = [p for p in sys.path if p not in ("/data", "/data/torchtitan")]
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    print("@@@ MONARCH BOOTSTRAP sys.path:", sys.path, flush=True)


def _resolve_tpu_host_ips(mesh_hosts: list[str] | None) -> list[str]:
    """Resolves headless worker hostnames to IP addresses for TPU communication."""
    import socket
    import os
    active_hosts = mesh_hosts if mesh_hosts else [os.getenv("WORKERS_0_HOSTNAME", "localhost")]
    ip_hosts = []
    for h in active_hosts:
        if h == "localhost" or h == "127.0.0.1" or h.replace('.', '').isdigit():
            ip_hosts.append(h)
        else:
            fqdn = h
            if ".monarch-worker-headless" not in h:
                fqdn = f"{h}.monarch-worker-headless.default.svc.cluster.local"
            try:
                ip = socket.gethostbyname(fqdn)
                ip_hosts.append(ip)
            except Exception:
                ip_hosts.append(fqdn)
    return ip_hosts


def _build_tpu_topology_mapping(
    ip_hosts: list[str],
    num_gpus: int,
    host_rank: int,
    local_rank: int,
) -> tuple[list[str], int, str, str, int, int]:
    """Calculates the physical TPU 2D/3D torus grid coordinates, ranks, and ports.
    
    Returns:
        tuple containing:
        - tpu_worker_addresses (list[str]): List of all worker address ports.
        - global_rank (int): Global rank of the current process.
        - tpu_topology (str): Topology boundary string (host_x, host_y, 1).
        - tpu_process_bounds (str): Process boundary string.
        - host_x_dim (int): Host dimension along X.
        - host_y_dim (int): Host dimension along Y.
    """
    num_hosts = len(ip_hosts)
    
    if num_gpus == 1:
        tpu_worker_addresses = [f"{host_ip}:8471" for host_ip in ip_hosts]
        global_rank = host_rank
        tpu_topology = f"{num_hosts},1,1"
        tpu_process_bounds = f"{num_hosts},1,1"
        return tpu_worker_addresses, global_rank, tpu_topology, tpu_process_bounds, num_hosts, 1

    elif num_gpus == 2:
        tpu_worker_addresses = [None] * (num_hosts * 2)
        for h_rank, host_ip in enumerate(ip_hosts):
            for l_rank in range(2):
                p_x = h_rank * 2 + l_rank
                tpu_worker_addresses[p_x] = f"{host_ip}:{8471 + l_rank}"
        
        global_rank = host_rank * 2 + local_rank
        tpu_topology = f"{num_hosts * 2},1,1"
        tpu_process_bounds = f"{num_hosts * 2},1,1"
        return tpu_worker_addresses, global_rank, tpu_topology, tpu_process_bounds, num_hosts * 2, 1

    elif num_gpus == 4:
        if num_hosts == 4:
            host_x_dim, host_y_dim = 2, 2
        elif num_hosts == 2:
            host_x_dim, host_y_dim = 2, 1
        elif num_hosts == 8:
            host_x_dim, host_y_dim = 4, 2
        else:
            host_x_dim, host_y_dim = num_hosts, 1
            
        tpu_worker_addresses = [None] * (num_hosts * 4)
        for h_rank, host_ip in enumerate(ip_hosts):
            hx = h_rank % host_x_dim
            hy = h_rank // host_x_dim
            for l_rank in range(4):
                lx = l_rank % 2
                ly = l_rank // 2
                X = hx * 2 + lx
                Y = hy * 2 + ly
                g_rank = Y * (host_x_dim * 2) + X
                tpu_worker_addresses[g_rank] = f"{host_ip}:{8471 + l_rank}"
        
        hx = host_rank % host_x_dim
        hy = host_rank // host_x_dim
        lx = local_rank % 2
        ly = local_rank // 2
        X = hx * 2 + lx
        Y = hy * 2 + ly
        global_rank = Y * (host_x_dim * 2) + X
        tpu_topology = f"{host_x_dim},{host_y_dim},1"
        tpu_process_bounds = f"{host_x_dim * 2},{host_y_dim * 2},1"
        return tpu_worker_addresses, global_rank, tpu_topology, tpu_process_bounds, host_x_dim, host_y_dim

    else:
        raise ValueError(f"Unsupported num_gpus per host: {num_gpus}")


def _build_tpu_distributed_env(
    num_gpus: int,
    gpu_ids: list[int],
    host_rank: int,
    local_rank: int,
    mesh_hosts: list[str] | None,
    master_port: str,
) -> dict[str, str]:
    """Generates all process-level environment variables for distributed TPU coordination."""
    env = {}
    env["TPU_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    env["TPU_VISIBLE_CHIPS"] = ",".join(str(g) for g in gpu_ids)
    env["PJRT_DEVICE"] = "TPU"

    # 1. Resolve headless hosts to GKE internal IPs
    ip_hosts = _resolve_tpu_host_ips(mesh_hosts)
    num_hosts = len(ip_hosts)

    # 2. Build worker address port mapping and ranks
    tpu_worker_addresses, global_rank, tpu_topology, tpu_process_bounds, host_x_dim, host_y_dim = (
        _build_tpu_topology_mapping(ip_hosts, num_gpus, host_rank, local_rank)
    )

    tpu_worker_addresses_str = ",".join(tpu_worker_addresses)
    world_size = num_hosts * num_gpus
    master_addr = ip_hosts[0]

    env["WORLD_SIZE"] = str(world_size)
    env["RANK"] = str(global_rank)
    env["LOCAL_RANK"] = str(local_rank)
    env["LOCAL_WORLD_SIZE"] = str(num_gpus)
    env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = master_port
    env["PYTHONUNBUFFERED"] = "1"

    if num_hosts > 1:
        env["TPU_PROCESS_ADDRESSES"] = tpu_worker_addresses_str
        env["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = tpu_worker_addresses_str
        env["CLOUD_TPU_TASK_ID"] = str(host_rank)
        env["TPU_TASK_ID"] = str(host_rank)
        env["TPU_WORKER_HOSTNAMES"] = ",".join(ip_hosts)
        env["GLOG_v"] = "2"
        env["TF_CPP_MIN_LOG_LEVEL"] = "0"
        env["PT_XLA_DEBUG"] = "1"
        env["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

        if num_gpus == 4:
            env["TPU_PROCESS_PORT"] = str(8471 + local_rank)
            env["TPU_HOST_BOUNDS"] = tpu_topology
            x_chips = host_x_dim * 2
            y_chips = host_y_dim * 2
            env["TPU_PROCESS_BOUNDS"] = f"{x_chips},{y_chips},1"
            env["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
            env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
            env["TORCH_TPU_TOPOLOGY"] = f"{x_chips},{y_chips},1"
            
            env["HOST_BOUNDS"] = tpu_topology
            env["CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
        else:
            env["TPU_PROCESS_PORT"] = str(8471 + local_rank)
            env["TORCH_TPU_TOPOLOGY"] = tpu_topology
            env["TPU_HOST_BOUNDS"] = tpu_topology
            env["TPU_PROCESS_BOUNDS"] = tpu_process_bounds
            env["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
            env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
            env["HOST_BOUNDS"] = tpu_topology
            env["CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
    else:
        visible_devs = ",".join(str(g) for g in gpu_ids)
        env["TPU_VISIBLE_DEVICES"] = visible_devs
        env["TPU_VISIBLE_CHIPS"] = visible_devs
        env["TPU_WORKER_HOSTNAMES"] = "127.0.0.1"
        env["GLOG_v"] = "2"
        env["TF_CPP_MIN_LOG_LEVEL"] = "0"

        env["TPU_HOST_BOUNDS"] = "1,1,1"
        env["HOST_BOUNDS"] = "1,1,1"
        env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"

        if num_gpus == 4:
            env["TORCH_TPU_TOPOLOGY"] = "2,2,1"
            env["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
            env["CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
            env["TPU_PROCESS_BOUNDS"] = "2,2,1"
        else:
            tpu_topology = f"{num_gpus},1,1"
            env["TORCH_TPU_TOPOLOGY"] = tpu_topology
            env["TPU_CHIPS_PER_HOST_BOUNDS"] = tpu_topology
            env["CHIPS_PER_HOST_BOUNDS"] = tpu_topology
            env["TPU_PROCESS_BOUNDS"] = tpu_topology
    return env


class Provisioner:
    """Allocates non-overlapping GPU/TPU ranges for Monarch proc meshes.

    In non-colocated mode, the trainer and generator run on separate device
    meshes (e.g. devices 0-3 for training, devices 4-7 for generation). Each
    call to ``allocate(n)`` reserves the next *n* devices and returns a
    bootstrap command builder that sets process-level environment variables
    before initialization.
    """

    def __init__(self, total_gpus: int = 8, mesh_hosts: list[str] | None = None):
        self.total_gpus = total_gpus
        self.next_gpu = 0
        self.mesh_hosts = mesh_hosts

    @property
    def available(self) -> int:
        return self.total_gpus - self.next_gpu

    def allocate(self, num_gpus: int, device_dim: str, master_port: str = "29500", is_trainer: bool = False) -> Callable[[Any], Any]:
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} devices but only {self.available} "
                f"available (total={self.total_gpus}, allocated={self.next_gpu})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus
        mesh_hosts = self.mesh_hosts

        from monarch._src.actor.host_mesh import default_bootstrap_cmd

        def _get_cmd(point) -> Any:
            labels = point.keys()
            host_rank = point["hosts"] if "hosts" in labels else 0
            local_rank = point[device_dim] if device_dim in labels else 0

            is_tpu = os.environ.get("FORCE_DEVICE") == "tpu" or "TPU_HOST_BOUNDS" in os.environ
            
            env = {}
            if is_tpu:
                env = _build_tpu_distributed_env(
                    num_gpus=num_gpus,
                    gpu_ids=gpu_ids,
                    host_rank=host_rank,
                    local_rank=local_rank,
                    mesh_hosts=mesh_hosts,
                    master_port=master_port,
                )
            else:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[local_rank])
                if mesh_hosts:
                    world_size = len(mesh_hosts) * num_gpus
                    global_rank = host_rank * num_gpus + local_rank
                    master_addr = mesh_hosts[0]
                    env["WORLD_SIZE"] = str(world_size)
                    env["RANK"] = str(global_rank)
                    env["LOCAL_RANK"] = str(local_rank)
                    env["MASTER_ADDR"] = master_addr
                    env["MASTER_PORT"] = master_port
                    env["PYTHONUNBUFFERED"] = "1"

            if is_tpu:
                # GKE auto-injects default TPU_TOPOLOGY (e.g. 4x4) and TPU_ACCELERATOR_TYPE.
                # Since we partition the physical cluster into separate non-overlapping meshes
                # (Trainer vs Generator), these defaults will conflict with our manually sharded 
                # settings (e.g. TORCH_TPU_TOPOLOGY). We unset them at shell level right before
                # spawning Python so PJRT is forced to use our process-level configurations.
                cmd = default_bootstrap_cmd()
                cmd.program = "/bin/bash"
                cmd.args = [
                    "-c",
                    (
                        "unset TPU_TOPOLOGY; unset TPU_ACCELERATOR_TYPE; "
                        "exec python3 -u -c \""
                        "from monarch.config import configure, ChannelTransport; "
                        "configure(default_transport=ChannelTransport.TcpWithHostname); "
                        "from monarch._src.actor.bootstrap_main import invoke_main; "
                        "invoke_main()\" "
                        "\"$@\" > /tmp/monarch_bash_child_\\$\\$.log 2>&1"
                    ),
                    "--"
                ]
                return cmd.with_env(env)
            else:
                return default_bootstrap_cmd().with_env(env)

        return _get_cmd


def _mean_rewards(steps: list[Step]) -> dict[str, float]:
    """Per-component mean reward across a list of Steps."""
    return {k: sum(s.rewards[k] for s in steps) / len(steps) for k in steps[0].rewards}


def _format_rewards(components: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:+.3f}" for k, v in components.items())


def _format_validation(result: dict) -> str:
    return (
        f"mean_reward={result['mean_reward']:+.3f} "
        f"({_format_rewards(result['components'])})"
    )


def _log_samples(items: list[Episode] | list[Completion]) -> None:
    """Log the first sample per prompt for debugging."""
    seen_prompts: set[int] = set()
    for item in items:
        if item.prompt_idx in seen_prompts:
            continue
        seen_prompts.add(item.prompt_idx)
        reward_str = f" reward={item.reward:+.1f}" if hasattr(item, "reward") else ""
        logger.info(f"  [prompt {item.prompt_idx}]{reward_str}")
        logger.info(f"       A: {item.text[:300].replace(chr(10), ' ').strip()}")


class RLTrainer(Configurable):
    """Top-level RL training orchestrator."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Top-level config for RL training."""

        model_spec: Annotated[ModelSpec | None, tyro.conf.Suppress] = None
        """Model specification shared by trainer and generator.
        Set programmatically via config_registry (not from CLI)."""

        hf_assets_path: str = "./tests/assets/tokenizer"
        """Path to HF assets folder (model weights, tokenizer, config files)."""

        num_steps: int = 10
        """Number of RL training steps."""

        dump_folder: str = "outputs/rl"
        """Root output folder for RL artifacts (temp weights, logs, etc.)."""

        num_prompts_per_step: int = 5
        """Number of distinct prompts (= GRPO groups) drawn per training step.

        The total episodes per step is ``num_prompts_per_step * group_size``,
        where ``group_size`` is ``generator.sampling.n`` (completions per prompt).
        """

        num_validation_samples: int = 20
        """Number of held-out prompts scored greedily (temp=0, n=1) per validation pass."""

        env: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Env config for training rollouts."""

        validation_env: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Env config for validation rollouts."""

        log_samples: bool = False
        """Log first completion per episode during training and validation."""

        compile: CompileConfig = field(default_factory=CompileConfig)
        """torch.compile config shared by trainer and generator."""

        trainer: PolicyTrainer.Config = field(
            default_factory=lambda: PolicyTrainer.Config(loss=GRPOLoss.Config())
        )
        """PolicyTrainer config. Controls optimizer, training, parallelism."""

        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

        def __post_init__(self):
            if self.trainer.debug.batch_invariant:
                if not self.trainer.debug.deterministic:
                    raise ValueError("batch_invariant requires deterministic=True")
                # TODO: Replace trainer dtype constraint to use mixed
                #  training enabled by FSDP.
                if self.trainer.training.dtype != "bfloat16":
                    raise ValueError(
                        f"batch_invariant requires bfloat16 training dtype, "
                        f"got {self.trainer.training.dtype!r}"
                    )
                if self.generator.model_dtype != "bfloat16":
                    raise ValueError(
                        f"batch_invariant requires bfloat16 generator dtype, "
                        f"got {self.generator.model_dtype!r}"
                    )
                if self.trainer.parallelism.enable_sequence_parallel:
                    raise ValueError(
                        "batch_invariant mode doesn't support SP now. "
                        "SP uses reduce-scatter which only supports Ring in NCCL "
                        "and has not been validated for determinism."
                    )

    def __init__(self, config: Config):
        self.config = config
        self._proc_meshes = []

    async def cleanup(self):
        """Stop all proc meshes to release GPU memory."""
        for mesh in self._proc_meshes:
            try:
                await mesh.stop()
            except Exception:
                pass
        self._proc_meshes = []

    def _get_rank_0_value(self, result, has_devices: bool = True):
        """Extract rank 0 result, handling both single and multi-node meshes.

        Monarch actor endpoints return results from all ranks in the mesh.
        This picks out rank 0's result by indexing into the host and device
        dimensions as needed (multi-node meshes have an extra host dimension).
        This should be used in cases where all ranks return the same result.
        """
        kwargs = {}
        if self._multi_node:
            kwargs["hosts"] = 0
        if has_devices:
            device_dim = "tpus" if get_device_type() == "tpu" else "gpus"
            kwargs[device_dim] = 0
        return result.item(**kwargs)

    @staticmethod
    def _compute_world_size(p: ParallelismConfig) -> int:
        """Compute world size from all parallel dimensions."""
        dp_shard = max(p.data_parallel_shard_degree, 1)
        return (
            p.data_parallel_replicate_degree
            * dp_shard
            * p.tensor_parallel_degree
            * p.pipeline_parallel_degree
            * p.context_parallel_degree
        )

    def _shard_episodes(self, episodes: list[Episode]) -> list[list[Episode]]:
        """Round-robin partition episodes across DP ranks."""
        return [
            [episodes[i] for i in range(rank, len(episodes), self.trainer_dp_degree)]
            for rank in range(self.trainer_dp_degree)
        ]

    @staticmethod
    def _collate_episodes(episodes: list[Episode]) -> TrainBatch:
        """Pack episodes into a single varlen-packed TrainBatch."""
        all_ids: list[int] = []
        prompt_lens: list[int] = []
        response_lens: list[int] = []

        for ep in episodes:
            all_ids.extend(ep.prompt_token_ids + ep.token_ids)
            prompt_lens.append(len(ep.prompt_token_ids))
            response_lens.append(len(ep.token_ids))

        return TrainBatch(
            token_ids=torch.tensor([all_ids], dtype=torch.long),
            prompt_lens=prompt_lens,
            response_lens=response_lens,
            seq_lens=[p + r for p, r in zip(prompt_lens, response_lens)],
            advantages=torch.tensor(
                [ep.advantage for ep in episodes],
                dtype=torch.float32,
            ),
            token_logprobs=[ep.token_logprobs for ep in episodes],
        )

    async def setup(
        self,
        *,
        host_mesh=None,
        trainer_nodes: int | None = None,
        generator_nodes: int | None = None,
        gpus_per_node: int | None = None,
        mesh_hosts: list[str] | None = None,
    ):
        """Spawn Monarch actors on separate meshes and initialize weights.

        Creates separate GPU meshes for trainer and generator and
        synchronizes initial weights from trainer to generator. Must be
        called before :meth:`train`.

        Args:
            host_mesh: Optional multi-node HostMesh. When provided,
                whole nodes are dedicated to trainer vs generator
                roles instead of partitioning GPUs on a single host.
            trainer_nodes: Number of nodes for the trainer (required when
                host_mesh is provided).
            generator_nodes: Number of nodes for the generator (required when
                host_mesh is provided).
            gpus_per_node: GPUs per node, assumed to be the same across all
                nodes (no heterogeneous node configurations). Required when
                host_mesh is provided.
            mesh_hosts: Optional list of all worker pod hostnames in the GKE cluster.
        """
        config = self.config
        barrier = None

        self.trainer_world_size = self._compute_world_size(config.trainer.parallelism)
        self.generator_world_size = self._compute_world_size(
            config.generator.parallelism
        )
        trainer_parallelism = config.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )

        total_gpus = self.trainer_world_size + self.generator_world_size
        logger.info(
            f"{self.generator_world_size} generator GPUs + "
            f"{self.trainer_world_size} trainer GPUs = {total_gpus} total"
        )

        self._multi_node = host_mesh is not None

        if host_mesh is not None:

            # Multi-node mode: dedicate whole nodes to trainer vs generator
            if (
                trainer_nodes is None
                or generator_nodes is None
                or gpus_per_node is None
            ):
                raise ValueError(
                    "trainer_nodes, generator_nodes, and gpus_per_node are "
                    "required when host_mesh is provided"
                )
            # Validate that world sizes are evenly divisible by node counts
            assert self.trainer_world_size % trainer_nodes == 0, (
                f"trainer_world_size ({self.trainer_world_size}) must be "
                f"evenly divisible by trainer_nodes ({trainer_nodes})"
            )
            assert self.generator_world_size % generator_nodes == 0, (
                f"generator_world_size ({self.generator_world_size}) must be "
                f"evenly divisible by generator_nodes ({generator_nodes})"
            )

            # Compute GPUs per node for each role based on the config's
            # world size and number of nodes allocated to that role
            trainer_gpus_per_node = self.trainer_world_size // trainer_nodes
            generator_gpus_per_node = self.generator_world_size // generator_nodes

            trainer_host_mesh = host_mesh.slice(hosts=slice(0, trainer_nodes))
            generator_host_mesh = host_mesh.slice(
                hosts=slice(trainer_nodes, trainer_nodes + generator_nodes)
            )

            trainer_mesh_hosts = mesh_hosts[:trainer_nodes] if mesh_hosts else None
            generator_mesh_hosts = mesh_hosts[trainer_nodes:trainer_nodes + generator_nodes] if mesh_hosts else None

            # Use Provisioner to set CUDA_VISIBLE_DEVICES so each role
            # only sees its own GPUs and doesn't conflict with other
            # processes on the node
            trainer_provisioner = Provisioner(total_gpus=gpus_per_node, mesh_hosts=trainer_mesh_hosts)
            generator_provisioner = Provisioner(total_gpus=gpus_per_node, mesh_hosts=generator_mesh_hosts)

            device_dim = "tpus" if get_device_type() == "tpu" else "gpus"
            trainer_mesh = trainer_host_mesh.spawn_procs(
                per_host={device_dim: trainer_gpus_per_node},
                bootstrap=_default_python_bootstrap,
                bootstrap_command=trainer_provisioner.allocate(trainer_gpus_per_node, device_dim, master_port="29500", is_trainer=True),
            )
            generator_mesh = generator_host_mesh.spawn_procs(
                per_host={device_dim: generator_gpus_per_node},
                bootstrap=_default_python_bootstrap,
                bootstrap_command=generator_provisioner.allocate(generator_gpus_per_node, device_dim, master_port="29501", is_trainer=False),
            )
        else:
            # Single-node mode: partition GPUs/TPUs on this_host()
            device_dim = "tpus" if get_device_type() == "tpu" else "gpus"
            provisioner = Provisioner(total_gpus=total_gpus)
            trainer_mesh = this_host().spawn_procs(
                per_host={device_dim: self.trainer_world_size},
                bootstrap=_default_python_bootstrap,
                bootstrap_command=provisioner.allocate(self.trainer_world_size, device_dim, master_port="29500", is_trainer=True),
            )
            generator_mesh = this_host().spawn_procs(
                per_host={device_dim: self.generator_world_size},
                bootstrap=_default_python_bootstrap,
                bootstrap_command=provisioner.allocate(self.generator_world_size, device_dim, master_port="29501", is_trainer=False),
            )

        # Store proc meshes for cleanup
        self._proc_meshes = [trainer_mesh, generator_mesh]

        # Ranks and networking environment variables are pre-configured by Provisioner.allocate().
        # Bypassing sequential elastic rank setup to prevent TPU rank-to-IP mismatches.
        # await setup_torch_elastic_env_async(trainer_mesh, use_ipaddr=AddrType.IPv4)
        # await setup_torch_elastic_env_async(generator_mesh, use_ipaddr=AddrType.IPv4)

        # Spawn actors on their respective meshes
        # Spawn actors concurrently on their respective meshes to avoid PJRT deadlock on TPU
        import asyncio
        self.trainer, self.generator = await asyncio.gather(
            asyncio.to_thread(
                trainer_mesh.spawn,
                "trainer",
                PolicyTrainer,
                config.trainer,
                model_spec=config.model_spec,
                hf_assets_path=config.hf_assets_path,
                generator_dtype=config.generator.model_dtype,
                compile_config=config.compile,
            ),
            asyncio.to_thread(
                generator_mesh.spawn,
                "generator",
                VLLMGenerator,
                config.generator,
                model_spec=config.model_spec,
                model_path=config.hf_assets_path,
                compile_config=config.compile,
                max_num_seqs=max(
                    config.num_prompts_per_step * config.generator.sampling.n,
                    config.num_validation_samples,
                ),
            )
        )

        # Setup trainer processes with staggered sleep to avoid TPU JIT/vbar deadlocks
        await self.trainer.setup.call()

        # Initialize TorchStore for weight sync between trainer and generator (bypassed on TPU).
        # StorageVolumes are spawned on the trainer mesh so they are colocated
        # with the weight source for faster data access in the non-RDMA path.
        # LocalRankStrategy: routes each process to a storage volume based on
        #   LOCAL_RANK, so colocated processes share the same volume.
        # https://github.com/meta-pytorch/torchstore
        if get_device_type() != "tpu":
            await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

        # push weights from trainer
        await self.trainer.push_model_state_dict.call()
        # pull weights for policy version 0 (initial weights)
        await self.generator.pull_model_state_dict.call(0)

    async def _collect_rollouts(self, num_groups: int, step: int) -> list[Trajectory]:
        """Collect group rollouts: one single-use env per group, scored and returned."""
        envs = [
            self.config.env.build(step=step, group_idx=i) for i in range(num_groups)
        ]
        completions = self._get_rank_0_value(
            await self.generator.generate.call([env.prompt for env in envs])
        )
        trajectories: list[Trajectory] = []
        for c in completions:
            step_result = envs[c.prompt_idx].step(c.text)
            trajectories.append(
                Trajectory(sample_idx=c.prompt_idx, transitions=[(c, step_result)])
            )
        return trajectories

    @staticmethod
    def _build_episodes(trajectories: list[Trajectory]) -> list[Episode]:
        """Group trajectories by sample, apply mean-baseline advantage, flatten to Episodes."""
        groups: dict[int, list[Trajectory]] = {}
        for t in trajectories:
            groups.setdefault(t.sample_idx, []).append(t)

        episodes: list[Episode] = []
        for sample_idx, group in groups.items():
            rewards = [t.total_reward for t in group]
            group_mean = sum(rewards) / len(rewards)
            for t in group:
                # Single-turn: exactly one (completion, step) per trajectory.
                c, _ = t.transitions[0]
                episodes.append(
                    Episode(
                        policy_version=c.policy_version,
                        prompt_idx=sample_idx,
                        prompt_token_ids=c.prompt_token_ids,
                        text=c.text,
                        token_ids=c.token_ids,
                        token_logprobs=c.token_logprobs,
                        reward=t.total_reward,
                        advantage=t.total_reward - group_mean,
                    )
                )
        return episodes

    async def validate(self) -> dict:
        """Run validation on held-out prompts using greedy sampling.
        TODO: investigate using pass@k."""
        num_samples = self.config.num_validation_samples
        envs = [
            self.config.validation_env.build(step=0, group_idx=i)
            for i in range(num_samples)
        ]
        greedy = SamplingConfig(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.config.generator.sampling.max_tokens,
        )
        completions = self._get_rank_0_value(
            await self.generator.generate.call(
                [env.prompt for env in envs], sampling_config=greedy
            )
        )

        steps = [env.step(completions[i].text) for i, env in enumerate(envs)]

        if self.config.log_samples:
            _log_samples(completions)

        components = _mean_rewards(steps)
        return {
            "mean_reward": sum(components.values()),
            "components": components,
            "total": num_samples,
        }

    async def train(self):
        num_steps = self.config.num_steps
        num_groups = self.config.num_prompts_per_step
        logger.info(f"Pre-training validation; then {num_steps} steps of RL training")
        pre_validation = await self.validate()
        logger.info(f"Pre:  {_format_validation(pre_validation)}")

        for step in range(num_steps):
            step_start = time.perf_counter()

            # --- Collect data and create episodes --- #
            trajectories = await self._collect_rollouts(num_groups, step=step)
            episodes = self._build_episodes(trajectories)

            if self.config.log_samples:
                _log_samples(episodes)

            # --- Train step --- #
            batches = [
                self._collate_episodes(per_rank_episodes)
                for per_rank_episodes in self._shard_episodes(episodes)
            ]
            fwd_bwd_metrics = self._get_rank_0_value(
                await self.trainer.forward_backward.call(batches)
            )
            optim_metrics = self._get_rank_0_value(await self.trainer.optim_step.call())
            metrics = {**fwd_bwd_metrics, **optim_metrics}

            # --- Weight sync --- #
            t0 = time.perf_counter()
            await self.trainer.push_model_state_dict.call()
            t_push = time.perf_counter() - t0
            await self.generator.pull_model_state_dict.call(metrics["policy_version"])
            t_sync = time.perf_counter() - t0
            logger.info(f"Weight sync: push={t_push:.3f}s, total={t_sync:.3f}s")

            # --- Logging --- #
            steps = [t.transitions[0][1] for t in trajectories]
            components = _mean_rewards(steps)
            avg_tokens = sum(len(ep.token_ids) for ep in episodes) / len(episodes)
            logger.info(
                f"Step {step:2d} | Loss: {metrics['loss']:+.4f} | "
                f"Reward: {sum(components.values()):+.3f} ({_format_rewards(components)}) | "
                f"Avg tokens: {avg_tokens:>3.0f} | "
                f"Logprob diff: mean={metrics['logprob_diff_mean']:.4e}, "
                f"max={metrics['logprob_diff_max']:.4e} | "
                f"Time: {time.perf_counter() - step_start:.1f}s"
            )

            if not math.isfinite(metrics["loss"]):
                logger.error("Loss is NaN/Inf; training diverged")
                break

        logger.info("Post-training validation")
        post_validation = await self.validate()
        logger.info(
            f"Summary:\n  Pre:  {_format_validation(pre_validation)}\n"
            f"  Post: {_format_validation(post_validation)}"
        )


async def main():
    """Run the distributed RL training loop using Monarch."""
    import logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        handler = logging.StreamHandler()
        root_logger.addHandler(handler)
    import os
    if os.getenv("FORCE_DEVICE") == "tpu":
        from torchtitan.tools.utils import set_device_type
        set_device_type("tpu")
        logger.info("Forced device type to TPU")

    config = ConfigManager().parse_args()
    rl_trainer = RLTrainer(config)

    # Automatically detect GKE or Slurm multi-node cluster environment
    import os
    workers_env = os.getenv("MONARCH_WORKERS")
    if workers_env:
        try:
            from monarch.actor import attach_to_workers
            workers = [w.strip() for w in workers_env.split(",") if w.strip()]
            logger.info(f"Connecting to remote monarch workers: {workers}")
            host_mesh = attach_to_workers(ca="trust_all_connections", workers=workers)

            # Parse hostnames out of the worker endpoints
            mesh_hosts = []
            for w in workers:
                h = w
                if h.startswith("tcp://"):
                    h = h[len("tcp://"):]
                if ":" in h:
                    h = h.split(":")[0]
                mesh_hosts.append(h)

            # Derive node counts and device configuration
            gpus_per_node = int(os.getenv("LOCAL_WORLD_SIZE", "4"))
            total_nodes = len(workers)

            # Split cluster nodes between Trainer and Generator meshes, defaulting to 50/50
            trainer_nodes = int(os.getenv("TRAINER_NODES", str(total_nodes // 2)))
            generator_nodes = int(os.getenv("GENERATOR_NODES", str(total_nodes - trainer_nodes)))
            assert trainer_nodes + generator_nodes == total_nodes, f"Sum of trainer_nodes ({trainer_nodes}) and generator_nodes ({generator_nodes}) must equal total_nodes ({total_nodes})"

            logger.info(
                f"Distributed GKE HostMesh initialized with {total_nodes} nodes. "
                f"Allocating {trainer_nodes} hosts to training and {generator_nodes} hosts to sampling."
            )

            await rl_trainer.setup(
                host_mesh=host_mesh,
                trainer_nodes=trainer_nodes,
                generator_nodes=generator_nodes,
                gpus_per_node=gpus_per_node,
                mesh_hosts=mesh_hosts,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize remote GKE workers: {e}. Falling back to single-node.")
            await rl_trainer.setup()
    else:
        await rl_trainer.setup()

    await rl_trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
