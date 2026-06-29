"""
tpu_env_utils.py

TPU Cluster Topology, layout discovery, and multi-process environment generation utilities
for Ray-native non-colocated RL.
"""

import os
import ray

# --- GLOBAL CONFIGURATION / CONSTANTS ---
# =====================================================================
# TPU vs GPU Memory Preallocation and Runtime Semantics:
# 1. GPU Allocation: CUDA relies on PyTorch's dynamic caching allocator, requesting HBM on-demand.
# 2. TPU Preallocation: The PJRT runtime (used by TorchTPU and JAX) preallocates 90% of TPU HBM
#    upfront by default. In non-colocated RL where multiple distinct components (FSDP training
#    and vLLM generation) run in parallel, we must set XLA_PYTHON_CLIENT_PREALLOCATE="false"
#    and bound memory fractions (e.g., JAX_MEM_FRACTION/XLA_PYTHON_CLIENT_MEM_FRACTION = 0.45)
#    to prevent initialization OOMs/deadlocks.
# 3. Launch Barriers: We disable enhanced launch barriers to prevent PJRT from hanging when 
#    spawning multiple separate distributed runtimes on the same physical host.
# =====================================================================

TRAINER_MASTER_PORT = "8450"
TRAINER_BASE_PORT = 8471
GENERATOR_BASE_PORT = 8070

JAX_MEM_FRACTION = "0.45"
JAX_THREE_G_MEM_ALLOC_ON_FREE = "true"
XLA_PYTHON_CLIENT_PREALLOCATE = "false"
XLA_PYTHON_CLIENT_MEM_FRACTION = "0.45"
LIBTPU_INIT_ARGS = "--xla_tpu_use_enhanced_launch_barrier=false"
VLLM_ENABLE_V1_MULTIPROCESSING = "1"
TORCH_DYNAMO_RECOMPILE_LIMIT = 100


def discover_tpu_cluster_layout(config) -> dict:
    """
    Query Ray nodes to discover physical TPU node topology, host bounds, and worker arrangements.
    """
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
    trainer_world_size = 1 if is_local_run else (trainer_tpu_count if trainer_tpu_count > 0 else 4)
    generator_world_size = config.generator.parallelism.tensor_parallel_degree
    
    trainer_device_type = "cpu" if is_local_run else "tpu"
    generator_device_type = "tpu"
    
    if trainer_nodes_info:
        trainer_nodes_info.sort(key=lambda n: int(n["Labels"].get("ray.io/tpu-worker-id", "0")))
        trainer_node_ips = [node["NodeManagerAddress"] for node in trainer_nodes_info]
    else:
        trainer_node_ips = ["127.0.0.1"]

    if sampler_nodes_info:
        sampler_nodes_info.sort(key=lambda n: int(n["Labels"].get("ray.io/tpu-worker-id", "0")))
        sampler_node_ips = [node["NodeManagerAddress"] for node in sampler_nodes_info]
    else:
        sampler_node_ips = ["127.0.0.1"]

    chips_per_host = int(trainer_nodes_info[0]["Resources"].get("TPU", 4)) if trainer_nodes_info else 4
    
    return {
        "is_local_run": is_local_run,
        "trainer_world_size": trainer_world_size,
        "generator_world_size": generator_world_size,
        "trainer_device_type": trainer_device_type,
        "generator_device_type": generator_device_type,
        "trainer_nodes_info": trainer_nodes_info,
        "sampler_nodes_info": sampler_nodes_info,
        "trainer_node_ips": trainer_node_ips,
        "sampler_node_ips": sampler_node_ips,
        "chips_per_host": chips_per_host,
        "tpu_type": tpu_type,
    }


def build_trainer_env_vars(rank: int, layout: dict, worker_pythonpath: str) -> tuple[dict, str]:
    """
    Build environment variables dictionary for the given trainer rank.
    """
    chips_per_host = layout["chips_per_host"]
    trainer_device_type = layout["trainer_device_type"]
    trainer_world_size = layout["trainer_world_size"]
    trainer_node_ips = layout["trainer_node_ips"]
    tpu_type = layout["tpu_type"]
    trainer_nodes_info = layout["trainer_nodes_info"]
    
    local_chip_index = rank % chips_per_host
    host_index = rank // chips_per_host
    host_ip = trainer_node_ips[host_index]
    
    trainer_sb_addresses = []
    if trainer_nodes_info:
        for node in trainer_nodes_info:
            dns_name = node["NodeManagerAddress"]
            for chip_idx in range(chips_per_host):
                trainer_sb_addresses.append(f"{dns_name}:{TRAINER_BASE_PORT + chip_idx}")
    else:
        for chip_idx in range(chips_per_host):
            trainer_sb_addresses.append(f"127.0.0.1:{TRAINER_BASE_PORT + chip_idx}")
            
    env_vars = {
        "TORCHTITAN_DEVICE_TYPE": trainer_device_type,
        "RANK": str(rank),
        "LOCAL_RANK": str(local_chip_index),
        "WORLD_SIZE": str(trainer_world_size),
        "MASTER_ADDR": trainer_node_ips[0],
        "MASTER_PORT": TRAINER_MASTER_PORT,
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(trainer_sb_addresses),
        "TPU_PROCESS_ADDRESSES": ",".join(trainer_sb_addresses),
        "TPU_PROCESS_PORT": str(TRAINER_BASE_PORT + local_chip_index),
        "CLOUD_TPU_TASK_ID": str(rank),
        "TPU_WORKER_HOSTNAMES": ",".join(trainer_node_ips),
        "JAX_MEM_FRACTION": JAX_MEM_FRACTION,
        "JAX_THREE_G_MEM_ALLOC_ON_FREE": JAX_THREE_G_MEM_ALLOC_ON_FREE,
        "XLA_PYTHON_CLIENT_PREALLOCATE": XLA_PYTHON_CLIENT_PREALLOCATE,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": XLA_PYTHON_CLIENT_MEM_FRACTION,
        "TORCH_DYNAMO_RECOMPILE_LIMIT": str(TORCH_DYNAMO_RECOMPILE_LIMIT),
        "PYTHONPATH": worker_pythonpath,
    }
    
    if tpu_type == "v6e-8":
        env_vars.update({
            "TORCH_TPU_TOPOLOGY": "2,4,1" if trainer_world_size == 8 else ("2,2,1" if trainer_world_size == 4 else "1,1,1"),
            "TPU_HOST_BOUNDS": "2,4,1" if trainer_world_size == 8 else ("2,2,1" if trainer_world_size == 4 else "1,1,1"),
            "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
            "CHIPS_PER_HOST": "4",
        })
    else:
        env_vars.update({
            "TORCH_TPU_TOPOLOGY": "2,2,1" if trainer_world_size == 4 else "1,1,1",
            "TPU_HOST_BOUNDS": "1,1,1",
            "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1" if trainer_world_size == 4 else "1,1,1",
            "CHIPS_PER_HOST": "4",
        })
        
    return env_vars, host_ip


def build_generator_env_vars(layout: dict, worker_pythonpath: str) -> dict:
    """
    Build environment variables dictionary for the generator.
    """
    chips_per_host = layout["chips_per_host"]
    sampler_nodes_info = layout["sampler_nodes_info"]
    generator_world_size = layout["generator_world_size"]
    generator_device_type = layout["generator_device_type"]
    
    generator_sb_addresses = []
    if sampler_nodes_info:
        for node in sampler_nodes_info:
            dns_name = node["NodeManagerAddress"]
            for chip_idx in range(chips_per_host):
                generator_sb_addresses.append(f"{dns_name}:{GENERATOR_BASE_PORT + chip_idx}")
    else:
        for chip_idx in range(chips_per_host):
            generator_sb_addresses.append(f"127.0.0.1:{GENERATOR_BASE_PORT + chip_idx}")

    generator_env_vars = {
        "TORCHTITAN_DEVICE_TYPE": generator_device_type,
        "SKIP_JAX_PRECOMPILE": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": VLLM_ENABLE_V1_MULTIPROCESSING,
        "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
        "TPU_HOST_BOUNDS": "2,4,1" if generator_world_size == 8 else ("2,2,1" if generator_world_size == 4 else "1,1,1"),
        "TORCH_TPU_TOPOLOGY": "2,4,1" if generator_world_size == 8 else ("2,2,1" if generator_world_size == 4 else "1,1,1"),
        "CHIPS_PER_HOST": "4",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(generator_sb_addresses),
        "TPU_PROCESS_ADDRESSES": ",".join(generator_sb_addresses),
        "LIBTPU_INIT_ARGS": LIBTPU_INIT_ARGS,
        "TORCH_DYNAMO_RECOMPILE_LIMIT": str(TORCH_DYNAMO_RECOMPILE_LIMIT),
        "PYTHONPATH": worker_pythonpath,
    }
    
    if generator_world_size > 1:
        generator_env_vars["TPU_MULTIHOST_BACKEND"] = "ray"
        
    return generator_env_vars
