"""
Dynamic distributed RL training on TPU with Ray.
"""

import sys
import socket
import ray
from torchtitan.experiments.tpu.rl.ray_orchestrator import GRPOOrchestrator

PJRT_BASE_PORT = 8471

def main():
    is_noncolocated = "--noncolocated" in sys.argv
    if is_noncolocated:
        sys.argv.remove("--noncolocated")
        print("@@@ Running in NONCOLLOCATED mode (separate Trainer/Sampler slices)")
    else:
        print("@@@ Running in COLLOCATED mode (Trainer/Sampler on same slice)")

    print("@@@ Initializing Ray...")
    ray.init()
    print(f"@@@ Ray Cluster Resources: {ray.cluster_resources()}")

    # Get all TPU nodes
    all_tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {})]

    # Filter nodes by the chosen slice name to handle multi-slice clusters!
    if all_tpu_nodes:
        # Group by slice name
        slices = {}
        for n in all_tpu_nodes:
            slice_name = n.get("Labels", {}).get("ray.io/tpu-slice-name", "default")
            slices.setdefault(slice_name, []).append(n)
        
        slice_names = list(slices.keys())
        # Pick the first slice for trainer
        trainer_slice = slice_names[0]
        trainer_nodes_info = slices[trainer_slice]
        
        if is_noncolocated and len(slice_names) > 1:
            sampler_slice = slice_names[1]
            sampler_nodes_info = slices[sampler_slice]
        else:
            if is_noncolocated:
                print("@@@ WARNING: --noncolocated requested but only 1 slice found! Falling back to collocated mode.")
            sampler_slice = trainer_slice
            sampler_nodes_info = trainer_nodes_info
        
        tpu_type = trainer_nodes_info[0].get("Labels", {}).get("ray.io/tpu-pod-type")
        
        trainer_tpu_count = int(sum([n.get("Resources", {}).get("TPU", 0) for n in trainer_nodes_info]))
        sampler_tpu_count = int(sum([n.get("Resources", {}).get("TPU", 0) for n in sampler_nodes_info]))
        print(f"@@@ Selected Trainer slice: {trainer_slice} with {len(trainer_nodes_info)} hosts")
        print(f"@@@ Selected Sampler slice: {sampler_slice} with {len(sampler_nodes_info)} hosts")
    else:
        trainer_nodes_info = []
        sampler_nodes_info = []
        trainer_tpu_count = 0
        sampler_tpu_count = 0
        tpu_type = None

    trainer_world_size = trainer_tpu_count if trainer_tpu_count > 0 else 4
    sampler_world_size = sampler_tpu_count if sampler_tpu_count > 0 else 4
    if trainer_tpu_count == 0:
        print(f"Warning: No TPU resources detected by Ray, defaulting to trainer_world_size={trainer_world_size}.")
    else:
        print(f"Detected {trainer_world_size} TPUs in trainer slice and {sampler_world_size} in sampler slice.")

    # one worker per chip is the standard for Ray
    tpu_resources = {"TPU": 1} if trainer_tpu_count > 0 else {}

    master_addr = ray.util.get_node_ip_address()
    # Use a fixed port that is whitelisted/allowed in network policies across TPU VM hosts.
    master_port = 8450

    # Sort them strictly by their GKE tpu-worker-id label to guarantee Rank alignment!
    trainer_nodes_info.sort(key=lambda n: int(n["Labels"].get("ray.io/tpu-worker-id", "0")))
    trainer_node_ips = [node["NodeManagerAddress"] for node in trainer_nodes_info]
    
    sampler_nodes_info.sort(key=lambda n: int(n["Labels"].get("ray.io/tpu-worker-id", "0")))
    sampler_node_ips = [node["NodeManagerAddress"] for node in sampler_nodes_info]

    # Dynamically resolve chips per host from the first TPU node resources
    chips_per_host = int(trainer_nodes_info[0]["Resources"].get("TPU", 4)) if trainer_nodes_info else 4
    
    # Use GKE's headless service DNS names to satisfy libtpu routing
    headless_service = "ray-tpu-cluster-headless"
    trainer_sb_addresses = []
    for node in trainer_nodes_info:
        # Always use NodeManagerAddress (IP) instead of DNS names which may fail to resolve
        dns_name = node["NodeManagerAddress"]
        for chip_idx in range(chips_per_host):
            # 8471 is the standard base port used by PJRT/libtpu for TPU process communication.
            # We assign consecutive ports for each local chip on the host.
            port = PJRT_BASE_PORT + chip_idx
            trainer_sb_addresses.append(f"{dns_name}:{port}")

    sampler_sb_addresses = []
    for node in sampler_nodes_info:
        dns_name = node["NodeManagerAddress"]
        for chip_idx in range(chips_per_host):
            port = PJRT_BASE_PORT + chip_idx
            sampler_sb_addresses.append(f"{dns_name}:{port}")

    print(f"@@@ Trainer SliceBuilder addresses: {trainer_sb_addresses}")
    print(f"@@@ Sampler SliceBuilder addresses: {sampler_sb_addresses}")

    if tpu_type is None:
        raise ValueError(
            "Could not detect TPU pod type ('ray.io/tpu-pod-type') from Ray cluster nodes. "
            "Please ensure your TPU cluster is properly configured and active."
        )
    print(f"@@@ Resolved dynamic TPU type: {tpu_type}")

    orchestrator = GRPOOrchestrator(
        sys_argv=sys.argv,
        world_size=trainer_world_size,
        sampler_world_size=sampler_world_size,
        tpu_resources=tpu_resources,
        master_addr=master_addr,
        master_port=master_port,
        sb_addresses=trainer_sb_addresses,
        sampler_sb_addresses=sampler_sb_addresses,
        tpu_nodes=trainer_node_ips,
        sampler_tpu_nodes=sampler_node_ips,
        tpu_type=tpu_type,
    )
    
    try:
        orchestrator.run()
    finally:
        print("\n@@@ Shutting down Ray...")
        ray.shutdown()

if __name__ == "__main__":
    main()
