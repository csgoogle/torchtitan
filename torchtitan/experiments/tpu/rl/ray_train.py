"""
Dynamic distributed RL training on TPU with Ray.
"""

import sys
import socket
import ray
from torchtitan.experiments.tpu.rl.ray_orchestrator import GRPOOrchestrator

def main():
    print("@@@ Initializing Ray...")
    ray.init()
    print(f"@@@ Ray Cluster Resources: {ray.cluster_resources()}")

    # Get all TPU nodes
    all_tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {})]

    # Filter nodes by the chosen slice name to handle multi-slice clusters!
    # In a Kubernetes (GKE) environment, KubeRay often groups multiple independent TPU slices 
    # under a single Ray cluster for easier management. If we blindly aggregate the 'TPU' 
    # resources from all nodes, we'll get a massive world_size (e.g., 5 slices of v6e-8 = 40 TPUs) 
    # and the script will try to launch a single FSDP group spanning across physically disjoint 
    # network slices, which will instantly hang or crash because ICI (Inter-Chip Interconnect) 
    # cannot span across different slices.
    # 
    # By grouping nodes using the 'ray.io/tpu-slice-name' label, we ensure the training job
    # runs cohesively inside exactly one physical slice (e.g., world_size = 8).
    if all_tpu_nodes:
        # Group by slice name
        slices = {}
        for n in all_tpu_nodes:
            slice_name = n.get("Labels", {}).get("ray.io/tpu-slice-name", "default")
            slices.setdefault(slice_name, []).append(n)
        
        # Pick the first slice
        chosen_slice = list(slices.keys())[0]
        tpu_nodes_info = slices[chosen_slice]
        
        # Sum the TPU count ONLY within the chosen slice
        tpu_count = int(sum([n.get("Resources", {}).get("TPU", 0) for n in tpu_nodes_info]))
        tpu_type = tpu_nodes_info[0].get("Labels", {}).get("ray.io/tpu-pod-type")
        print(f"@@@ Selected TPU slice: {chosen_slice} with {len(tpu_nodes_info)} hosts")
    else:
        tpu_nodes_info = []
        tpu_count = 0
        tpu_type = None

    world_size = tpu_count if tpu_count > 0 else 4
    if tpu_count == 0:
        print(f"Warning: No TPU resources detected by Ray, defaulting to world_size={world_size}.")
    else:
        print(f"Detected {world_size} TPUs in selected slice.")

    # one worker per chip is the standard for Ray
    tpu_resources = {"TPU": 1} if tpu_count > 0 else {}

    master_addr = ray.util.get_node_ip_address()
    # Use a fixed port that is whitelisted/allowed in network policies across TPU VM hosts.
    master_port = 8450

    # Sort them strictly by their GKE tpu-worker-id label to guarantee Rank alignment!
    tpu_nodes_info.sort(key=lambda n: int(n["Labels"].get("ray.io/tpu-worker-id", "0")))
    tpu_node_ips = [node["NodeManagerAddress"] for node in tpu_nodes_info]
    # Dynamically resolve chips per host from the first TPU node resources
    chips_per_host = int(tpu_nodes_info[0]["Resources"].get("TPU", 4)) if tpu_nodes_info else 4
    
    # Use GKE's headless service DNS names to satisfy libtpu routing
    headless_service = "ray-tpu-cluster-headless"
    sb_addresses = []
    for node in tpu_nodes_info:
        # Always use NodeManagerAddress (IP) instead of DNS names which may fail to resolve
        dns_name = node["NodeManagerAddress"]
        for chip_idx in range(chips_per_host):
            # 8471 is the standard base port used by PJRT/libtpu for TPU process communication.
            # We assign consecutive ports for each local chip on the host.
            port = 8471 + chip_idx
            sb_addresses.append(f"{dns_name}:{port}")

    print(f"@@@ SliceBuilder addresses: {sb_addresses}")

    if tpu_type is None:
        raise ValueError(
            "Could not detect TPU pod type ('ray.io/tpu-pod-type') from Ray cluster nodes. "
            "Please ensure your TPU cluster is properly configured and active."
        )
    print(f"@@@ Resolved dynamic TPU type: {tpu_type}")

    orchestrator = GRPOOrchestrator(
        sys_argv=sys.argv,
        world_size=world_size,
        tpu_resources=tpu_resources,
        master_addr=master_addr,
        master_port=master_port,
        sb_addresses=sb_addresses,
        tpu_nodes=tpu_node_ips,
        tpu_type=tpu_type,
    )
    
    try:
        orchestrator.run()
    finally:
        print("\n@@@ Shutting down Ray...")
        ray.shutdown()

if __name__ == "__main__":
    main()
