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

    # Determine world size based on available TPUs (assume 4 for v6e-4 if not specified)
    tpu_count = int(ray.cluster_resources().get("TPU", 0))
    world_size = tpu_count if tpu_count > 0 else 4
    if tpu_count == 0:
        print(f"Warning: No TPU resources detected by Ray, defaulting to world_size={world_size}.")
    else:
        print(f"Detected {world_size} TPUs.")

    # one worker per chip is the standard for Ray
    tpu_resources = {"TPU": 1} if tpu_count > 0 else {}

    master_addr = ray.util.get_node_ip_address()
    # Use a fixed port that is whitelisted/allowed in network policies across TPU VM hosts.
    master_port = 8450

    # Construct SliceBuilder addresses dynamically from TPU nodes in the Ray cluster
    tpu_nodes_info = [node for node in ray.nodes() if "TPU" in node["Resources"]]
    # Sort them strictly by their GKE tpu-worker-id label to guarantee Rank alignment!
    tpu_nodes_info.sort(key=lambda n: int(n["Labels"].get("ray.io/tpu-worker-id", "0")))
    tpu_node_ips = [node["NodeManagerAddress"] for node in tpu_nodes_info]
    # Dynamically resolve chips per host from the first TPU node resources
    chips_per_host = int(tpu_nodes_info[0]["Resources"].get("TPU", 4)) if tpu_nodes_info else 4
    
    # Use GKE's headless service DNS names to satisfy libtpu routing
    headless_service = "ray-tpu-cluster-headless"
    sb_addresses = []
    for node in tpu_nodes_info:
        replica_index = node["Labels"].get("replicaIndex", "tpu-group-0")
        worker_id = node["Labels"].get("ray.io/tpu-worker-id", "0")
        dns_name = f"{replica_index}-{worker_id}.{headless_service}"
        for chip_idx in range(chips_per_host):
            # 8471 is the standard base port used by PJRT/libtpu for TPU process communication.
            # We assign consecutive ports for each local chip on the host.
            port = 8471 + chip_idx
            sb_addresses.append(f"{dns_name}:{port}")

    print(f"@@@ SliceBuilder addresses: {sb_addresses}")

    # Dynamically resolve the physical TPU type from Ray cluster resources
    tpu_type = None
    for node in ray.nodes():
        if "TPU" in node.get("Resources", {}):
            tpu_type = node.get("Labels", {}).get("ray.io/tpu-pod-type")
            if tpu_type:
                break  # Found the slice type

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
