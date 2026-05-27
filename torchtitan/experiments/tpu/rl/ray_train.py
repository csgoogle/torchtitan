import sys
import socket
import ray
from torchtitan.experiments.tpu.rl.ray_orchestrator import GRPOOrchestrator

def get_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]

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
    master_port = get_free_port()

    sb_ports = [get_free_port() for _ in range(world_size)]
    sb_addresses = ",".join([f"localhost:{p}" for p in sb_ports])
    print(f"@@@ SliceBuilder addresses: {sb_addresses}")

    orchestrator = GRPOOrchestrator(
        sys_argv=sys.argv,
        world_size=world_size,
        tpu_resources=tpu_resources,
        master_addr=master_addr,
        master_port=master_port,
        sb_addresses=sb_addresses,
    )
    
    try:
        orchestrator.run()
    finally:
        print("\n@@@ Shutting down Ray...")
        ray.shutdown()

if __name__ == "__main__":
    main()
