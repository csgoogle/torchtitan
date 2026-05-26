import os
import sys
import socket
import ray
from torchtitan.config.manager import ConfigManager

# Force line-buffering on stdout and stderr
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def get_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]

@ray.remote
class TrainerActor:
    def __init__(self, sys_argv):
        self.sys_argv = sys_argv

    def run(self):
        # Override sys.argv so any internal library (like tyro) parsing it works
        sys.argv = self.sys_argv
        
        # Pass the sys_argv from the launcher directly into the ConfigManager
        from torchtitan.experiments.tpu.rl.grpo_job_config import GRPOJobConfig
        config_manager = ConfigManager()
        job_config = config_manager.parse_args(args=self.sys_argv[1:])
        
        # Dynamically convert to GRPOJobConfig based on the config name if needed.
        # Actually, if we use --module torchtitan.experiments.tpu.rl, 
        # it will be correctly loaded from config_registry.py
        
        from torchtitan.experiments.tpu.rl.train_grpo import start_trainer
        start_trainer(job_config)
        return True

def main():
    print("@@@ Initializing Ray...")
    ray.init()
    print(f"@@@ Ray Cluster Resources: {ray.cluster_resources()}")

    # Determine world size based on available TPUs (assume 4 for v6e-4 if not specified)
    # Actually, we can check ray cluster resources or default to 4
    tpu_count = int(ray.cluster_resources().get("TPU", 0))
    world_size = tpu_count if tpu_count > 0 else 4
    if tpu_count == 0:
        print(f"Warning: No TPU resources detected by Ray, defaulting to world_size={world_size}.")
    else:
        print(f"Detected {world_size} TPUs.")

    tpu_resources = {"TPU": 1} if tpu_count > 0 else {}

    master_addr = ray.util.get_node_ip_address()
    master_port = get_free_port()

    sb_ports = [get_free_port() for _ in range(world_size)]
    sb_addresses = ",".join([f"localhost:{p}" for p in sb_ports])
    print(f"@@@ SliceBuilder addresses: {sb_addresses}")

    actors = []
    for rank in range(world_size):
        # TODO: setup those env var automatically
        env_vars = {
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": sb_addresses,
            
            # Disable eager memory pre-allocation to prevent OOM
            "JAX_MEM_FRACTION": "0.45",
            "JAX_THREE_G_MEM_ALLOC_ON_FREE": "true",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.45",
            
            # TPU Topology and configuration for v6e-4 VM
            # TODO: support other topology
            "TORCH_TPU_TOPOLOGY": "2,2,1",
            "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
        }
        
        actor_options = {
            "runtime_env": {"env_vars": env_vars},
        }
        if tpu_resources:
            actor_options["resources"] = tpu_resources

        print(f"@@@ Spawning TrainerActor Rank {rank} with env: {env_vars}")
        actor = TrainerActor.options(**actor_options).remote(sys.argv)
        actors.append(actor)

    print("\n@@@ Starting training on all actors...")
    futures = [actor.run.remote() for actor in actors]
    
    try:
        results = ray.get(futures)
        print("\n@@@ Training completed successfully:", results)
    except Exception as e:
        print("\n@@@ Training failed with exception:", e)
    finally:
        print("\n@@@ Shutting down Ray...")
        ray.shutdown()

if __name__ == "__main__":
    main()
