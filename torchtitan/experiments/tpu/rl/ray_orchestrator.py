import sys
import ray
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.tpu.rl.ray_worker import FusedWorker

class GRPOOrchestrator:
    """
    Ray Driver (Orchestrator) for TPU-based GRPO Training.
    
    This class runs on the CPU head node and drives the high-level Reinforcement 
    Learning training loop. It coordinates a fleet of remote TPU workers (Ray Actors, one per TPU chip) 
    that perform parallel generation (vLLM) and policy optimization (FSDP). 
    They are called FusedWorker, becuase they do both training and sampling.

    Architecture:
                  ┌────────────────────────┐
                  │   GRPOOrchestrator     │ (Ray Driver / Head Node)
                  │   - Drives step loop   │
                  │   - Global GRPO Math   │
                  └───────────┬────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    ▼                         ▼                         ▼
  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
  │   FusedWorker    │      │   FusedWorker    │      │   FusedWorker    │ ... 
  │  - PyTorch FSDP  │      │  - PyTorch FSDP  │      │  - PyTorch FSDP  │
  │  - vLLM Engine   │      │  - vLLM Engine   │      │  - vLLM Engine   │
  │  - TPU Chip 0    │      │  - TPU Chip 1    │      │  - TPU Chip 2    │
  └──────────────────┘      └──────────────────┘      └──────────────────┘
    """
    def __init__(self, sys_argv, world_size, tpu_resources, master_addr, master_port, sb_addresses):
        self.sys_argv = sys_argv
        self.world_size = world_size
        self.tpu_resources = tpu_resources
        self.master_addr = master_addr
        self.master_port = master_port
        self.sb_addresses = sb_addresses
        self.workers = []
        
        # Parse torchtitan configs
        config_manager = ConfigManager()
        self.job_config = config_manager.parse_args(args=self.sys_argv[1:])
        
    def setup_workers(self):
        """Spawns Ray Actor workers for each physical TPU chip."""
        for rank in range(self.world_size):
            # TODO: config those env var automatically
            env_vars = {
                "WORLD_SIZE": str(self.world_size),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "MASTER_ADDR": self.master_addr,
                "MASTER_PORT": str(self.master_port),
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": self.sb_addresses,
                
                # Disable eager memory pre-allocation to prevent OOM
                "JAX_MEM_FRACTION": "0.45",
                "JAX_THREE_G_MEM_ALLOC_ON_FREE": "true",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.45",
                
                # TPU Topology and configuration for v6e-4 VM
                # TODO: support other topology
                "TORCH_TPU_TOPOLOGY": "2,2,1",
                "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
            }
            
            actor_options = {
                "runtime_env": {"env_vars": env_vars},
            }
            if self.tpu_resources:
                actor_options["resources"] = self.tpu_resources

            print(f"@@@ Spawning FusedWorker Rank {rank} with env: {env_vars}")
            worker = FusedWorker.options(**actor_options).remote(self.sys_argv)
            self.workers.append(worker)

    def run(self):
        """Main training loop."""
        self.setup_workers()

        print("\n@@@ Initializing models on all workers...")

        # NOTE on Parallelism:
        # Calling `.remote()` dispatches the task asynchronously and instantly returns a Future.
        # The list comprehension `[w.method.remote() for w in self.workers]` fires off the 
        # task to all workers simultaneously, achieving true parallel execution.
        # `ray.get(...)` then acts as a barrier synchronization, pausing the driver until 
        # all workers have completed their tasks.
        init_futures = [worker.init_models.remote() for worker in self.workers]
        ray.get(init_futures)
        
        print("\n@@@ Checking heartbeat on all workers...")
        heartbeat_futures = [worker.heartbeat.remote() for worker in self.workers]
        ray.get(heartbeat_futures)

        print("\n@@@ Starting training loop on driver...")
        steps = self.job_config.training.steps
        
        try:
            for step in range(steps):
                print(f"@@@ Driver initiating Step {step + 1}/{steps} ...")
                
                # 1. Load prompts on workers locally (handles DP/TP correctly)
                ray.get([w.load_next_batch.remote() for w in self.workers])

                # 2. Parallel Rollout Generation on Workers
                print(f"    - Parallel Rollout Generation...")
                rollouts = ray.get([w.generate_rollouts.remote() for w in self.workers])
                
                # 3. Compute reference log-probs
                print(f"    - Computing Reference Log Probs...")
                ray.get([w.compute_ref_log_probs.remote() for w in self.workers])
                
                # 4. Trigger PPO Steps on Workers (Advantages are computed locally per-prompt)
                print(f"    - Triggering PPO Epochs (Local Advantage Math)...")
                metrics = ray.get([
                    w.train_ppo_step.remote(step) 
                    for w in self.workers
                ])
                
                # 5. Weight Sync
                print(f"    - Syncing weights...")
                ray.get([w.sync_weights.remote() for w in self.workers])
                
            finish_futures = [worker.finish.remote() for worker in self.workers]
            ray.get(finish_futures)
            print("\n@@@ Training completed successfully!")
        except Exception as e:
            print("\n@@@ Training failed with exception:", e)



