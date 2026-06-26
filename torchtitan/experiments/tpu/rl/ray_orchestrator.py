import sys
import ray
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.tpu.rl.ray_worker import FusedWorker

DEFAULT_TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"

# Global TPU Hardware Topology Registry
TPU_TOPOLOGY_REGISTRY = {
    # Google Cloud TPU v6e - v6e-16 slice (4x v6e-4 hosts)
    "v6e-16": {
        "TORCH_TPU_TOPOLOGY": "4,4,1",
        "TPU_HOST_BOUNDS": "4,4,1",
        "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
        "CHIPS_PER_HOST": "4"
    },
    # Google Cloud TPU v6e - v6e-8 slice (2x v6e-4 hosts)
    "v6e-8": {
        "TORCH_TPU_TOPOLOGY": "2,4,1",
        "TPU_HOST_BOUNDS": "2,2,1",
        "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
        "CHIPS_PER_HOST": "4"
    },
    # Google Cloud TPU v6e - v6e-4 local vm (1x v6e-4 host)
    "v6e-4": {
        "TORCH_TPU_TOPOLOGY": "2,2,1",
        "TPU_HOST_BOUNDS": "1,1,1",
        "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
        "CHIPS_PER_HOST": "4"
    }
}


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
    def __init__(self, sys_argv, world_size, tpu_resources, master_addr, master_port, sb_addresses, tpu_nodes, tpu_type, sampler_world_size=None, sampler_sb_addresses=None, sampler_tpu_nodes=None):
        self.sys_argv = sys_argv
        self.world_size = world_size
        self.sampler_world_size = sampler_world_size if sampler_world_size is not None else world_size
        self.tpu_resources = tpu_resources
        self.master_addr = master_addr
        self.master_port = master_port
        self.sb_addresses = sb_addresses
        self.sampler_sb_addresses = sampler_sb_addresses if sampler_sb_addresses is not None else sb_addresses
        self.tpu_nodes = tpu_nodes
        self.sampler_tpu_nodes = sampler_tpu_nodes if sampler_tpu_nodes is not None else tpu_nodes
        self.tpu_type = tpu_type
        
        # We might have separate trainers and samplers
        self.trainers = []
        self.samplers = []
        self.is_collocated = (self.sb_addresses == self.sampler_sb_addresses)
        
        if not self.is_collocated:
            assert self.world_size == self.sampler_world_size, "Noncolocated mode requires identical slice sizes for Trainer and Sampler."
        
        # Parse torchtitan configs
        config_manager = ConfigManager()
        self.job_config = config_manager.parse_args(args=self.sys_argv[1:])
        
    def _spawn_workers(self, role, world_size, sb_addresses, tpu_nodes, master_port_offset=0):
        unique_hostnames = list(dict.fromkeys([addr.split(":")[0] for addr in sb_addresses]))
        tpu_worker_hostnames = ",".join(unique_hostnames)
        rank0_hostname = unique_hostnames[0]

        topology_config = TPU_TOPOLOGY_REGISTRY.get(self.tpu_type, TPU_TOPOLOGY_REGISTRY["v6e-4"])
        chips_per_host = int(topology_config.get("CHIPS_PER_HOST", 4))

        workers = []
        for rank in range(world_size):
            local_chip_index = rank % chips_per_host
            process_port = 8471 + local_chip_index
            tpu_worker_addresses_str = ",".join(sb_addresses) if isinstance(sb_addresses, list) else sb_addresses

            env_vars = {
                "WORKER_ROLE": role,
                "IS_NONCOLLOCATED": "1" if not self.is_collocated else "0",
                "WORLD_SIZE": str(world_size),
                "RANK": str(rank),
                "LOCAL_RANK": str(local_chip_index),
                "MASTER_ADDR": rank0_hostname,
                "MASTER_PORT": str(self.master_port + master_port_offset),
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": tpu_worker_addresses_str,
                
                "TPU_PROCESS_ADDRESSES": tpu_worker_addresses_str,
                "TPU_PROCESS_PORT": str(process_port),
                "CLOUD_TPU_TASK_ID": str(rank),
                "TPU_WORKER_HOSTNAMES": tpu_worker_hostnames,

                "JAX_MEM_FRACTION": "0.45",
                "JAX_THREE_G_MEM_ALLOC_ON_FREE": "true",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.45",
            }
            
            env_vars.update(topology_config)
            
            actor_options = {
                "runtime_env": {"env_vars": env_vars},
            }
            if self.tpu_resources:
                host_index = rank // chips_per_host
                if host_index >= len(tpu_nodes):
                    raise ValueError(
                        f"Not enough TPU nodes: rank {rank} requires host {host_index}, "
                        f"but only {len(tpu_nodes)} nodes available."
                    )
                node_ip = tpu_nodes[host_index]
                actor_options["resources"] = {
                    "TPU": 1,
                    f"node:{node_ip}": 0.01,
                }

            print(f"@@@ Spawning FusedWorker Rank {rank} ({role}) with env: {env_vars}")
            worker = FusedWorker.options(**actor_options).remote(self.sys_argv)
            workers.append(worker)
        return workers

    def setup_workers(self):
        """Spawns Ray Actor workers for each physical TPU chip."""
        self.trainers = self._spawn_workers(
            "Trainer", 
            self.world_size, 
            self.sb_addresses, 
            self.tpu_nodes, 
            master_port_offset=0
        )
        
        if self.is_collocated:
            self.samplers = self.trainers
        else:
            self.samplers = self._spawn_workers(
                "Sampler", 
                self.sampler_world_size, 
                self.sampler_sb_addresses, 
                self.sampler_tpu_nodes, 
                master_port_offset=100  # Offset master port to avoid collision if on same node (though they are on diff nodes)
            )

    def compute_global_grpo_advantages(self, rollout_results, step, tokenizer):
        """
        Computes group-relative advantages globally across all completions generated
        by the cluster, guaranteeing mathematical parity with single-node reference.
        """
        import torch
        from torchtitan.experiments.rl.sum_digits import SumDigitsEnv
        
        all_rewards = []
        all_correctness = []
        all_format = []
        group_size = self.job_config.grpo.group_size
        global_prompt_batch_size = getattr(self.job_config.grpo, "global_prompt_batch_size", 16)
        num_workers = len(self.samplers)
        prompts_per_worker = global_prompt_batch_size // num_workers
        
        sample_prompt = ""
        sample_completion = ""
        for w_idx, res in enumerate(rollout_results):
            completed_ids = res["completed_ids"]
            batch_size = completed_ids.shape[0]
            rewards = []
            for i in range(batch_size):
                b_idx = i // group_size
                env_config = SumDigitsEnv.Config()
                env = SumDigitsEnv(env_config, step=step, group_idx=w_idx * prompts_per_worker + b_idx)
                
                # Decode to text to calculate reward
                target_len = self.job_config.training.seq_len - self.job_config.sampler.max_new_tokens
                completion_text = tokenizer.decode(completed_ids[i][target_len:].tolist(), skip_special_tokens=True)
                
                if w_idx == 0 and i == 0:
                    sample_prompt = env.prompt
                    sample_completion = completion_text
                
                step_result = env.step(completion_text)
                
                # Sum the specific reward components
                correctness = step_result.rewards.get("correctness", 0.0)
                format_r = step_result.rewards.get("format", 0.0)
                reward = correctness + format_r
                rewards.append(reward)
                all_correctness.append(correctness)
                all_format.append(format_r)
                
            all_rewards.append(torch.tensor(rewards, device='cpu', dtype=torch.float32))
            
        # Concatenate all rewards
        global_rewards = torch.cat(all_rewards, dim=0)
        
        avg_reward = global_rewards.mean().item()
        avg_correctness_val = sum(all_correctness) / len(all_correctness) if all_correctness else 0.0
        avg_format_val = sum(all_format) / len(all_format) if all_format else 0.0
        
        # Centralized GRPO Math: computing the mean and std across the entire cluster
        global_rewards_grouped = global_rewards.view(-1, group_size)
        global_mean = global_rewards_grouped.mean(dim=1, keepdim=True)
        global_std = global_rewards_grouped.std(dim=1, unbiased=False, keepdim=True)
        
        global_advantages_grouped = (global_rewards_grouped - global_mean) / (global_std + 1e-4)
        global_advantages = global_advantages_grouped.view(-1)
        
        # Distribute advantages back to workers
        worker_advantages = []
        offset = 0
        for r in all_rewards:
            size = r.shape[0]
            worker_advantages.append(global_advantages[offset:offset+size])
            offset += size
            
        return worker_advantages, avg_correctness_val, avg_format_val, avg_reward, sample_prompt, sample_completion

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
        all_actors = self.trainers + (self.samplers if not self.is_collocated else [])
        init_futures = [worker.init_models.remote() for worker in all_actors]
        ray.get(init_futures)
        
        print("\n@@@ Checking heartbeat on all workers...")
        heartbeat_futures = [worker.heartbeat.remote() for worker in all_actors]
        ray.get(heartbeat_futures)

        print("\n@@@ Initializing Environment and Tokenizer on Driver...")
        import torch
        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.experiments.rl.sum_digits import SumDigitsEnv
        
        # We instantiate a local tokenizer on the driver to encode the prompts
        tokenizer_path = getattr(self.job_config.checkpoint, "initial_load_path", DEFAULT_TOKENIZER_NAME)
        import os
        if not tokenizer_path or not os.path.exists(tokenizer_path):
            tokenizer_path = DEFAULT_TOKENIZER_NAME
        tokenizer = HuggingFaceTokenizer(tokenizer_path=tokenizer_path)

        # Initial Weight Sync for Non-Collocated Samplers
        # The Trainer loads the checkpoint during init_models, but the Sampler does not.
        # We must sync the pretrained weights from Trainer to Sampler BEFORE the first step!
        if not self.is_collocated:
            print("\n@@@ Performing initial weight sync to Samplers...")
            state_dicts = ray.get([w.get_full_state_dict.remote() for w in self.trainers])
            full_sd = next((sd for sd in state_dicts if sd is not None), None)
            if full_sd is not None:
                ray.get([w.load_full_state_dict.remote(full_sd) for w in self.samplers])

        print("\n@@@ Starting training loop on driver...")
        steps = self.job_config.training.steps
        max_new_tokens = self.job_config.sampler.max_new_tokens
        seq_len = self.job_config.training.seq_len
        local_batch_size = self.job_config.training.local_batch_size
        group_size = self.job_config.grpo.group_size
        global_prompt_batch_size = getattr(self.job_config.grpo, "global_prompt_batch_size", 16)
        num_workers = len(self.samplers)
        prompts_per_worker = global_prompt_batch_size // num_workers
        
        try:
            for step in range(steps):
                print(f"@@@ Driver initiating Step {step + 1}/{steps} ...")
                
                # 1. Fetch prompts from environment (runs on Driver/CPU)
                prompt_ids_list = []
                ds = self.job_config.training.dataset.lower() 
                if ds == "random":
                    # Use random dataset for debug only
                    print("Using random dataset for debugging...")
                    import random
                    vocab_size = tokenizer.get_vocab_size()
                    target_len = seq_len - max_new_tokens
                    for w_idx in range(num_workers):
                        batch_prompt_ids = []
                        for b_idx in range(prompts_per_worker):
                            tokens = [random.randint(0, vocab_size - 1) for _ in range(target_len)]
                            batch_prompt_ids.append(tokens)
                        prompt_ids_list.append(batch_prompt_ids)
                elif ds.startswith("sumdigits"):
                    for w_idx in range(num_workers):
                        batch_prompt_ids = []
                        for b_idx in range(prompts_per_worker):
                            env_config = SumDigitsEnv.Config()
                            env = SumDigitsEnv(env_config, step=step, group_idx=w_idx * prompts_per_worker + b_idx)
                            tokens = tokenizer.encode(env.prompt)
                            # Left pad with EOS token (or 0)
                            target_len = seq_len - max_new_tokens
                            if len(tokens) > target_len:
                                tokens = tokens[-target_len:]
                            else:
                                eos_id = getattr(tokenizer, "eos_id", 0)
                                # tokens = [eos_id] * (target_len - len(tokens)) + tokens  # <- This causes hang in step 2 
                                # HACK: pad sequence to be [random tokens] + [EOS] + [prompt tokens]
                                tokens = [eos_id] + tokens
                                if len(tokens) < target_len:
                                    import random
                                    vocab_size = tokenizer.get_vocab_size()
                                    pad_tokens = [random.randint(0, vocab_size - 1) for _ in range(target_len - len(tokens))]
                                    tokens = pad_tokens + tokens

                            batch_prompt_ids.append(tokens)
                        prompt_ids_list.append(batch_prompt_ids)
                else:
                    raise ValueError("Only sumdigits or random datasets are supported!")
                
                # 1b. Load prompts on workers
                ray.get([w.load_next_batch.remote(batch_prompt_ids) for w, batch_prompt_ids in zip(self.samplers, prompt_ids_list)])
                if not self.is_collocated:
                    ray.get([w.load_next_batch.remote(batch_prompt_ids) for w, batch_prompt_ids in zip(self.trainers, prompt_ids_list)])

                # 2. Parallel Rollout Generation on Workers
                print(f"    - Parallel Rollout Generation...")
                rollouts = ray.get([w.generate_rollouts.remote() for w in self.samplers])
                
                # If non-collocated, we must transfer the rollouts to the trainers
                if not self.is_collocated:
                    print(f"    - Transferring Rollouts to Trainers...")
                    ray.get([
                        t.set_rollout_data.remote(r["completed_ids"], r["token_log_probs"], r["sampling_time"]) 
                        for t, r in zip(self.trainers, rollouts)
                    ])
                
                # 3. Compute reference log-probs
                print(f"    - Computing Reference Log Probs...")
                ray.get([w.compute_ref_log_probs.remote() for w in self.trainers])
                
                # 4. Centralized Reward Scoring & Global Advantage Math
                print(f"    - Centralized Reward Scoring & Global Advantage Math...")
                advantages_list, avg_correctness, avg_format, avg_reward, sample_prompt, sample_completion = self.compute_global_grpo_advantages(rollouts, step, tokenizer)
                print(f"\n[titan] @@@ Sample Output Step {step + 1} @@@\nPrompt: {sample_prompt}\nCompletion:\n{sample_completion}\n===========================\n")
                print(f"\n[titan] @@@ Step {step + 1} Global Average Reward: {avg_reward:.4f} @@@\n")

                # 5. Trigger PPO Steps on Workers
                print(f"    - Triggering PPO Epochs...")
                metrics = ray.get([
                    w.train_ppo_step.remote(advantages_list[i], step, avg_correctness, avg_format) 
                    for i, w in enumerate(self.trainers)
                ])
                
                m = metrics[0]
                log_freq = getattr(getattr(self.job_config, "metrics", None), "log_freq", 1)
                should_log = (step + 1) % log_freq == 0 or step == (steps - 1)
                if should_log:
                    print(f"\n[titan] @@@ Step {step + 1} Metrics @@@")
                    print(f"[titan] Loss: {m['loss']:.4f} | Grad Norm: {m['grad_norm']:.4f}")
                    print(f"[titan] Times: Sample={m['extra_metrics']['sampling_time']:.2f}s, Ref={m['extra_metrics']['ref_time']:.2f}s, Train={m['extra_metrics']['train_time']:.2f}s, Total={m['extra_metrics']['total_step_time']:.2f}s")
                    print(f"[titan] Reward: Format={avg_format:.4f}, Correctness={avg_correctness:.4f}\n")
                
                # 6. Weight Sync
                print(f"    - Syncing weights...")
                
                # Fetch weights from trainers and sync to samplers if non-collocated
                if not self.is_collocated:
                    # In a real DCN environment we'd serialize the weights from Trainer Rank 0 and send to Sampler Rank 0, 
                    # but with Ray we can rely on ray.get/ray.put to handle the object store transfers.
                    # First, we fetch the full sharded state dict from rank 0 of the trainers.
                    state_dicts = ray.get([w.get_full_state_dict.remote() for w in self.trainers])
                    # rank 0 state dict is not None
                    full_sd = next((sd for sd in state_dicts if sd is not None), None)
                    if full_sd is not None:
                        # Then, we push it to all the samplers. They will shard it internally via FSDP if necessary.
                        ray.get([w.load_full_state_dict.remote(full_sd) for w in self.samplers])
                else:
                    # If collocated, the worker can simply copy the weights from the trainer model to the sampler model internally
                    ray.get([w.sync_weights.remote() for w in self.trainers])
                
            finish_futures = [worker.finish.remote() for worker in self.trainers + (self.samplers if not self.is_collocated else [])]
            ray.get(finish_futures)
            print("\n@@@ Training completed successfully!")
        except Exception as e:
            import traceback
            print("\n@@@ Training failed with exception:")
            traceback.print_exc()
