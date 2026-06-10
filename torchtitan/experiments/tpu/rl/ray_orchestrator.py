import sys
import ray
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.tpu.rl.ray_worker import FusedWorker

DEFAULT_TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"

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
        num_workers = len(self.workers)
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
        init_futures = [worker.init_models.remote() for worker in self.workers]
        ray.get(init_futures)
        
        print("\n@@@ Checking heartbeat on all workers...")
        heartbeat_futures = [worker.heartbeat.remote() for worker in self.workers]
        ray.get(heartbeat_futures)

        print("\n@@@ Initializing Environment and Tokenizer on Driver...")
        import torch
        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.experiments.rl.sum_digits import SumDigitsEnv
        
        # We instantiate a local tokenizer on the driver to encode the prompts
        tokenizer_path = getattr(self.job_config, "hf_assets_path", DEFAULT_TOKENIZER_NAME)
        if not tokenizer_path:
            tokenizer_path = DEFAULT_TOKENIZER_NAME
        tokenizer = HuggingFaceTokenizer(tokenizer_path=tokenizer_path)

        print("\n@@@ Starting training loop on driver...")
        steps = self.job_config.training.steps
        max_new_tokens = self.job_config.sampler.max_new_tokens
        seq_len = self.job_config.training.seq_len
        local_batch_size = self.job_config.training.local_batch_size
        group_size = self.job_config.grpo.group_size
        global_prompt_batch_size = getattr(self.job_config.grpo, "global_prompt_batch_size", 16)
        num_workers = len(self.workers)
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
                ray.get([w.load_next_batch.remote(batch_prompt_ids) for w, batch_prompt_ids in zip(self.workers, prompt_ids_list)])

                # 2. Parallel Rollout Generation on Workers
                print(f"    - Parallel Rollout Generation...")
                rollouts = ray.get([w.generate_rollouts.remote() for w in self.workers])
                
                # 3. Compute reference log-probs
                print(f"    - Computing Reference Log Probs...")
                ray.get([w.compute_ref_log_probs.remote() for w in self.workers])
                
                # 4. Centralized Reward Scoring & Global Advantage Math
                print(f"    - Centralized Reward Scoring & Global Advantage Math...")
                advantages_list, avg_correctness, avg_format, avg_reward, sample_prompt, sample_completion = self.compute_global_grpo_advantages(rollouts, step, tokenizer)
                print(f"\n[titan] @@@ Sample Output Step {step + 1} @@@\nPrompt: {sample_prompt}\nCompletion:\n{sample_completion}\n===========================\n")
                print(f"\n[titan] @@@ Step {step + 1} Global Average Reward: {avg_reward:.4f} @@@\n")

                # 5. Trigger PPO Steps on Workers
                print(f"    - Triggering PPO Epochs...")
                metrics = ray.get([
                    w.train_ppo_step.remote(advantages_list[i], step, avg_correctness, avg_format) 
                    for i, w in enumerate(self.workers)
                ])
                
                m = metrics[0]
                log_freq = getattr(getattr(self.job_config, "metrics", None), "log_freq", 1)
                should_log = (step + 1) % log_freq == 0 or step == (steps - 1)
                if should_log:
                    print(f"\n[titan] @@@ Step {step + 1} Metrics @@@")
                    print(f"Loss: {m['loss']:.4f} | Grad Norm: {m['grad_norm']:.4f}")
                    print(f"Times: Sample={m['extra_metrics']['sampling_time']:.2f}s, Ref={m['extra_metrics']['ref_time']:.2f}s, Train={m['extra_metrics']['train_time']:.2f}s, Total={m['extra_metrics']['total_step_time']:.2f}s")
                    print(f"Reward: Format={avg_format:.4f}, Correctness={avg_correctness:.4f}\n")
                
                # 6. Weight Sync
                print(f"    - Syncing weights...")
                ray.get([w.sync_weights.remote() for w in self.workers])
                
            finish_futures = [worker.finish.remote() for worker in self.workers]
            ray.get(finish_futures)
            print("\n@@@ Training completed successfully!")
        except Exception as e:
            import traceback
            print("\n@@@ Training failed with exception:")
            traceback.print_exc()



