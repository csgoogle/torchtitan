import asyncio
import logging
import math
import time
import ray

from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.rl.grpo import RLTrainer, _mean_rewards, _format_rewards, _format_validation, _log_samples
from torchtitan.experiments.tpu.rl_ray.actors import RayPolicyTrainerWorker, RayVLLMGeneratorWorker

logger = logging.getLogger(__name__)

class RayRLTrainer(RLTrainer):
    """
    TPU & Ray orchestrated RL Trainer using 1-TPU Worker groups.
    """
    
    async def setup(self, **kwargs):
        config = self.config
        
        if not ray.is_initialized():
            ray.init()
            
        logger.info("Spawning 1-TPU Ray Worker Groups...")
        
        # We use Ray's node IP for master synchronization of PyTorch XLA process groups
        master_addr = ray.util.get_node_ip_address()
        self.world_size = 4 # Adjust as needed (e.g. 4 for TP-4/DP-4)
        
        # Trainer process group uses master port 29500
        self.trainer_workers = [
            RayPolicyTrainerWorker.remote(
                rank=i,
                world_size=self.world_size,
                master_addr=master_addr,
                master_port=29500,
                config=config.trainer,
                model_spec=config.model_spec,
                hf_assets_path=config.hf_assets_path,
                generator_dtype=config.generator.model_dtype,
                compile_config=config.compile,
            ) for i in range(self.world_size)
        ]

        # Generator process group uses master port 29600
        self.generator_workers = [
            RayVLLMGeneratorWorker.remote(
                rank=i,
                world_size=self.world_size,
                master_addr=master_addr,
                master_port=29600,
                config=config.generator,
                model_spec=config.model_spec,
                model_path=config.hf_assets_path,
                compile_config=config.compile,
                max_num_seqs=max(
                    config.num_prompts_per_step * config.generator.sampling.n,
                    config.num_validation_samples,
                ),
            ) for i in range(self.world_size)
        ]

        # Initial weight sync
        self.sync_weights()

    def sync_weights(self):
        """Synchronize sharded weights directly from Trainer Rank i to Generator Rank i."""
        logger.info("Performing sharded weight synchronization Rank-to-Rank...")
        refs = []
        for i in range(self.world_size):
            weights_ref = self.trainer_workers[i].get_sharded_weights.remote()
            refs.append(self.generator_workers[i].load_sharded_weights.remote(weights_ref))
        ray.get(refs)
        logger.info("Weight sync complete.")

    def _collect_rollouts(self, num_groups: int, step: int):
        envs = [self.config.env.build(step=step, group_idx=i) for i in range(num_groups)]
        prompts = [env.prompt for env in envs]
        
        # In external_launcher mode, all vLLM workers must execute generate collectively
        futures = [worker.generate.remote(prompts) for worker in self.generator_workers]
        results = ray.get(futures)
        
        # All workers return the same output, so we take Rank 0's completions
        completions = results[0]
        
        trajectories = []
        for c in completions:
            step_result = envs[c.prompt_idx].step(c.text)
            from torchtitan.experiments.rl.types import Trajectory
            trajectories.append(
                Trajectory(sample_idx=c.prompt_idx, transitions=[(c, step_result)])
            )
        return trajectories

    async def train(self):
        num_steps = self.config.num_steps
        num_groups = self.config.num_prompts_per_step
        
        logger.info(f"Starting {num_steps} steps of RL training on TPU using Ray worker groups")

        for step in range(num_steps):
            step_start = time.perf_counter()

            # --- 1. Collect data (Collective Generator Workers) --- #
            trajectories = self._collect_rollouts(num_groups, step=step)
            episodes = self._build_episodes(trajectories)

            if self.config.log_samples:
                _log_samples(episodes)

            # --- 2. Train step (Trainer Workers) --- #
            # Collate sharded episodes per rank and dispatch them to the corresponding worker
            sharded_episodes = self._shard_episodes(episodes)
            batches = [self._collate_episodes(rank_episodes) for rank_episodes in sharded_episodes]
            
            # Dispatch batch to each rank-specific trainer worker
            # The original trainer expects a list of batches of length world_size, sharded by rank.
            # So worker i receives the full list and indexes its own batch using `self.dp_rank`
            fwd_bwd_futures = [
                self.trainer_workers[i].forward_backward.remote(batches)
                for i in range(self.world_size)
            ]
            fwd_bwd_results = ray.get(fwd_bwd_futures)
            
            # Step the optimizers collectively
            optim_futures = [worker.optim_step.remote() for worker in self.trainer_workers]
            optim_results = ray.get(optim_futures)
            
            # Combine metrics from rank 0 worker
            metrics = {**fwd_bwd_results[0], **optim_results[0]}

            # --- 3. Sharded Weight sync --- #
            t0 = time.perf_counter()
            self.sync_weights()
            t_sync = time.perf_counter() - t0

            # --- Logging --- #
            steps = [t.transitions[0][1] for t in trajectories]
            components = _mean_rewards(steps)
            avg_tokens = sum(len(ep.token_ids) for ep in episodes) / len(episodes)
            
            loss_val = metrics.get('loss', 0.0)
            logger.info(
                f"Step {step:2d} | Loss: {loss_val:+.4f} | "
                f"Reward: {sum(components.values()):+.3f} ({_format_rewards(components)}) | "
                f"Avg tokens: {avg_tokens:>3.0f} | "
                f"Sync Time: {t_sync:.3f}s | "
                f"Step Time: {time.perf_counter() - step_start:.1f}s"
            )

            if not math.isfinite(loss_val):
                logger.error("Loss is NaN/Inf; training diverged")
                break


async def main():
    config = ConfigManager().parse_args()
    ray_rl_trainer = RayRLTrainer(config)
    await ray_rl_trainer.setup()
    await ray_rl_trainer.train()

if __name__ == "__main__":
    asyncio.run(main())
