import os
import sys
import time
import functools
import gc
import ray
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.tpu.rl import grpo_job_config

@ray.remote
class FusedWorker:
    def __init__(self, sys_argv):
        self.sys_argv = sys_argv
        self.job_config: 'grpo_job_config.GRPOJobConfig' = None
        self.model = None
        self.ref_model = None
        self.sampler_model = None
        self.vllm_sampler = None
        self.optimizer = None
        self.metrics_processor = None
        self.parallel_dims = None
        self.device = None
        self.model_args = None
        self.profiler = None
        self.profiler_ctx = None
        
        self.ntokens_seen = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.accumulated_tokens = 0
        self.accumulated_time = 0.0
        self.accumulated_steps = 0
        self.warmup_steps = 0
        self.steps = 0
        self.local_batch_size = 0
        self.seq_len = 0
        self.log_freq = 0
        self.num_flops_per_token = 0
        self.peak_flops = 0

    def init_models(self):
        # We put all heavy imports here so they are evaluated AFTER runtime_env is set
        import typing
        import torch
        import torch_tpu
        from torch.distributed import fsdp
        import torchtitan.config
        import torchtitan.distributed
        from torchtitan.distributed import utils as dist_utils
        from torchtitan.experiments.tpu import profiler_workaround
        from torchtitan.experiments.tpu import utils as tpu_utils
        import torchtitan.experiments.tpu.llama3 
        import torchtitan.experiments.tpu.qwen3 
        import torchtitan.experiments.tpu.rl.grpo_sampler as grpo_sampler
        import torchtitan.experiments.tpu.rl.grpo_utils as grpo_utils
        import torchtitan.protocols.train_spec as train_spec_module
        from torchtitan.tools import utils
        import torchtitan.tools.logging

        self.torch = torch
        self.torch_tpu = torch_tpu
        self.fsdp = fsdp
        self.dist_utils = dist_utils
        self.tpu_utils = tpu_utils
        self.grpo_sampler = grpo_sampler
        self.grpo_utils = grpo_utils
        self.train_spec_module = train_spec_module
        self.utils = utils
        from jax import profiler as jax_profiler
        self.jax_profiler = jax_profiler
        import torch.nn.functional as F
        self.F = F
        self.profiler_workaround = profiler_workaround

        sys.argv = self.sys_argv
        config_manager = ConfigManager()
        self.job_config = config_manager.parse_args(args=self.sys_argv[1:])

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if rank == 0:
            torchtitan.tools.logging.init_logger()
            self.job_config.maybe_log()

        self.device = tpu_utils.get_device()

        if world_size > 1:
            dist_utils.init_distributed(
                self.job_config.comm,
                enable_cpu_backend=self.job_config.training.enable_cpu_offload,
                base_folder=self.job_config.dump_folder,
            )
            torch.distributed.barrier()

        dp_replicate = self.job_config.parallelism.data_parallel_replicate_degree
        dp_shard = self.job_config.parallelism.data_parallel_shard_degree

        if dp_replicate == -1:
            dp_shard = 1
            cp = self.job_config.parallelism.context_parallel_degree
            tp = self.job_config.parallelism.tensor_parallel_degree
            pp = self.job_config.parallelism.pipeline_parallel_degree
            dp_replicate = world_size // (dp_shard * cp * tp * pp)

        self.parallel_dims = torchtitan.distributed.ParallelDims(
            dp_shard=dp_shard,
            dp_replicate=dp_replicate,
            cp=self.job_config.parallelism.context_parallel_degree,
            tp=self.job_config.parallelism.tensor_parallel_degree,
            pp=self.job_config.parallelism.pipeline_parallel_degree,
            ep=1,
            world_size=world_size,
        )

        seed = self.job_config.debug.seed or 42
        if utils.get_device_type() == "tpu":
            torch.manual_seed(seed)
        else:
            if world_size > 1:
                dist_utils.set_determinism(
                    self.parallel_dims,
                    self.device,
                    self.job_config.debug,
                    distinct_seed_mesh_dims=["pp"],
                )

        self.log_freq = self.job_config.metrics.log_freq

        train_spec = train_spec_module.get_train_spec(self.job_config.model.name)
        self.model_args = train_spec.model_args[self.job_config.model.flavor]
        self.model_args.update_from_config(trainer_config=self.job_config)

        with (
            torch.device("meta"),
            utils.set_default_dtype(torchtitan.config.TORCH_DTYPE_MAP[self.job_config.training.dtype]),
        ):
            self.model = typing.cast(torch.nn.Module, train_spec.model_cls(self.model_args))

        if self.job_config.reference.use_reference_model:
            with (
                torch.device("meta"),
                utils.set_default_dtype(torchtitan.config.TORCH_DTYPE_MAP[self.job_config.training.dtype]),
            ):
                self.ref_model = typing.cast(torch.nn.Module, train_spec.model_cls(self.model_args))
        else:
            self.ref_model = None

        if self.job_config.sampler.use_separate_sampler_model:
            with (
                torch.device("meta"),
                utils.set_default_dtype(torchtitan.config.TORCH_DTYPE_MAP[self.job_config.training.dtype]),
            ):
                self.sampler_model = typing.cast(torch.nn.Module, train_spec.model_cls(self.model_args))
        else:
            self.sampler_model = None

        if world_size > 1:
            def call_parallelize(model_to_par, p_dims):
                return train_spec.parallelize_fn(
                    model=model_to_par,
                    parallel_dims=p_dims,
                    training=self.job_config.training,
                    parallelism=self.job_config.parallelism,
                    compile_config=self.job_config.compile,
                    ac_config=self.job_config.activation_checkpoint,
                    dump_folder=self.job_config.dump_folder,
                )
            call_parallelize(self.model, self.parallel_dims)
            if self.ref_model is not None and self.job_config.reference.distributed_strategy == "fsdp":
                call_parallelize(self.ref_model, self.parallel_dims)
            if self.sampler_model is not None and self.job_config.sampler.distributed_strategy == "fsdp":
                call_parallelize(self.sampler_model, self.parallel_dims)

        if not self.job_config.training.enable_cpu_offload:
            self.model = self.model.to_empty(device=self.device)
            if self.ref_model is not None:
                self.ref_model = self.ref_model.to_empty(device=self.device)
            if self.sampler_model is not None:
                self.sampler_model = self.sampler_model.to_empty(device=self.device)

        with torch.no_grad():
            self.model.init_weights()
            if self.ref_model is not None:
                self.ref_model.init_weights()
            if self.sampler_model is not None:
                self.sampler_model.init_weights()

        optimizers_container = self.job_config.optimizer.build(model_parts=[self.model])
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.job_config.optimizer.lr,
            eps=self.job_config.optimizer.eps,
            foreach=True,   # Fused optimizer is not supported on TPU, set foreach=True
        )
        optimizers_container.optimizers[0] = self.optimizer

        if hasattr(self.job_config, "checkpoint") and self.job_config.checkpoint.enable:
            self.checkpointer = self.job_config.checkpoint.build(
                dataloader=None,
                model_parts=[self.model],
                optimizers=optimizers_container,
                lr_schedulers=None,
                states={},
                base_folder=self.job_config.dump_folder,
                sd_adapter=train_spec.state_dict_adapter(
                    self.model_args, self.job_config.hf_assets_path
                ) if train_spec.state_dict_adapter else None,
            )
            self.checkpointer.load(step=self.job_config.checkpoint.load_step)

        if self.ref_model is not None:
            try:
                with fsdp.FullyShardedDataParallel.summon_full_params(self.ref_model, recurse=True, writeback=True):
                    self.ref_model.load_state_dict(self.model.state_dict())
            except Exception as e:
                self.ref_model.load_state_dict(self.model.state_dict())

        if self.sampler_model is not None:
            self.sampler_model.load_state_dict(self.model.state_dict())
            for p in self.sampler_model.parameters():
                p.requires_grad = False

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.steps = self.job_config.training.steps
        self.local_batch_size = self.job_config.training.local_batch_size
        self.seq_len = self.job_config.training.seq_len

        _, self.num_flops_per_token = self.model_args.get_nparams_and_flops(self.model, self.seq_len)

        self.metrics_processor = self.job_config.metrics.build(
            parallel_dims=self.parallel_dims,
            dump_folder=self.job_config.dump_folder,
            pp_schedule=self.job_config.parallelism.pipeline_parallel_schedule,
            config_dict=self.job_config.to_dict(),
            has_quantization=False,
        )
        self.metrics_processor.num_flops_per_token = self.num_flops_per_token

        device_name = tpu_utils.get_device_module().get_device_name()
        self.peak_flops = utils.get_peak_flops(device_name)
        self.warmup_steps = self.job_config.lr_scheduler.warmup_steps

        if self.job_config.sampler.use_vllm:
            from torchtitan.experiments.tpu.rl.grpo_vllm_sampler import VLLMSampler
            self.vllm_sampler = VLLMSampler(self.job_config)
            self.vllm_sampler.sync_weights(self.model)
        else:
            self.vllm_sampler = None
        
        maybe_enable_profiling = functools.partial(
            profiler_workaround.maybe_enable_profiling, job_config=self.job_config
        )
        self.profiler_ctx = maybe_enable_profiling(
            self.job_config.profiler,
            global_step=0,
            base_folder=self.job_config.dump_folder,
        )
        self.profiler = self.profiler_ctx.__enter__()

        return True

    def heartbeat(self):
        """
        Simple RPC ping to verify the Ray Actor is alive and responsive.
        
        TODO: A more realistic heartbeat in a production system might check:
        - TPU memory utilization thresholds (to proactively catch leaks).
        - XLA compilation cache states or TPU metrics.
        - Hang detection (e.g., time since last forward pass or step).
        - Device Inter-Chip Interconnect (ICI) health.
        """
        return True

    def load_next_batch(self, prompt_ids_list):
        """Loads the next batch of prompts from the Driver."""
        self.current_prompt_ids = self.torch.tensor(
            prompt_ids_list, dtype=self.torch.long, device=self.device
        )
        return True

    def generate_rollouts(self):
        gc.collect()
        self.torch_tpu._internal.sync.synchronize(wait=True)

        t_sample_start = time.perf_counter()
        
        group_size = self.job_config.grpo.group_size
        prompt_ids_repeated = self.current_prompt_ids.repeat_interleave(group_size, dim=0)
        self.current_prompt_ids_repeated = prompt_ids_repeated

        sampling_model = self.sampler_model if self.sampler_model is not None else self.model
        sampling_model.eval()

        with self.jax_profiler.TraceAnnotation("sampling"):
            if self.vllm_sampler is not None:
                from vllm import SamplingParams
                sampling_params = SamplingParams(
                    temperature=self.job_config.sampler.temperature,
                    max_tokens=self.job_config.sampler.max_new_tokens,
                    ignore_eos=True,   # HACK: make the completion from vllm same length
                )
                completed_ids, token_log_probs = self.vllm_sampler.generate(prompt_ids_repeated, sampling_params)
            else:
                with self.torch.no_grad():
                    # For non-vLLM sampling, we run generation directly on the FSDP model.
                    # PyTorch FSDP automatically performs a layer-by-layer all-gather during each 
                    # forward pass, discarding the full weights immediately after the layer computes.
                    # This keeps memory usage flat and avoids OOMs.
                    # Note: Generation speed could be improved by wrapping this entirely in 
                    # `fsdp.FullyShardedDataParallel.summon_full_params(recurse=True)` to un-shard 
                    # the entire model at once, but at the cost of massive memory spikes that usually 
                    # lead to Out-Of-Memory (OOM) crashes on TPUs for large models.
                    if self.job_config.sampler.use_fake_sampler:
                        completed_ids, token_log_probs = self.grpo_sampler.generate_fake(
                            sampling_model,
                            prompt_ids_repeated,
                            max_seq_len=self.job_config.training.seq_len,
                            max_new_tokens=self.job_config.sampler.max_new_tokens,
                            temperature=self.job_config.sampler.temperature,
                            top_k=self.job_config.sampler.top_k if self.job_config.sampler.top_k > 0 else None,
                        )
                    else:
                        completed_ids, token_log_probs = self.grpo_sampler.generate(
                            sampling_model,
                            prompt_ids_repeated,
                            max_seq_len=self.job_config.training.seq_len,
                            max_new_tokens=self.job_config.sampler.max_new_tokens,
                            temperature=self.job_config.sampler.temperature,
                            top_k=self.job_config.sampler.top_k if self.job_config.sampler.top_k > 0 else None,
                        )

        sampling_model.train()
        self.torch_tpu._internal.sync.synchronize(completed_ids, wait=True)
        self.current_t_sample = time.perf_counter() - t_sample_start

        self.current_completed_ids = completed_ids
        self.current_token_log_probs = token_log_probs

        return {
            # We MUST move the tensor to CPU here because it needs to be serialized and sent 
            # over the network to the Ray Driver (Orchestrator) for centralized GRPO math.
            # While worker-to-worker synchronization (like FSDP) leverages the ultra-fast 
            # TPU ICI (Inter-Chip Interconnect) to bypass the host, worker-to-driver 
            # communication always goes through host CPU memory.
            "completed_ids": completed_ids.cpu(),  
        }

    def compute_ref_log_probs(self):
        t_ref_start = time.perf_counter()
        
        prompt_len = self.current_prompt_ids_repeated.shape[1]

        if self.ref_model is not None:
            self.ref_model.eval()
            with self.jax_profiler.TraceAnnotation("reference_forward"):
                with self.torch.no_grad():
                    outputs = self.ref_model(self.current_completed_ids)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    gen_ref_logits = logits[:, prompt_len - 1 : -1, :]
                    gen_targets = self.current_completed_ids[:, prompt_len:]
                    ref_log_probs = self.F.log_softmax(gen_ref_logits, dim=-1)
                    ref_token_log_probs = ref_log_probs.gather(2, gen_targets.unsqueeze(-1)).squeeze(-1)
        else:
            ref_token_log_probs = self.current_token_log_probs

        self.torch_tpu._internal.sync.synchronize(ref_token_log_probs, wait=True)
        self.current_t_ref = time.perf_counter() - t_ref_start

        self.current_ref_token_log_probs = ref_token_log_probs

        return True

    def train_ppo_step(self, advantages, step, avg_correctness=0.0, avg_format=0.0):
        """Executes the PPO optimization loop for the current batch of rollouts."""
        gc.collect()
        self.torch_tpu._internal.sync.synchronize(wait=True)
            
        t_train_start = time.perf_counter()
        
        advantages = advantages.to(self.device)

        train_time = 0.0
        grad_norm = self.torch.tensor(0.0, device=self.device)
        loss = self.torch.tensor(0.0, device=self.device)

        for epoch in range(self.job_config.grpo.ppo_epochs):
            t_epoch_start = time.perf_counter()

            with self.jax_profiler.TraceAnnotation("optimizer.zero_grad"):
                self.optimizer.zero_grad()

            with self.jax_profiler.TraceAnnotation("grpo_loss"):
                loss = self.grpo_utils.compute_grpo_loss(
                    self.model,
                    self.current_prompt_ids_repeated,
                    self.current_completed_ids,
                    ref_log_probs=self.current_ref_token_log_probs,
                    advantages=advantages,
                    old_log_probs=self.current_token_log_probs,
                    ppo_clip_eps=self.job_config.grpo.ppo_clip_eps,
                    grpo_beta=self.job_config.grpo.grpo_beta,
                )

            with self.jax_profiler.TraceAnnotation("loss.backward"):
                loss.backward()

            if self.job_config.training.max_norm > 0.0:
                with self.jax_profiler.TraceAnnotation("grad_norm"):
                    grad_norm = self.dist_utils.clip_grad_norm_(
                        self.model.parameters(), self.job_config.training.max_norm
                    )

            with self.jax_profiler.TraceAnnotation("optimizer.step"):
                self.optimizer.step()
                
            self.torch_tpu._internal.sync.synchronize(loss, wait=True)
            train_time += time.perf_counter() - t_epoch_start

        self.torch_tpu._internal.sync.synchronize(grad_norm, wait=True)
        try:
            grad_norm_val = grad_norm.item()
            loss_cpu = loss.cpu().item()
        except:
            grad_norm_val = 0.0
            loss_cpu = 0.0

        if self.profiler:
            self.profiler.step()

        t_total = time.perf_counter() - t_train_start

        step_tokens = self.local_batch_size * self.seq_len
        self.ntokens_seen += step_tokens
        self.metrics_processor.ntokens_since_last_log += step_tokens
        self.accumulated_tokens += step_tokens
        self.accumulated_time += t_total
        self.accumulated_steps += 1

        if step >= self.warmup_steps:
            self.total_tokens += step_tokens
            self.total_time += t_total

        # TODO: move the metrics handling out of train_ppo_step
        should_log = (step + 1) % self.log_freq == 0 or step == (self.steps - 1)
        if should_log:
            average_step_time = self.accumulated_time / self.accumulated_steps
            self.accumulated_tokens = 0
            self.accumulated_time = 0.0
            self.accumulated_steps = 0

            extra_metrics = {
                "lr": self.optimizer.param_groups[0]["lr"],
                "n_tokens_seen": self.ntokens_seen,
                "avg_step_time": average_step_time,
                "sampling_time": self.current_t_sample,
                "ref_time": self.current_t_ref,
                "train_time": train_time,
                "total_step_time": t_total,
            }
            if grad_norm_val == 0.0:
                import torchtitan.tools.logging
                torchtitan.tools.logging.logger.info("Step %d: Grad Norm: %s", step + 1, grad_norm_val)

            import torchtitan.tools.logging
            torchtitan.tools.logging.logger.info(
                "Step %d: Times: Sample=%.2fs, Ref=%.2fs, Train=%.2fs, Total=%.2fs | Reward: Format=%.4f, Correctness=%.4f",
                step + 1, self.current_t_sample, self.current_t_ref, train_time, t_total, avg_format, avg_correctness
            )

            # Dataloader runs on the driver now, so we add a dummy value to avoid ZeroDivisionError
            self.metrics_processor.data_loading_times.append(0.0)
            self.metrics_processor.log(
                step + 1,
                loss_cpu,
                loss_cpu,
                grad_norm_val,
                extra_metrics=extra_metrics,
            )

        rank = int(os.environ.get("RANK", 0))
        if step == self.steps - 1 and rank == 0:
            avg_tps = self.total_tokens / self.total_time if self.total_time > 0 else 0.0
            avg_tflops = self.num_flops_per_token * avg_tps / 1e12
            avg_mfu = 100 * self.num_flops_per_token * avg_tps / self.peak_flops
            import torchtitan.tools.logging
            torchtitan.tools.logging.logger.info(
                "Average TPS (excl. %d warmup steps): %.0f  "
                "avg TFlops: %.2f  avg MFU: %.2f%%",
                self.warmup_steps,
                avg_tps,
                avg_tflops,
                avg_mfu,
            )

        return True

    def sync_weights(self):
        if self.vllm_sampler is not None:
            self.vllm_sampler.sync_weights(self.model)
        elif self.sampler_model is not None:
            self.sampler_model.load_state_dict(self.model.state_dict())
        return True

    def finish(self):
        if self.profiler_ctx:
            self.profiler_ctx.__exit__(None, None, None)
        if self.torch.distributed.is_initialized():
            self.torch.distributed.destroy_process_group()
        import torchtitan.tools.logging
        torchtitan.tools.logging.logger.info("Process group destroyed")
        return True

