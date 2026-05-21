# Copyright 2026 The TorchTitan Authors. All Rights Reserved.

r"""Trainer for GRPO on TPU, with optional separate sharded sampler model.

Essentially following torchtitan/experiments/tpu/train_minimal.py
but changing the SFT task to GRPO, using vllm for sampling

  ############ on v6e vm ################
  # Qwen3 0.6B with vLLM (fits comfortably):
  # Times: Sample=6.97s, Ref=0.07s, Reward=0.00s, Train=1.62s, Total=8.66s
  torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.rl.train_grpo \
    --module=torchtitan.experiments.tpu.rl \
    --config=grpo_qwen3_0_6b_glp \
    --training.steps=8 \
    --sampler.use_vllm \
    --training.local_batch_size=2

  # Qwen3 1.7B with vLLM (requires local_batch_size=1):
  # Times: Sample=7.14s, Ref=0.06s, Reward=0.00s, Train=2.47s, Total=9.67s
  torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.rl.train_grpo \
    --module=torchtitan.experiments.tpu.rl \
    --config=grpo_qwen3_1_7b_gf \
    --training.steps=8 \
    --sampler.use_vllm \
    --training.local_batch_size=1
"""

# pylint: disable=protected-access

import functools
import os
import time
import typing


from jax import profiler as jax_profiler
import torch
from torch.distributed import fsdp

import torch.nn.functional as F
import torch_tpu
from torchtitan.components import metrics
import torchtitan.components.tokenizer
import torchtitan.config
import torchtitan.config.manager
import torchtitan.distributed
from torchtitan.distributed import utils as dist_utils

from torchtitan.experiments.tpu import profiler_workaround
from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu import utils as tpu_utils

import torchtitan.experiments.tpu.llama3  # trigger llama3_tpu model registration
import torchtitan.experiments.tpu.qwen3  # trigger qwen3_tpu model registration
from torchtitan.experiments.tpu.rl import grpo_job_config
import torchtitan.experiments.tpu.rl.grpo_sampler as grpo_sampler
import torchtitan.experiments.tpu.rl.grpo_utils as grpo_utils
import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.tools import utils
import torchtitan.tools.logging


TORCH_DTYPE_MAP = torchtitan.config.TORCH_DTYPE_MAP
ParallelDims = torchtitan.distributed.ParallelDims

logger = torchtitan.tools.logging.logger

# ---------------------------------------------------------------------------
# Training Step
# ---------------------------------------------------------------------------


def grpo_step(
    model: torch.nn.Module,
    ref_model: torch.nn.Module | None,
    sampler_model: torch.nn.Module | None,
    vllm_sampler,
    prompt_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    job_config: grpo_job_config.GRPOJobConfig,
    device: torch.device,
    step: int,
    parallel_dims: ParallelDims,
    group_size: int = 4,
    vocab_size: int = 151936,
) -> tuple[
    torch.Tensor, torch.Tensor, float, float, float, float, float, float
]:
  """Single GRPO training step."""
  t_step_start = time.perf_counter()

  # Sync weights to sampler model if using separate one
  if sampler_model is not None:
    logger.info("Step %d: Syncing weights to sampler model...", step)
    grpo_utils.sync_model_weights(model, sampler_model, parallel_dims)

  # Repeat prompts for group size
  prompt_ids_repeated = prompt_ids.repeat_interleave(group_size, dim=0)

  # Sample completions using SAMPLER model or POLICY model
  sampling_model = sampler_model if sampler_model is not None else model
  sampling_model.eval()

  # Use summon_full_params for sampling if sampling_model is FSDP
  t_sample_start = time.perf_counter()

  with jax_profiler.TraceAnnotation("sampling"):
    if vllm_sampler is not None:
      logger.info("Step %d: Generating with vLLM engine...", step)
      from vllm import SamplingParams
      sampling_params = SamplingParams(
          temperature=job_config.sampler.temperature,
          top_k=job_config.sampler.top_k if job_config.sampler.top_k > 0 else -1,
          max_tokens=job_config.sampler.max_new_tokens,
      )
      
      completed_ids, token_log_probs = vllm_sampler.generate(prompt_ids_repeated, sampling_params)
      
    else:
      with torch.no_grad():
        with fsdp.FullyShardedDataParallel.summon_full_params(
            sampling_model, recurse=True, writeback=False
        ):
          if job_config.sampler.use_fake_sampler:
            completed_ids, token_log_probs = grpo_sampler.generate_fake(
                sampling_model,
                prompt_ids_repeated,
                max_seq_len=job_config.training.seq_len,
                max_new_tokens=job_config.sampler.max_new_tokens,
                temperature=job_config.sampler.temperature,
                top_k=job_config.sampler.top_k
                if job_config.sampler.top_k > 0
                else None,
                vocab_size=vocab_size,
            )
          else:
            completed_ids, token_log_probs = grpo_sampler.generate(
                sampling_model,
                prompt_ids_repeated,
                max_seq_len=job_config.training.seq_len,
                max_new_tokens=job_config.sampler.max_new_tokens,
                temperature=job_config.sampler.temperature,
                top_k=job_config.sampler.top_k
                if job_config.sampler.top_k > 0
                else None,
            )

  sampling_model.train()
  torch_tpu._internal.sync.synchronize(completed_ids, wait=True)
  t_sample_end = time.perf_counter()

  sampling_time = t_sample_end - t_sample_start

  # Compute ref_log_probs
  t_ref_start = time.perf_counter()
  if job_config.reference.use_reference_model and ref_model is not None:

    logger.info(
        "Step %d: Computing reference log probs with ref model...", step
    )
    ref_model.eval()
    with jax_profiler.TraceAnnotation("reference_forward"):
      with torch.no_grad():
        outputs = ref_model(completed_ids)
        if isinstance(outputs, tuple):
          ref_logits = outputs[0]
        else:
          ref_logits = outputs

        prompt_len = prompt_ids.shape[1]
        gen_ref_logits = ref_logits[:, prompt_len - 1 : -1, :]
        gen_targets = completed_ids[:, prompt_len:]
        ref_log_probs = F.log_softmax(gen_ref_logits, dim=-1)
        ref_token_log_probs = ref_log_probs.gather(
            2, gen_targets.unsqueeze(-1)
        ).squeeze(-1)

  else:
    logger.info("Step %d: Using generation log probs as reference...", step)
    ref_token_log_probs = token_log_probs

  torch_tpu._internal.sync.synchronize(ref_token_log_probs, wait=True)
  t_ref_end = time.perf_counter()

  ref_time = t_ref_end - t_ref_start

  # Dummy reward (sum of token IDs / vocab_size + noise) to ensure non-zero
  # gradients for testing.
  t_reward_start = time.perf_counter()
  # Generate purely random rewards to ensure non-zero advantages and gradients
  # without depending on model vocab size or token IDs.
  rewards = torch.randn(completed_ids.shape[0], device=device)
  t_reward = time.perf_counter() - t_reward_start

  rewards_mean = rewards.mean() / job_config.sampler.max_new_tokens

  torch_tpu._internal.sync.synchronize(rewards_mean, wait=True)
  try:
    avg_reward_val = rewards_mean.cpu().item()
  except Exception as e:  # pylint: disable=broad-exception-caught
    # Catching all exceptions to avoid crashing the training loop if lazy
    # execution fails to pull metrics to host.
    logger.info("Failed to get rewards_mean.cpu().item(): %s", e)
    avg_reward_val = 0.0
    logger.info("Step %d: Average Reward: %s", step, rewards_mean)

  # Compute advantages
  advantages, _ = grpo_utils.compute_grpo_advantages(
      rewards, group_size=group_size
  )

  logger.info("Step %d: Computing loss and backward pass...", step)

  train_time = 0.0
  grad_norm = torch.tensor(0.0, device=device)
  loss = torch.tensor(0.0, device=device)
  # Compute loss and do backward

  for epoch in range(job_config.grpo.ppo_epochs):
    t_epoch_start = time.perf_counter()

    logger.info(
        "Step %d: PPO Epoch %d/%d", step, epoch + 1, job_config.grpo.ppo_epochs
    )

    with jax_profiler.TraceAnnotation("optimizer.zero_grad"):
      optimizer.zero_grad()

    with jax_profiler.TraceAnnotation("grpo_loss"):
      loss = grpo_utils.compute_grpo_loss(
          model,
          prompt_ids_repeated,
          completed_ids,
          ref_log_probs=ref_token_log_probs,
          advantages=advantages,
          ppo_clip_eps=job_config.grpo.ppo_clip_eps,
          grpo_beta=job_config.grpo.grpo_beta,
      )

    with jax_profiler.TraceAnnotation("loss.backward"):
      loss.backward()

    if job_config.training.max_norm > 0.0:
      with jax_profiler.TraceAnnotation("grad_norm"):
        grad_norm = dist_utils.clip_grad_norm_(
            model.parameters(), job_config.training.max_norm
        )

    with jax_profiler.TraceAnnotation("optimizer.step"):
      optimizer.step()
    torch_tpu._internal.sync.synchronize(loss, wait=True)
    t_epoch_end = time.perf_counter()

    epoch_time = t_epoch_end - t_epoch_start
    train_time += epoch_time

  t_step_end = time.perf_counter()
  total_step_time = t_step_end - t_step_start

  torch_tpu._internal.sync.synchronize(grad_norm, wait=True)
  try:
    grad_norm_val = grad_norm.item()
  except Exception as e:  # pylint: disable=broad-exception-caught
    # Catching all exceptions to avoid crashing the training loop if lazy
    # execution fails to pull metrics to host.
    logger.info("Failed to get grad_norm.item(): %s", e)
    grad_norm_val = 0.0

  return (
      loss,
      grad_norm,
      avg_reward_val,
      grad_norm_val,
      sampling_time,
      ref_time,
      t_reward,
      train_time,
      total_step_time,
  )


# ---------------------------------------------------------------------------

# Main Trainer
# ---------------------------------------------------------------------------


def start_trainer(job_config: grpo_job_config.GRPOJobConfig) -> None:
  """Starts the training process."""
  rank = int(os.environ.get("RANK", 0))
  world_size = int(os.environ.get("WORLD_SIZE", 1))

  if rank == 0:
    torchtitan.tools.logging.init_logger()
    job_config.maybe_log()

  device = tpu_utils.get_device()

  if world_size > 1:
    dist_utils.init_distributed(
        job_config.comm,
        enable_cpu_backend=job_config.training.enable_cpu_offload,
        base_folder=job_config.dump_folder,
    )

    torch.distributed.barrier()

  dp_replicate = job_config.parallelism.data_parallel_replicate_degree
  dp_shard = job_config.parallelism.data_parallel_shard_degree

  if dp_replicate == -1:
    dp_shard = 1
    cp = job_config.parallelism.context_parallel_degree
    tp = job_config.parallelism.tensor_parallel_degree
    pp = job_config.parallelism.pipeline_parallel_degree
    dp_replicate = world_size // (dp_shard * cp * tp * pp)

  parallel_dims = ParallelDims(
      dp_shard=dp_shard,
      dp_replicate=dp_replicate,
      cp=job_config.parallelism.context_parallel_degree,
      tp=job_config.parallelism.tensor_parallel_degree,
      pp=job_config.parallelism.pipeline_parallel_degree,
      ep=1,
      world_size=world_size,
  )


  logger.info("parallel_dims: %s", parallel_dims)
  job_config.maybe_log()



  seed = job_config.debug.seed or 42
  if utils.get_device_type() == "tpu":
    torch.manual_seed(seed)
    logger.info("Set manual seed to %d", seed)
  else:
    if world_size > 1:
      dist_utils.set_determinism(
          parallel_dims,
          device,
          job_config.debug,
          distinct_seed_mesh_dims=["pp"],
      )

  log_freq = job_config.metrics.log_freq

  train_spec = train_spec_module.get_train_spec(job_config.model.name)

  model_args = train_spec.model_args[job_config.model.flavor]
  model_args.update_from_config(trainer_config=job_config)





  if job_config.training.dataset == "random":
    logger.info("Using random data loader.")
    dataloader = grpo_utils.build_random_dataloader(
        job_config, model_args.vocab_size, device
    )

  else:
    raise NotImplementedError("Only random dataset is supported for now in GRPO.")


  logger.info("Building model...")
  with (
      torch.device("meta"),
      utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
  ):
    model = typing.cast(torch.nn.Module, train_spec.model_cls(model_args))

  if job_config.reference.use_reference_model:
    logger.info("Building reference model...")
    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
    ):
      ref_model = typing.cast(torch.nn.Module, train_spec.model_cls(model_args))
  else:
    ref_model = None

  if job_config.sampler.use_separate_sampler_model:
    logger.info("Building separate sampler model...")
    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
    ):
      sampler_model = typing.cast(
          torch.nn.Module, train_spec.model_cls(model_args)
      )
  else:
    sampler_model = None

  # Parallelize first if world_size > 1
  if world_size > 1:
    def call_parallelize(model_to_par, p_dims):
        return train_spec.parallelize_fn(
            model=model_to_par,
            parallel_dims=p_dims,
            training=job_config.training,
            parallelism=job_config.parallelism,
            compile_config=job_config.compile,
            ac_config=job_config.activation_checkpoint,
            dump_folder=job_config.dump_folder,
        )

    call_parallelize(model, parallel_dims)
    logger.info("Parallelized policy model using FSDP.")

    if ref_model is not None:
      if job_config.reference.distributed_strategy == "fsdp":
        call_parallelize(ref_model, parallel_dims)
        logger.info("Parallelized reference model using FSDP.")
      else:
        logger.info("Keeping reference model unparallelized.")

    if sampler_model is not None:
      if job_config.sampler.distributed_strategy == "fsdp":
        call_parallelize(sampler_model, parallel_dims)
        logger.info("Parallelized sampler model using FSDP.")
      else:
        logger.info("Keeping sampler model unparallelized.")

  # Materialize models
  logger.info("Moving model to device %s", device)
  if not job_config.training.enable_cpu_offload:
    model = model.to_empty(device=device)
    if ref_model is not None:
      ref_model = ref_model.to_empty(device=device)
    if sampler_model is not None:
      sampler_model = sampler_model.to_empty(device=device)

  with torch.no_grad():
    model.init_weights()
    if ref_model is not None:
      ref_model.init_weights()
    if sampler_model is not None:
      sampler_model.init_weights()

  # Copy weights AFTER materialization and initialization!
  # Since both are FSDP, state dict structure matches!
  if ref_model is not None:
    try:
      with fsdp.FullyShardedDataParallel.summon_full_params(
          ref_model, recurse=True, writeback=True
      ):

        ref_model.load_state_dict(model.state_dict())
    except Exception as e:  # pylint: disable=broad-exception-caught
      # Catching all exceptions to fall back to direct load if summon_full_params fails.
      logger.info(
          "Failed to use summon_full_params for ref_model copy, trying direct"
          " load. Error: %s",
          e,
      )
      ref_model.load_state_dict(model.state_dict())
  if sampler_model is not None:

    grpo_utils.sync_model_weights(model, sampler_model, parallel_dims)

    for p in sampler_model.parameters():
      p.requires_grad = False

  trainable_params = [p for p in model.parameters() if p.requires_grad]

  optimizer = torch.optim.AdamW(
      trainable_params,
      lr=job_config.optimizer.lr,
      eps=job_config.optimizer.eps,
      foreach=True,
  )

  if torch.distributed.is_initialized():
    torch.distributed.barrier()

  steps = job_config.training.steps
  local_batch_size = job_config.training.local_batch_size
  seq_len = job_config.training.seq_len

  _, num_flops_per_token = model_args.get_nparams_and_flops(
      model, job_config.training.seq_len
  )

  metrics_processor = job_config.metrics.build(
      parallel_dims=parallel_dims,
      dump_folder=job_config.dump_folder,
      pp_schedule=job_config.parallelism.pipeline_parallel_schedule,
      config_dict=job_config.to_dict(),
      has_quantization=False,
  )

  metrics_processor.num_flops_per_token = num_flops_per_token

  device_name = tpu_utils.get_device_module().get_device_name()
  peak_flops = utils.get_peak_flops(device_name)

  total_tokens = 0
  total_time = 0.0
  accumulated_tokens = 0
  accumulated_time = 0.0
  accumulated_steps = 0

  warmup_steps = job_config.lr_scheduler.warmup_steps

  if job_config.sampler.use_vllm:
    from torchtitan.experiments.tpu.rl.grpo_vllm_sampler import VLLMSampler
    vllm_sampler = VLLMSampler(job_config)
  else:
    vllm_sampler = None

  maybe_enable_profiling = functools.partial(
      profiler_workaround.maybe_enable_profiling, job_config=job_config
  )

  ntokens_seen = 0
  with maybe_enable_profiling(
      job_config.profiler,
      global_step=0,
      base_folder=job_config.dump_folder,
  ) as profiler:

    data_iterator = iter(dataloader)
    for step in range(steps):
      step_start = time.perf_counter()
      
      if vllm_sampler is not None:
        vllm_sampler.sync_weights(model)

      step_tokens = local_batch_size * seq_len
      ntokens_seen += step_tokens
      metrics_processor.ntokens_since_last_log += step_tokens

      t0 = time.perf_counter()
      batch = next(data_iterator)
      metrics_processor.data_loading_times.append(time.perf_counter() - t0)

      max_new_tokens = job_config.sampler.max_new_tokens
      prompt_ids = batch[0]["input"][:, : seq_len - max_new_tokens].to(device)

      should_log = (step + 1) % log_freq == 0 or step == (steps - 1)

      with jax_profiler.TraceAnnotation("train", step_num=step):
        (
            loss,
            _,
            avg_reward,
            grad_norm_val,
            t_sample,
            t_ref,
            t_reward,
            t_train,
            t_total,
        ) = grpo_step(
            model=model,
            ref_model=ref_model,
            sampler_model=sampler_model,
            vllm_sampler=vllm_sampler,
            prompt_ids=prompt_ids,
            optimizer=optimizer,
            job_config=job_config,
            device=device,
            step=step + 1,
            parallel_dims=parallel_dims,
            group_size=job_config.grpo.group_size,
            vocab_size=model_args.vocab_size,
        )

      if should_log:
        with jax_profiler.TraceAnnotation("D2H", step_num=step):
          try:
            loss_cpu = loss.cpu().item()
          except Exception as e:
            logger.info("Failed to get loss.cpu().item(): %s", e)
            loss_cpu = 0.0
            logger.info("Step %d: Loss: %s", step + 1, loss)

      else:
        loss_cpu = float("nan")

      if profiler:
        profiler.step()

      step_end = time.perf_counter()
      step_time = step_end - step_start

      accumulated_tokens += step_tokens
      accumulated_time += step_time
      accumulated_steps += 1

      if step >= warmup_steps:
        total_tokens += step_tokens
        total_time += step_time

      if should_log:
        average_step_time = accumulated_time / accumulated_steps
        accumulated_tokens = 0
        accumulated_time = 0.0
        accumulated_steps = 0

        extra_metrics = {
            "lr": optimizer.param_groups[0]["lr"],
            "n_tokens_seen": ntokens_seen,
            "avg_step_time": average_step_time,
            "avg_reward": avg_reward,
            "sampling_time": t_sample,
            "ref_time": t_ref,
            "reward_time": t_reward,
            "train_time": t_train,
            "total_step_time": t_total,
        }
        if grad_norm_val == 0.0:
          logger.info("Step %d: Grad Norm: %s", step + 1, grad_norm_val)

        logger.info(
            "Step %d: Avg Reward: %.4f, Times: Sample=%.2fs, Ref=%.2fs,"
            " Reward=%.2fs, Train=%.2fs, Total=%.2fs",
            step + 1,
            avg_reward,
            t_sample,
            t_ref,
            t_reward,
            t_train,
            t_total,
        )

        metrics_processor.log(
            step + 1,
            loss_cpu,
            loss_cpu,
            grad_norm_val,
            extra_metrics=extra_metrics,
        )

  avg_tps = total_tokens / total_time if total_time > 0 else 0.0
  avg_tflops = num_flops_per_token * avg_tps / 1e12
  avg_mfu = 100 * num_flops_per_token * avg_tps / peak_flops
  if rank == 0:
    logger.info(
        "Average TPS (excl. %d warmup steps): %.0f  "
        "avg TFlops: %.2f  avg MFU: %.2f%%",
        warmup_steps,
        avg_tps,
        avg_tflops,
        avg_mfu,
    )

  if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
  logger.info("Process group destroyed")


if __name__ == "__main__":
  gmain.handle_main(start_trainer)
