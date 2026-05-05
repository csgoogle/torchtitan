r"""Conformer minimal training script using torchtitan infra.

Example:
  torchrun --nproc_per_node=4 \
  -m torchtitan.experiments.tpu.conformer.train_minimal \
  --job.config_file=torchtitan/experiments/tpu/conformer/train_configs/conformer.toml \
  --model.name=conformer_tpu \
  --model.flavor=test
"""

import os
import time
import typing
from typing import Tuple

from jax import profiler as jax_profiler
import torch
import torch.nn.functional as F
from torchtitan.components import metrics
import torchtitan.components.tokenizer
import torchtitan.config
import torchtitan.distributed
from torchtitan.distributed import utils as dist_utils
import torchtitan.experiments.tpu.conformer  # trigger conformer_tpu model registration
import torchtitan.experiments.tpu.gmain as gmain
import torchtitan.experiments.tpu.model_annotator as model_annotator
import torchtitan.experiments.tpu.tpu_job_config
import torchtitan.experiments.tpu.utils as tpu_utils
import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.tools import utils
import torchtitan.tools.logging

TORCH_DTYPE_MAP = torchtitan.config.TORCH_DTYPE_MAP
JobConfig = torchtitan.config.JobConfig
ParallelDims = torchtitan.distributed.ParallelDims
TPUJobConfig = torchtitan.experiments.tpu.tpu_job_config.TPUJobConfig
logger = torchtitan.tools.logging.logger


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------


def train_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    graph_split: bool = False,
    use_ctc: bool = False,
    output_lengths: torch.Tensor = None,
    target_lengths: torch.Tensor = None,
) -> torch.Tensor:
  """Single training step."""
  with jax_profiler.TraceAnnotation("optimizer.zero_grad"):
    optimizer.zero_grad()

  if graph_split:
    import torch_tpu._internal.sync  # pylint: disable=g-import-not-at-top

  logits = model(inputs)
  with jax_profiler.TraceAnnotation("loss"):
    if use_ctc:
      # CTC loss expects (T, N, C)
      log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(
          0, 1
      )
      criterion = torch.nn.CTCLoss()
      loss = criterion(log_probs, targets, output_lengths, target_lengths)
    else:
      loss = F.cross_entropy(
          logits.reshape(-1, logits.shape[-1]),
          targets.reshape(-1),
          reduction="sum",
      )

  if graph_split:
    torch_tpu._internal.sync.synchronize(loss, wait=False)
  loss.backward()

  with jax_profiler.TraceAnnotation("optimizer.step"):
    optimizer.step()

  return loss


def build_random_dataloader(job_config, hidden_dim, vocab_size, device):
  local_batch_size = job_config.training.local_batch_size
  seq_len = job_config.training.seq_len
  dtype = TORCH_DTYPE_MAP[job_config.training.dtype]
  x = torch.rand(
      local_batch_size, seq_len, hidden_dim, device=device, dtype=dtype
  )
  targets = torch.randint(
      0, vocab_size, (local_batch_size, seq_len), device=device
  )
  while True:
    yield {"input": x}, targets


def start_trainer(job_config: JobConfig) -> None:
  rank = int(os.environ.get("RANK", 0))
  world_size = int(os.environ.get("WORLD_SIZE", 1))

  if rank == 0:
    torchtitan.tools.logging.init_logger()
    job_config.maybe_log()

  device = tpu_utils.get_device()

  if world_size == 1 and device.type == "tpu":
    from torch_tpu._internal.device._device_module import TpuDeviceModule

    if not TpuDeviceModule.is_initialized():
      logger.info("Initializing PjRt runtime for single chip.")
      try:
        TpuDeviceModule._init_runtime_options()
      except RuntimeError as e:
        if "PrivateUse1HooksInterface only could be registered once" in str(e):
          logger.warning("PjRt hooks already registered, ignoring error.")
        else:
          raise

  if world_size > 1:
    dist_utils.init_distributed(
        job_config.comm,
        enable_cpu_backend=job_config.training.enable_cpu_offload,
        base_folder=job_config.job.dump_folder,
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
      ep=job_config.parallelism.expert_parallel_degree,
      etp=job_config.parallelism.expert_tensor_parallel_degree,
      world_size=world_size,
  )
  logger.info("parallel_dims: %s", parallel_dims)
  job_config.maybe_log()

  batch_degree, batch_rank = 1, 0
  if world_size > 1:
    if parallel_dims.dp_enabled:
      batch_mesh = parallel_dims.get_mesh("batch")
      batch_degree, batch_rank = (
          batch_mesh.size(),
          batch_mesh.get_local_rank(),
      )
    else:
      batch_degree, batch_rank = 1, 0

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

  use_graph_split = (
      isinstance(job_config, TPUJobConfig)
      and job_config.tpu_config.use_graph_split
  )
  log_freq = job_config.metrics.log_freq

  train_spec = train_spec_module.get_train_spec(job_config.model.name)

  model_args = train_spec.model_args[job_config.model.flavor]
  model_args.update_from_config(job_config)

  tokenizer = typing.cast(
      torchtitan.components.tokenizer.HuggingFaceTokenizer,
      train_spec.build_tokenizer_fn(job_config)
      if train_spec.build_tokenizer_fn is not None
      else None,
  )

  if job_config.training.dataset == "random":
    logger.info("Using random data loader instead of real dataset.")
    dataloader = build_random_dataloader(
        job_config, model_args.hidden_dim, model_args.vocab_size, device
    )
  else:
    dataloader = train_spec.build_dataloader_fn(
        dp_world_size=batch_degree,
        dp_rank=batch_rank,
        tokenizer=tokenizer,
        job_config=job_config,
    )

  logger.info(
      "Building %s %s with %s",
      job_config.model.name,
      job_config.model.flavor,
      model_args,
  )

  metrics_processor = metrics.build_metrics_processor(
      job_config, parallel_dims, model_args
  )

  with (
      torch.device("meta"),
      utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
  ):
    model = typing.cast(torch.nn.Module, train_spec.model_cls(model_args))

  model_param_count, _ = model_args.get_nparams_and_flops(
      model, job_config.training.seq_len
  )
  logger.info(
      "Model %s %s size: %d total parameters",
      job_config.model.name,
      job_config.model.flavor,
      model_param_count,
  )

  if job_config.training.enable_cpu_offload:
    logger.info("Materializing model on CPU for CPU offloading")
    model = model.to_empty(device="cpu")

  use_simple_fsdp = (
      isinstance(
          job_config, torchtitan.experiments.tpu.tpu_job_config.TPUJobConfig
      )
      and job_config.tpu_config.use_simple_fsdp
  )

  if use_simple_fsdp:
    # SimpleFSDP requires weights to be materialized and initialized BEFORE
    # wrapping/parallelization. This is because SimpleFSDP replaces module
    # parameters with properties, which breaks in-place initialization (like
    # model.init_weights()) if called after wrapping.
    logger.info(
        "Moving model to device %s before parallelization (SimpleFSDP flow).",
        device,
    )
    if not job_config.training.enable_cpu_offload:
      model = model.to_empty(device=device)
    with torch.no_grad():
      model.init_weights()

  model_compile_enabled = (
      job_config.compile.enable and "model" in job_config.compile.components
  )
  if world_size > 1 or model_compile_enabled:
    if train_spec.parallelize_fn is not None:
      compiled_model = train_spec.parallelize_fn(
          model, parallel_dims, job_config
      )
      if compiled_model is not None:
        model = compiled_model
    else:
      logger.warning("No parallelize_fn provided in TrainSpec.")

  if not use_simple_fsdp:
    # For FSDP2 and other strategies, we prefer to parallelize (and optionally
    # compile) FIRST while the model is still on the meta device, and then
    # materialize and initialize weights on the correct device afterwards.
    logger.info(
        "Moving model to device %s after parallelization (Standard flow).",
        device,
    )
    if not job_config.training.enable_cpu_offload:
      model = model.to_empty(device=device)
    with torch.no_grad():
      model.init_weights()

  loss_fn = None  # Handled in train_step

  if not job_config.compile.enable and job_config.profiling.enable_profiling:
    model_annotator.wrap_model(model)

  trainable_params = [p for p in model.parameters() if p.requires_grad]
  logger.info(
      "Trainable parameters: %d / %d",
      sum(p.numel() for p in trainable_params),
      model_param_count,
  )

  if job_config.optimizer.implementation == "fused":
    logger.warning(
        "Fused optimizer is not supported on TPU, changing to foreach."
    )
    job_config.optimizer.implementation = "foreach"

  foreach = job_config.optimizer.implementation == "foreach"
  fused = job_config.optimizer.implementation == "fused"

  optimizer = torch.optim.AdamW(
      trainable_params,
      lr=job_config.optimizer.lr,
      eps=job_config.optimizer.eps,
      foreach=foreach,
      fused=fused,
  )

  if job_config.compile.enable and "optimizer" in job_config.compile.components:
    logger.info("Applying torch.compile to optimizer.step")
    optimizer.step = torch.compile(
        optimizer.step,
        backend=job_config.compile.backend,
        fullgraph=True,
        dynamic=False,
    )

  if torch.distributed.is_initialized():
    torch.distributed.barrier()

  steps = job_config.training.steps
  local_batch_size = job_config.training.local_batch_size
  seq_len = job_config.training.seq_len

  (
      model_param_count,
      num_flops_per_token,
  ) = model_args.get_nparams_and_flops(model, job_config.training.seq_len)
  metrics_processor.num_flops_per_token = num_flops_per_token

  device_name = tpu_utils.get_device_module().get_device_name()
  peak_flops = utils.get_peak_flops(device_name)
  logger.info(
      "Peak FLOPS for %s used for computing MFU: %d",
      device_name,
      peak_flops,
  )

  total_tokens = 0
  total_time = 0.0
  accumulated_tokens = 0
  accumulated_time = 0.0
  accumulated_steps = 0

  warmup_steps = job_config.lr_scheduler.warmup_steps

  import functools
  from torchtitan.experiments.tpu import profiler_workaround

  maybe_enable_profiling = functools.partial(
      profiler_workaround.maybe_enable_profiling, job_config=job_config
  )

  ntokens_seen = 0
  with maybe_enable_profiling(
      job_config.profiling,
      global_step=0,
      base_folder=job_config.job.dump_folder,
  ) as profiler:
    data_iterator = iter(dataloader)
    for step in range(steps):
      step_start = time.perf_counter()
      step_tokens = local_batch_size * seq_len
      ntokens_seen += step_tokens
      metrics_processor.ntokens_since_last_log += step_tokens

      t0 = time.perf_counter()
      batch = next(data_iterator)
      metrics_processor.data_loading_times.append(time.perf_counter() - t0)

      if parallel_dims.fsdp_enabled or (
          isinstance(job_config, TPUJobConfig)
          and job_config.tpu_config.use_simple_fsdp
      ):
        compute_dtype = TORCH_DTYPE_MAP[
            job_config.training.mixed_precision_param
        ]
        inputs = batch[0]["input"].to(device, dtype=compute_dtype)
      else:
        inputs = batch[0]["input"].to(
            device, dtype=next(model.parameters()).dtype
        )
      targets = batch[1].to(device)

      with jax_profiler.TraceAnnotation("train", step_num=step):
        use_ctc = (
            isinstance(job_config, TPUJobConfig)
            and job_config.conformer.use_ctc_loss
        )
        loss = train_step(
            model,
            inputs,
            targets,
            optimizer,
            loss_fn,
            graph_split=use_graph_split,
            use_ctc=use_ctc,
            output_lengths=torch.full(
                (inputs.size(0),),
                inputs.size(1),
                dtype=torch.long,
                device=device,
            ),
            target_lengths=torch.full(
                (targets.size(0),),
                targets.size(1),
                dtype=torch.long,
                device=device,
            ),
        )

      if (step + 1) % log_freq == 0 or step == (steps - 1):
        with jax_profiler.TraceAnnotation("D2H", step_num=step):
          loss_cpu = loss.cpu().item()
        should_log = True
      else:
        if use_graph_split:
          import torch_tpu._internal.sync

          torch_tpu._internal.sync.synchronize(loss, wait=False)
        loss_cpu = float("nan")
        should_log = False

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
        avg_step_time = accumulated_time / accumulated_steps
        accumulated_tokens = 0
        accumulated_time = 0.0
        accumulated_steps = 0

        extra_metrics = {
            "lr": optimizer.param_groups[0]["lr"],
            "n_tokens_seen": ntokens_seen,
            "avg_step_time": avg_step_time,
        }
        metrics_processor.log(
            step + 1,
            loss_cpu / step_tokens,
            loss_cpu / step_tokens,
            float("nan"),
            extra_metrics=extra_metrics,
        )

  avg_tps = total_tokens / total_time if total_time > 0 else 0.0
  avg_tflops = num_flops_per_token * avg_tps / 1e12
  avg_mfu = 100 * num_flops_per_token * avg_tps / peak_flops
  if rank == 0:
    logger.info(
        "Average TPS (i.e., Frames Per Second in this case, excl. %d warmup"
        " steps): %.0f  avg TFlops: %.2f  avg MFU: %.2f%%",
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
