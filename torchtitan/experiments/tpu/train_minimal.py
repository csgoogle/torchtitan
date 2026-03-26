"""Minimal trainer for TorchTitan models."""

import argparse
from collections.abc import Sequence
import json
import os
import time
import traceback
import typing
from typing import Any, Callable, List, Optional, Tuple

from absl import flags
from absl.flags import argparse_flags
import torch
from torch import nn
import torchtitan.components.tokenizer
import torchtitan.config
import torchtitan.distributed
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu import utils as tpu_utils
import torchtitan.experiments.tpu.deepseek_v3   # trigger model registration
import torchtitan.experiments.tpu.flux  # trigger model registration
import torchtitan.experiments.tpu.llama3  # trigger model registration
import torchtitan.experiments.tpu.qwen3   # trigger model registration
import torchtitan.experiments.tpu.tpu_job_config
import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.tools import utils
import torchtitan.tools.logging
import torchtitan.tools.profiling

TORCH_DTYPE_MAP = torchtitan.config.TORCH_DTYPE_MAP
JobConfig = torchtitan.config.JobConfig
ParallelDims = torchtitan.distributed.ParallelDims
TPUJobConfig = torchtitan.experiments.tpu.tpu_job_config.TPUJobConfig
logger = torchtitan.tools.logging.logger

TrainStepCallback = Callable[[int, nn.Module, torch.Tensor], None]


class TrainerMinimal:
  """Minimal trainer for TorchTitan models.

  This class sets up the model, tokenizer, dataloader, optimizer, and loss
  function for a basic training loop.
  """

  def __init__(
      self,
      device: torch.device,
      rank: int,
      world_size: int,
      job_config: JobConfig,
      step_callback: Optional[TrainStepCallback] = None,
  ):
    """Initializes the TrainerMinimal.

    Args:
      device: The device to run the training on.
      rank: The rank of the current process.
      world_size: The size of the world.
      job_config: The JobConfig object containing training and model
        configuration.
      step_callback: An optional function to be called after each optimizer
        step. It must accept three arguments: (step, model, loss).
    """
    self.device = device
    self.rank = rank
    self.world_size = world_size
    self.job_config = job_config

    if world_size > 1:
      dist_utils.init_distributed(
          job_config.comm,
          enable_cpu_backend=job_config.training.enable_cpu_offload,
          base_folder=job_config.job.dump_folder,
      )

    parallelism_config = self.job_config.parallelism
    self.parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        ep=parallelism_config.expert_parallel_degree,
        etp=parallelism_config.expert_tensor_parallel_degree,
        world_size=self.world_size,
    )

    logger.info(f"parallel_dims: {self.parallel_dims}")

    # Workaround if we run without distrubuted training.
    batch_degree, batch_rank = 1, 0
    if world_size > 1:
      if self.parallel_dims.dp_enabled:
        batch_mesh = self.parallel_dims.get_mesh("batch")
        batch_degree, batch_rank = (
            batch_mesh.size(), batch_mesh.get_local_rank())
      else:
        batch_degree, batch_rank = 1, 0

      logger.info(f"world_mesh: {self.parallel_dims.world_mesh}")

      # Set random seed, and maybe enable deterministic mode
      # (mainly for debugging, expect perf loss).
      dist_utils.set_determinism(
          self.parallel_dims,
          self.device,
          job_config.debug,
          distinct_seed_mesh_dims=["pp"],
      )

    self.train_spec = train_spec_module.get_train_spec(job_config.model.name)

    self.tokenizer = typing.cast(
        torchtitan.components.tokenizer.HuggingFaceTokenizer,
        self.train_spec.build_tokenizer_fn(job_config)
        if self.train_spec.build_tokenizer_fn is not None
        else None
    )

    self.dataloader = self.train_spec.build_dataloader_fn(
        dp_world_size=batch_degree,
        dp_rank=batch_rank,
        tokenizer=self.tokenizer,
        job_config=job_config,
    )

    model_args = self.train_spec.model_args[job_config.model.flavor]
    # set the model args from training job configs
    model_args.update_from_config(job_config)
    self.model_args: Any = model_args

    if (
        self.model_args.vocab_size is not None
        and self.tokenizer.vocab_size > self.model_args.vocab_size
    ):
      logger.warning(
          "Vocab size for tokenizer is %d while model args vocab size is %d."
          " Expanding model args vocab size to match tokenizer vocab size.",
          self.tokenizer.vocab_size,
          self.model_args.vocab_size,
      )
      self.model_args.vocab_size = self.tokenizer.vocab_size

    logger.info(
        f"Building {self.job_config.model.name} {self.job_config.model.flavor}"
        f"  with {self.model_args}"
    )

    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
    ):
      model: nn.Module = typing.cast(
          nn.Module, self.train_spec.model_cls(self.model_args)
      )

    # calculate model size and flops per token
    (
        model_param_count,
        _,
    ) = model_args.get_nparams_and_flops(model, job_config.training.seq_len)

    logger.info(
        f"Model {job_config.model.name} {job_config.model.flavor} "
        f"size: {model_param_count:,} total parameters"
    )

    self.loss_fn = self.train_spec.build_loss_fn(job_config)

    if self.world_size > 1:
      self.train_spec.parallelize_fn(
          model,
          self.parallel_dims,
          job_config)

    logger.info(
        f"Moving model {self.job_config.model.name}"
        f" {self.job_config.model.flavor} to device {self.device}"
    )
    self.model = model.to_empty(device=self.device)
    with torch.no_grad():
      self.model.init_weights()

    assert (
        self.device.type != "tpu" or
        job_config.optimizer.implementation != "fused"
    ), ("TODO b/45811390 - Fused optimizer not supported on TPU."
        "You can disable this by setting --optimizer.implementation=foreach")
    self.optimizers = self.train_spec.build_optimizers_fn(
        [self.model], job_config.optimizer, self.parallel_dims, None
    )

    loss_parallel_enabled = (
        self.parallel_dims.tp_enabled and
        not parallelism_config.disable_loss_parallel
    )
    self.train_context = dist_utils.get_train_context(loss_parallel_enabled)
    self.step_callback = step_callback

  def train(self):
    """Runs a simple training loop on the specified device."""

    data_iterator = iter(self.dataloader)

    try:

      def get_batch():
        batch = next(data_iterator)
        tokens = batch[0]["input"].to(self.device)
        labels = batch[1].to(self.device)
        return tokens, labels

      def _get_dummy_batch():
        tokens = torch.randint(
            0,
            self.model_args.vocab_size,
            (
                self.job_config.training.local_batch_size,
                self.model_args.max_seq_len,
            ),
            device=self.device,
        )
        labels = torch.randint(
            0,
            self.model_args.vocab_size,
            (
                self.job_config.training.local_batch_size,
                self.model_args.max_seq_len,
            ),
            device=self.device,
        )
        return tokens, labels

      total_tokens = 0
      total_time = 0.0

      for step in range(self.job_config.training.steps):
        with torchtitan.tools.profiling.maybe_enable_profiling(
            self.job_config.profiling,
            global_step=step,
            base_folder=self.job_config.job.dump_folder,
        ) as torch_profiler:
          step_start_time = time.time()
          self.optimizers.zero_grad()
          tokens, labels = get_batch()
          # train_context would take care of converting labels to DTensors using
          # https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.loss_parallel
          with self.train_context():  # pytype: disable=not-callable
            pred = self.model(tokens)
            loss = self.loss_fn(pred, labels)
            # need to free pred before bwd to avoid peaking memory
            del pred
            loss.backward()
          self.optimizers.step()

          step_time = time.time() - step_start_time

          if self.step_callback is not None:
            self.step_callback(step, self.model, loss)

          if step >= self.job_config.lr_scheduler.warmup_steps:
            total_tokens += (
                self.job_config.training.local_batch_size *
                self.model_args.max_seq_len)
            total_time += step_time

          tokens_per_sec = (
              self.job_config.training.local_batch_size *
              self.model_args.max_seq_len) / step_time

          logger.info(
              "Step %d/%d | Loss: %.4f | Step time: %.2f sec | "
              "Throughput: %.2f tokens/sec",
              step + 1,
              self.job_config.training.steps,
              loss.item(),
              step_time,
              tokens_per_sec,
          )
        if torch_profiler:
          torch_profiler.step()

      # Calculate average throughput excluding burn-in steps
      avg_throughput = total_tokens / total_time if total_time > 0 else 0
      logger.info(
          "\nAverage throughput on %s: %.2f tokens/sec",
          self.device,
          avg_throughput,
      )
      return avg_throughput

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
      logger.error("An error occurred: %s\n%s", e, traceback.format_exc())
      raise e


def start_trainer(config: JobConfig):
  """Initializes and starts the training process.

  Args:
    config: The JobConfig object containing training and model configuration.
  """
  rank = int(os.environ.get("RANK", 0))
  world_size = int(os.environ.get("WORLD_SIZE", 1))

  if rank == 0:
    torchtitan.tools.logging.init_logger()

  device = tpu_utils.get_device()

  if config.optimizer.implementation == "fused" and device.type == "tpu":
    # TODO b/45811390 - Remove this once fused optimizer is supported on TPU.
    logger.warning(
        "Fused optimizer is not supported on TPU, changing to foreach."
    )
    config.optimizer.implementation = "foreach"

  try:
    trainer = TrainerMinimal(device, rank, world_size, config)
    if rank == 0:
      logger.info(
          "Training with config: %s",
          json.dumps(config.to_dict(), indent=2, sort_keys=True)
      )
    trainer.train()
  finally:
    if torch.distributed.is_initialized():
      torch.distributed.destroy_process_group()
    logger.info("Process group destroyed")


if __name__ == "__main__":
  gmain.handle_main(start_trainer)
