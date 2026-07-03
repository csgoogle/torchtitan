"""Minimal trainer for TorchTitan models."""

import argparse
from collections.abc import Sequence
import json
import os
import time
import traceback
import typing
from typing import Callable, List, Optional, Tuple

from absl import flags
from absl.flags import argparse_flags
import torch
import torch.distributed as dist
from torch import nn
import torchtitan.components.tokenizer
import torchtitan.config
import torchtitan.distributed
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu import utils as tpu_utils
import torchtitan.experiments.tpu.deepseek_v3   # trigger model registration
import torchtitan.experiments.tpu.flux  # trigger model registration
import torchtitan.experiments.tpu.llama3  # trigger model registration
import torchtitan.experiments.tpu.profiler_workaround as profiler_workaround
import torchtitan.experiments.tpu.qwen3   # trigger model registration
import torchtitan.experiments.tpu.tpu_job_config
import torchtitan.experiments.tpu.workarounds as workarounds
from torchtitan.tools import utils
import torchtitan.tools.logging
import torchtitan.tools.profiler as torchtitan_profiler
import torchtitan.trainer
# import torchtitan.tools.profiling  # Deprecated upstream


TORCH_DTYPE_MAP = torchtitan.config.TORCH_DTYPE_MAP
JobConfig = torchtitan.trainer.Trainer.Config
ParallelDims = torchtitan.distributed.ParallelDims
logger = torchtitan.tools.logging.logger

TrainStepCallback = Callable[[int, nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], None]


def _gather_along_first_dim(tensor: torch.Tensor, group) -> torch.Tensor:
  """Gathers a standard PyTorch tensor along its first (batch) dimension across ranks.

  This is used to reconstruct the full global batch of tokens/labels from the sharded
  local batches on each rank, allowing the step callback to record/verify the global state.
  """
  world_size = dist.get_world_size(group)
  if world_size <= 1:
    return tensor

  output_shape = list(tensor.shape)
  output_shape[0] *= world_size
  output_tensor = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)

  dist.all_gather_into_tensor(output_tensor, tensor, group=group)
  return output_tensor


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
          base_folder=job_config.dump_folder,
      )

    parallelism_config = self.job_config.parallelism
    self.parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        ep=parallelism_config.expert_parallel_degree,
        world_size=self.world_size,
    )

    logger.info(f"parallel_dims: {self.parallel_dims}")

    # Workaround if we run without distrubuted training.
    batch_degree, batch_rank = 1, 0
    self.batch_mesh = None
    if world_size > 1:
      if self.parallel_dims.dp_enabled:
        self.batch_mesh = self.parallel_dims.get_mesh("batch")
        batch_degree, batch_rank = (
            self.batch_mesh.size(), self.batch_mesh.get_local_rank())
      else:
        batch_degree, batch_rank = 1, 0

      logger.info(f"world_mesh: {self.parallel_dims.world_mesh}")

      seed = job_config.debug.seed or 42
      torch.manual_seed(seed)
      logger.info(
          "Set manual seed to %d on all ranks (workaround for set_determinism"
          " hang)",
          seed,
      )
      # TODO b/498659628: Re-enable set_determinism once the hang is fixed.
      # # Set random seed, and maybe enable deterministic mode
      # # (mainly for debugging, expect perf loss).
      # dist_utils.set_determinism(
      #     self.parallel_dims,
      #     self.device,
      #     job_config.debug,
      #     distinct_seed_mesh_dims=["pp"],
      # )

    assert job_config.model_spec is not None
    self.model_spec = job_config.model_spec

    self.tokenizer = typing.cast(
        torchtitan.components.tokenizer.HuggingFaceTokenizer,
        job_config.tokenizer.build(tokenizer_path=job_config.hf_assets_path)
    )

    self.dataloader = job_config.dataloader.build(
        dp_world_size=batch_degree,
        dp_rank=batch_rank,
        tokenizer=self.tokenizer,
        seq_len=job_config.training.seq_len,
        local_batch_size=job_config.training.local_batch_size,
    )

    model_config = self.model_spec.model
    model_config.update_from_config(trainer_config=job_config)
    self.model_config = model_config

    if (
        hasattr(self.model_config, "vocab_size")
        and self.model_config.vocab_size is not None
        and self.tokenizer.vocab_size > self.model_config.vocab_size
    ):
      logger.warning(
          "Vocab size for tokenizer is %d while model config vocab size is %d."
          " Expanding model config vocab size to match tokenizer vocab size.",
          self.tokenizer.vocab_size,
          self.model_config.vocab_size,
      )
      self.model_config.vocab_size = self.tokenizer.vocab_size  # pyrefly: ignore[missing-attribute]

    logger.info(
        f"Building {self.model_spec.name} {self.model_spec.flavor}"
        f" with {self.model_config}"
    )

    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
    ):
      model = model_config.build()

    # calculate model size and flops per token
    (
        model_param_count,
        _,
    ) = model_config.get_nparams_and_flops(model, job_config.training.seq_len)

    logger.info(
        f"Model {self.model_spec.name} {self.model_spec.flavor} "
        f"size: {model_param_count:,} total parameters"
    )

    self.loss_fn = job_config.loss.build(compile_config=job_config.compile)

    # Apply TPU-specific patches/workarounds
    workarounds.apply_patches(model, job_config)

    if self.world_size > 1:
      self.model_spec.parallelize_fn(
          model,
          parallel_dims=self.parallel_dims,
          training=job_config.training,
          parallelism=job_config.parallelism,
          compile_config=job_config.compile,
          ac_config=job_config.activation_checkpoint,
          dump_folder=job_config.dump_folder,
      )

    logger.info(
        f"Moving model {self.model_spec.name}"
        f" {self.model_spec.flavor} to device {self.device}"
    )
    self.model = model.to_empty(device=self.device)
    with torch.no_grad():
      self.model.init_weights()

    from torchtitan.components.loss import ChunkedCELoss
    if isinstance(self.loss_fn, ChunkedCELoss):
      lm_head = self.model.lm_head
      assert lm_head is not None, "Model must have lm_head for ChunkedCELoss"
      self.loss_fn.set_lm_head(lm_head)
      self.model._skip_lm_head = True

    assert (
        self.device.type != "tpu" or
        job_config.optimizer.implementation != "fused"
    ), ("TODO b/45811390 - Fused optimizer not supported on TPU."
        "You can disable this by setting --optimizer.implementation=foreach")
    self.optimizers = job_config.optimizer.build(model_parts=[self.model])

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
        vocab_size = getattr(self.model_config, "vocab_size", 32000)
        tokens = torch.randint(
            0,
            vocab_size,
            (
                self.job_config.training.local_batch_size,
                self.job_config.training.seq_len,
            ),
            device=self.device,
        )
        labels = torch.randint(
            0,
            vocab_size,
            (
                self.job_config.training.local_batch_size,
                self.job_config.training.seq_len,
            ),
            device=self.device,
        )
        return tokens, labels

      total_tokens = 0
      total_time = 0.0

      with torchtitan_profiler.Profiler(
          self.job_config.profiler,
          global_step=0,
          base_folder=self.job_config.dump_folder,
      ) as profiler:
        for step in range(self.job_config.training.steps):
          step_start_time = time.time()
          self.optimizers.zero_grad()
          tokens, labels = get_batch()

          # In distributed settings, we sum locally calculated valid tokens
          # across all data-parallel ranks to get the global token count,
          # which is used to scale the loss and gradients.
          # We need this global count because TorchTitan's FSDP
          # implementation disables automatic gradient division in favor of
          # global token-based scaling.
          # Valid tokens exclude those marked with IGNORE_INDEX (pytorch
          # default is -100), representing padding or ignored tokens.
          local_valid_tokens = (labels != IGNORE_INDEX).sum().to(self.device)
          if self.world_size > 1 and self.parallel_dims.dp_enabled:
            global_valid_tokens = dist_utils.dist_sum(
                local_valid_tokens, self.batch_mesh
            )
          else:
            global_valid_tokens = local_valid_tokens.float()

          # train_context would take care of converting labels to DTensors using
          # https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.loss_parallel
          with self.train_context():  # pytype: disable=not-callable
            pred = self.model(tokens)
            loss = self.loss_fn(pred, labels, global_valid_tokens)
            # need to free pred before bwd to avoid peaking memory
            del pred
            loss.backward()
          self.optimizers.step()

          step_time = time.time() - step_start_time

          if self.step_callback is not None:
            gathered_tokens = tokens
            gathered_labels = labels
            if self.batch_mesh is not None:
              dp_group = self.batch_mesh.get_group()
              gathered_tokens = _gather_along_first_dim(tokens, dp_group)
              gathered_labels = _gather_along_first_dim(labels, dp_group)
            self.step_callback(step, self.model, loss, gathered_tokens, gathered_labels)

          if step >= self.job_config.lr_scheduler.warmup_steps:
            total_tokens += (
                self.job_config.training.local_batch_size *
                self.job_config.training.seq_len)
            total_time += step_time

          tokens_per_sec = (
              self.job_config.training.local_batch_size *
              self.job_config.training.seq_len) / step_time

          logger.info(
              "Step %d/%d | Loss: %.4f | Step time: %.2f sec | "
              "Throughput: %.2f tokens/sec",
              step + 1,
              self.job_config.training.steps,
              loss.item(),
              step_time,
              tokens_per_sec,
          )
          if profiler:
            profiler.step()

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

  import sys
  logger.info("JAX LOADED BEFORE INIT: %s", "jax" in sys.modules)
  profiler_workaround.init(config)
  logger.info("JAX LOADED AFTER INIT: %s", "jax" in sys.modules)
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
