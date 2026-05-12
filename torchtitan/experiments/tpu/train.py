# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer compatible with TorchTpu."""

import argparse
import contextlib
import functools
import json
import tempfile
from typing import List, Optional, Sequence, Tuple

from absl import flags
from absl.flags import argparse_flags
import os as os
from absl import app


class TPUTrainer(torchtitan.train.Trainer):
  """Trainer subclass with TPU-specific optimizations.

  Adds the following optional behaviors controlled by tpu_config flags:

  use_graph_split:
    Inserts a `synchronize(loss, wait=False)` call between the forward+loss
    computation and the backward pass. This splits the monolithic XLA
    computation graph into two smaller graphs (forward and backward),
    significantly reducing first-step XLA compilation time.
    The torch_tpu synchronize() materializes the forward graph without
    breaking autograd, so loss.backward() still propagates gradients
    correctly through the full model.
  """

  def __init__(self, job_config: Configurable.Config):
    super().__init__(job_config)
    if isinstance(job_config, tpu_job_config_module.TPUJobConfig):
      tpu_config = job_config.tpu_config
      if not tpu_config.enable_amp:
        logger.info("AMP is disabled, using uniform precision training.")
        self.maybe_enable_amp = contextlib.nullcontext()
      if self.parallel_dims.dp_replicate_enabled and self.device.type == "tpu":
        # TODO b/494360665: remove this once torch.autocast is supported on TPU.
        # DDP-replicate on TPU uses fully_shard instead of autocast.
        logger.info(
            "Mixed precision training is handled by fully_shard (TPU replicate,"
            f"param={self.job_config.training.mixed_precision_param}, "
            f"reduce={self.job_config.training.mixed_precision_reduce})")
        self.maybe_enable_amp = contextlib.nullcontext()

  def forward_backward_step(
      self,
      *,
      input_dict: dict[str, torch.Tensor],
      labels: torch.Tensor,
      global_valid_tokens: torch.Tensor,
  ) -> torch.Tensor:
    config = self.job_config
    use_graph_split = (
        isinstance(config, tpu_job_config_module.TPUJobConfig)
        and config.tpu_config.use_graph_split
        and not self.parallel_dims.pp_enabled
    )
    if not use_graph_split:
      return super().forward_backward_step(
          input_dict=input_dict,
          labels=labels,
          global_valid_tokens=global_valid_tokens,
      )

    # Graph-split forward/backward for TPU:
    # The forward pass and loss computation are materialized as one XLA graph,
    # and the backward pass is compiled separately. This reduces peak
    # compilation time vs. one monolithic forward+backward graph.
    from torch_tpu._internal.sync import synchronize  # pylint: disable=g-import-not-at-top

    assert len(self.model_parts) == 1
    inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
        input_dict, labels
    )

    with self.train_context():
      with self.maybe_enable_amp:
        pred = self.model_parts[0](inputs, **extra_inputs, **extra_kwargs)
        loss_sum = self.loss_fn(pred, labels)
        loss = loss_sum / global_valid_tokens
      del pred

      # Materialize the forward graph without waiting for execution to
      # complete. This splits the deferred-op graph so the backward pass
      # compiles as a separate (smaller) XLA program. Autograd state is
      # unaffected — loss.backward() can still propagate gradients.
      synchronize(loss, wait=False)

      loss.backward()

    return loss


def start_trainer(config: tpu_job_config_module.TPUJobConfig):
  """Starts the training process."""
  rank = int(os.getenv("RANK", "0"))
  if rank == 0:
    init_logger()

  if (tpu_utils.get_device_type() == "tpu" and
      config.optimizer.implementation == "fused"):
    # TODO: b/45811390 - Remove this once fused optimizer is supported on TPU.
    logger.warning(
        "Fused optimizer is not supported on TPU, changing to foreach."
    )
    config.optimizer.implementation = "foreach"

  torchtitan.train.maybe_enable_profiling = functools.partial(
      profiler_workaround.maybe_enable_profiling, job_config=config
  )

  trainer: Optional[torchtitan.train.Trainer] = None
  if config.model.name == "flux":
    from torchtitan.models.flux.train import FluxTrainer
    trainer_cls = FluxTrainer
  else:
    trainer_cls = TPUTrainer

  # Synchronize all ranks before trainer initialization. On TPU (XLA), the
  # device-setup graph for rank 0 may compile faster than for other ranks,
  # letting rank 0 race past set_determinism's broadcast_object_list into
  # train_step. There rank 0 waits at an XLA all_reduce while the other ranks
  # are still compiling their broadcast graph — a distributed deadlock. This
  # barrier forces all ranks to flush their pending device-init XLA ops
  # simultaneously before any trainer code runs.
  if dist.is_initialized():
    dist.barrier()

  # TODO: b/498659628 - Remove once hanging is fixed.
  if config.debug.seed is None:
    config.debug.seed = 42
    logger.info(
        f"Setting default seed {config.debug.seed} to avoid hang on TPU"
    )

  try:
    trainer = trainer_cls(config)
    config.maybe_log()

    if config.checkpoint.create_seed_checkpoint:
      assert int(os.environ["WORLD_SIZE"]) == 1, (
          "Must create seed checkpoint using a single device, to disable"
          " sharding."
      )
      assert (
          config.checkpoint.enable
      ), "Must enable checkpointing when creating a seed checkpoint."
      trainer.checkpointer.save(curr_step=0, last_step=True)
      logger.info("Created seed checkpoint")
    else:
      if rank == 0:
        logger.info(
            "Training with config: %s",
            json.dumps(config.to_dict(), indent=2, sort_keys=True)
        )
      trainer.train()
  except Exception:
    if trainer:
      trainer.close()
    raise
  else:
    trainer.close()
    torch.distributed.destroy_process_group()
    logger.info("Process group destroyed")


if __name__ == "__main__":
  gmain.handle_main(start_trainer)
