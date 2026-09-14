# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer compatible with TorchTpu."""

import json
import os
import torch
import torch.distributed as dist
from torchtitan.config.configurable import Configurable
from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu.grad_norm import CompiledGradNorm
import torchtitan.experiments.tpu.profiler_workaround as profiler_workaround
import torchtitan.experiments.tpu.redacted  # trigger model registration
import torchtitan.experiments.tpu.redacted  # trigger model registration
import torchtitan.experiments.tpu.deepseek_v3   # trigger model registration
import torchtitan.experiments.tpu.flux  # trigger model registration
from torchtitan.experiments.tpu.graph_trainer.scan_guards import check_scan_was_applied
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
import torchtitan.experiments.tpu.llama3  # trigger model registration
import torchtitan.experiments.tpu.qwen3   # trigger model registration
import torchtitan.experiments.tpu.tpu_job_config as tpu_job_config_module
from torchtitan.tools.logging import init_logger, logger
import torchtitan.train


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

  compile_grad_norm:
    Compiles the DTensor-aware global gradient norm as a standalone static XLA
    graph. In-place clipping remains on PyTorch's eager path.
  """

  def __init__(self, job_config: Configurable.Config):
    super().__init__(job_config)

    self._compiled_grad_norm: CompiledGradNorm | None = None
    if isinstance(job_config, tpu_job_config_module.TPUTrainerConfig):
      from torchtitan.experiments.tpu.loss import build_cross_entropy_loss
      self.loss_fn = build_cross_entropy_loss(job_config)

      tpu_config = job_config.tpu_config
      if not tpu_config.enable_amp:
        logger.info("AMP is disabled, using uniform precision training.")
      self._compiled_grad_norm = (
          CompiledGradNorm() if tpu_config.compile_grad_norm else None
      )
      if tpu_config.compile_optimizer:
        from torchtitan.experiments.tpu.compiled_optimizers.separate_graph import apply_compiled_optimizer  # pylint: disable=g-import-not-at-top
        apply_compiled_optimizer(self)

  def _clip_grad_norm(self) -> torch.Tensor:
    if self._compiled_grad_norm is None:
      return super()._clip_grad_norm()
    parameters = [p for m in self.model_parts for p in m.parameters()]
    return self._compiled_grad_norm.clip_grad_norm_(
        parameters,
        self.config.training.max_norm,
    )

  def forward_backward_step(
      self,
      *,
      input_dict: dict[str, torch.Tensor],
      labels: torch.Tensor,
      global_valid_tokens: float,
  ) -> torch.Tensor:
    config = self.config
    use_graph_split = (
        isinstance(config, tpu_job_config_module.TPUTrainerConfig)
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
    inputs, labels, extra_kwargs = self.post_dataloading_process(
        input_dict, labels
    )

    with self.train_context():
      pred = self.model_parts[0](inputs, **extra_kwargs)
      loss, _ = self.loss_fn(pred, labels, global_valid_tokens)
      del pred

      # Materialize the forward graph without waiting for execution to
      # complete. This splits the deferred-op graph so the backward pass
      # compiles as a separate (smaller) XLA program. Autograd state is
      # unaffected — loss.backward() can still propagate gradients.
      synchronize(loss, wait=False)

      loss.backward()

    return loss


def start_trainer(
    train_config: tpu_job_config_module.TPUTrainerConfig
    | TpuGraphTrainer.Config,
):
  """Starts the training process."""
  rank = int(os.getenv("RANK", "0"))
  if rank == 0:
    init_logger()

  profiler_workaround.init(train_config)

  trainer: torchtitan.trainer.Trainer | None = None
  model_spec = train_config.model_spec
  assert model_spec is not None
  model_name = model_spec.name
  is_graph_trainer = isinstance(train_config, TpuGraphTrainer.Config)
  if model_name == "flux":
    # FluxTrainer owns a separate train_step and bypasses the TPU integration.
    if train_config.tpu_config.compile_grad_norm:
      raise ValueError("compile_grad_norm does not support FluxTrainer")
    from torchtitan.models.flux.trainer import FluxTrainer
    trainer_cls = FluxTrainer
  elif is_graph_trainer:
    trainer_cls = TpuGraphTrainer
  else:
    if isinstance(train_config, tpu_job_config_module.TPUTrainerConfig):
      if train_config.tpu_config.compile_mode == "fullgraph":
        logger.warning(
            "Running fullgraph compile mode (`%s`) on regular TPUTrainer. "
            "Note: this wraps forward+backward together via `torch.compile(model, fullgraph=True, backend='tpu')`, "
            "but does NOT capture the optimizer step right inside the graph.",
            train_config.tpu_config.compile_mode,
        )
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
  if train_config.debug.seed is None:
    train_config.debug.seed = 42
    logger.info(
        f"Setting default seed {train_config.debug.seed} to avoid hang on TPU"
    )

  # Apply TPU-specific workarounds before model parallelization
  # Moved from deprecated args.py files
  orig_parallelize = model_spec.parallelize_fn

  def patched_parallelize(model, *args, **kwargs):
    from torchtitan.experiments.tpu import workarounds

    workarounds.apply_patches(model, train_config)
    if isinstance(train_config, tpu_job_config_module.TPUTrainerConfig):
      if model_spec.name in ("llama3", "qwen3"):
        kwargs["tpu_compile_mode"] = train_config.tpu_config.compile_mode
    if model_spec.name in (
        "redacted",
        "redacted_tpu",
        "redacted",
        "redacted_tpu",
    ):
      parallel_dims = kwargs.get("parallel_dims")
      return orig_parallelize(model, parallel_dims, train_config)
    return orig_parallelize(model, *args, **kwargs)

  if not is_graph_trainer:
    model_spec.parallelize_fn = patched_parallelize

  try:
    trainer = trainer_cls(train_config)
    # Asked of the built model rather than wrapped around `parallelize_fn`:
    # the graph trainer -- the only lane scan runs on -- builds its own config
    # and its own parallelize_fn, so a wrapper on the one patched above is
    # never called.
    for part in trainer.model_parts:
      check_scan_was_applied(
          part,
          use_scan_layers=(
              train_config.compile.use_scan_layers
              if is_graph_trainer
              else False
          ),
      )
    train_config.maybe_log()

    if train_config.checkpoint.create_seed_checkpoint:
      assert int(os.environ["WORLD_SIZE"]) == 1, (
          "Must create seed checkpoint using a single device, to disable"
          " sharding."
      )
      assert (
          train_config.checkpoint.enable
      ), "Must enable checkpointing when creating a seed checkpoint."
      trainer.checkpointer.save(curr_step=0, last_step=True)
      logger.info("Created seed checkpoint")
    else:
      if rank == 0:
        logger.info(
            "Training with config: %s",
            json.dumps(train_config.to_dict(), indent=2),
        )
      trainer.train()
  except Exception:
    if trainer:
      trainer.close()
    raise
  else:
    trainer.close()
    if torch.distributed.is_initialized():
      torch.distributed.barrier()
      torch.distributed.destroy_process_group()
    logger.info("Process group destroyed")


if __name__ == "__main__":
  gmain.handle_main(start_trainer)
