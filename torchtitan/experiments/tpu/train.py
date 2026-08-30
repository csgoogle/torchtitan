# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer compatible with TorchTpu."""

import argparse
import functools
import json
import tempfile
from typing import Any, List, Optional, Sequence, Tuple, cast

from absl import flags
from absl.flags import argparse_flags
import os
import torch
import torch.distributed as dist
from torchtitan.config.configurable import Configurable
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.tpu import gmain
import torchtitan.experiments.tpu.profiler_workaround as profiler_workaround
import torchtitan.experiments.tpu.redacted  # trigger model registration
import torchtitan.experiments.tpu.redacted  # trigger model registration
import torchtitan.experiments.tpu.deepseek_v3   # trigger model registration
import torchtitan.experiments.tpu.flux  # trigger model registration
from torchtitan.experiments.tpu.graph_trainer.config import (
    to_tpu_graph_trainer_config,
)
import torchtitan.experiments.tpu.llama3  # trigger model registration
import torchtitan.experiments.tpu.qwen3   # trigger model registration
import torchtitan.experiments.tpu.tpu_job_config as tpu_job_config_module
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
import torchtitan.train

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

    if isinstance(job_config, tpu_job_config_module.TPUTrainerConfig):
      from torchtitan.experiments.tpu.loss import build_cross_entropy_loss
      self.loss_fn = build_cross_entropy_loss(job_config)

      tpu_config = job_config.tpu_config
      if not tpu_config.enable_amp:
        logger.info("AMP is disabled, using uniform precision training.")
      if tpu_config.compile_optimizer:
        from torchtitan.experiments.tpu.compiled_optimizers.separate_graph import apply_compiled_optimizer  # pylint: disable=g-import-not-at-top
        apply_compiled_optimizer(self)


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


def start_trainer(train_config: tpu_job_config_module.TPUTrainerConfig):
  """Starts the training process."""
  rank = int(os.getenv("RANK", "0"))
  if rank == 0:
    init_logger()

  if (
      utils.device_type == "tpu"
      and train_config.optimizer.implementation == "fused"
  ):
    # TODO: b/45811390 - Remove this once fused optimizer is supported on TPU.
    logger.warning(
        "Fused optimizer is not supported on TPU, changing to foreach."
    )
    train_config.optimizer.implementation = "foreach"

  profiler_workaround.init(train_config)

  trainer: Optional[torchtitan.trainer.Trainer] = None
  model_spec = train_config.model_spec
  assert model_spec is not None
  model_name = model_spec.name
  if model_name == "flux":
    from torchtitan.models.flux.trainer import FluxTrainer
    trainer_cls = FluxTrainer
  # Note: TpuGraphTrainer is designed primarily for joint full-graph tracing
  # ("aot_fx_trace" / "fullgraph"). We keep `--tpu-config.use-graph-trainer`
  # as an independent switch for experimentation, but enabling `use_graph_trainer`
  # without fullgraph compilation (e.g. alongside `layer` compile mode) can cause
  # severe HBM fragmentation and stack reservation collisions (e.g. 78.46G memory
  # reservation bottlenecks across multiple disjoint overlay modules).
  elif (
      isinstance(train_config, tpu_job_config_module.TPUTrainerConfig)
      and (
          train_config.tpu_config.use_graph_trainer
          or train_config.tpu_config.compile_mode == "aot_fx_trace"
      )
  ):
    if (
        train_config.tpu_config.use_graph_trainer
        and train_config.tpu_config.compile_mode
        not in ("aot_fx_trace", "fullgraph")
    ):
      logger.warning(
          "TpuGraphTrainer enabled without fullgraph compile mode"
          f" (current mode: {train_config.tpu_config.compile_mode}). This may"
          " lead to severe HBM fragmentation and stack reservation OOMs."
      )
    from torchtitan.experiments.tpu.graph_trainer import TpuGraphTrainer, TpuGraphTrainerConfig

    if model_name == "qwen3":
      from torchtitan.experiments.graph_trainer.qwen3 import model_registry as gt_registry
    elif model_name == "llama3":
      from torchtitan.experiments.graph_trainer.llama3 import model_registry as gt_registry
    elif model_name == "deepseek_v3":
      from torchtitan.experiments.graph_trainer.deepseek_v3 import model_registry as gt_registry
    else:
      raise ValueError(
          f"Model {model_name} is not currently supported with GraphTrainer."
      )

    gt_cfg = to_tpu_graph_trainer_config(
        train_config, gt_registry, target_cls=TpuGraphTrainer.Config
    )

    splash_obj = cast(Any, getattr(train_config, "splash_attention_kernel", None))
    moe_obj = cast(Any, getattr(train_config, "moe", getattr(train_config, "redacted", None)))
    tpu_extra = TpuGraphTrainerConfig(
        device="tpu",
        attention_backend=getattr(
            splash_obj,
            "sa_attention_backend",
            "splash"
            if getattr(splash_obj, "use_splash_attention_kernel", False)
            else "sdpa-no-scale",
        )
        if splash_obj is not None
        else "sdpa-no-scale",
        moe_enabled=getattr(moe_obj, "enabled", False)
        if moe_obj is not None
        else False,
        moe_padded_dispatcher=getattr(moe_obj, "padded_dispatcher", False)
        if moe_obj is not None
        else False,
        moe_capacity_factor=getattr(moe_obj, "capacity_factor", 1.25)
        if moe_obj is not None
        else 1.25,
        ep_degree=getattr(
            train_config.parallelism,
            "expert_parallel_degree",
            getattr(train_config.parallelism, "ep", 1),
        ),
        optimizer_in_graph=train_config.tpu_config.optimizer_in_graph,
        optimizer_compile=train_config.tpu_config.compile_optimizer,
        compile_mode=train_config.tpu_config.compile_mode,
        enable_mfu_benchmark=train_config.tpu_config.enable_mfu_benchmark,
        mfu_benchmark_start_step=train_config.tpu_config.mfu_benchmark_start_step,
        mfu_benchmark_final_step=train_config.training.steps,
        job_config=train_config,
    )

    def trainer_cls(cfg):
      return TpuGraphTrainer(gt_cfg, tpu_extra=tpu_extra)
  else:
    if isinstance(train_config, tpu_job_config_module.TPUTrainerConfig):
      if train_config.tpu_config.compile_mode == "fullgraph":
        logger.warning(
            "Running fullgraph compile mode (`%s`) on regular Trainer (`--tpu-config.use-graph-trainer=False`). "
            "Note: this wraps forward+backward together via `torch.compile(model, fullgraph=True, backend='tpu')`, "
            "but does NOT capture the optimizer step right inside the graph.",
            train_config.tpu_config.compile_mode,
        )
      if getattr(train_config.compile, "optimizer_in_graph", False) or getattr(
          train_config.tpu_config, "optimizer_in_graph", False
      ):
        logger.warning(
            "`optimizer_in_graph` (`--compile.optimizer-in-graph` or `--tpu-config.optimizer-in-graph`) "
            "is set while running on regular Trainer (`use_graph_trainer=False`). "
            "Regular Trainer does not support optimizer-in-graph (capturing `optimizer.step()` inside the graph). "
            "To enable joint fwd+bwd+optimizer compilation, you must use `--tpu-config.use-graph-trainer`."
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
        if (
            train_config.tpu_config.use_graph_trainer
            or train_config.tpu_config.compile_mode == "aot_fx_trace"
        ):
          kwargs.pop("tpu_compile_mode", None)
        else:
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

  model_spec.parallelize_fn = patched_parallelize

  try:
    trainer = trainer_cls(train_config)
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
            json.dumps(train_config.to_dict(), indent=2, sort_keys=True),
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
