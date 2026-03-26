# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

import jax
import torch
import torch.distributed

import torchtitan.config
import torchtitan.tools.logging
import torchtitan.tools.profiling

import os as os

ProfilingConfig = torchtitan.config.Profiling
logger = torchtitan.tools.logging.logger
maybe_enable_memory_snapshot = (
    torchtitan.tools.profiling.maybe_enable_memory_snapshot
)

__all__ = ["maybe_enable_profiling", "maybe_enable_memory_snapshot"]


@contextlib.contextmanager
def maybe_enable_profiling(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
):
  """Context manager to conditionally enable JAX profiling based on config."""
  enable_profiling = profiling_config.enable_profiling

  if enable_profiling:
    trace_dir = os.path.join(base_folder, profiling_config.save_traces_folder)
    profile_freq, warmup, active = (
        profiling_config.profile_freq,
        profiling_config.profiler_warmup,
        profiling_config.profiler_active,
    )

    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    class JaxProfiler:
      """Context manager for JAX profiling.

      This class manages the start and stop of JAX traces based on the
      configured profiling frequency, warmup, and active periods.
      """

      def __init__(self, step_num: int) -> None:
        self.step_num = step_num
        self.tracing = False

      def step(self):
        """Steps the profiler, potentially starting or stopping a JAX trace.

        This method is called at each global step. It manages the JAX profiler
        based on the configured `profile_freq`, `profiler_warmup`, and
        `profiler_active` settings. A trace is started at the beginning of the
        active phase and stopped at the end.
        """
        self.step_num += 1
        cycle_step = self.step_num % profile_freq
        wait = profile_freq - warmup - active

        # Active phase starts
        if cycle_step == wait + warmup:
          curr_trace_dir_name = f"iteration_{self.step_num}"
          self.curr_trace_dir = os.path.join(
              trace_dir, curr_trace_dir_name, f"rank_{rank}", leaf_folder
          )
          if local_rank == 0:
            logger.info(
                "Starting JAX profiler trace at step %s", self.step_num
            )
            os.makedirs(self.curr_trace_dir, exist_ok=True)
            jax.profiler.start_trace(self.curr_trace_dir)
          self.tracing = True

        # Active phase ends
        elif (
            self.tracing
            and cycle_step == (wait + warmup + active) % profile_freq
        ):
          if local_rank == 0:
            logger.info(
                "Stopping JAX profiler trace at step %s", self.step_num
            )
            jax.profiler.stop_trace()
          self.tracing = False

    logger.info("JAX Profiling active. Traces will be saved at %s", trace_dir)

    if rank == 0:
      os.makedirs(trace_dir, exist_ok=True)

    profiler = JaxProfiler(global_step)

    try:
      yield profiler
    finally:
      if profiler.tracing and local_rank == 0:
        jax.profiler.stop_trace()
  else:
    yield None
