import contextlib
from typing import Any
import os as os
import torch

from torchtitan.tools.logging import logger
import torchtitan.tools.profiler as profiler_tool
ProfilingConfig = profiler_tool.Profiler.Config

@contextlib.contextmanager
def original_maybe_enable_profiling(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
):
  with profiler_tool.Profiler(
      profiling_config,
      global_step=global_step,
      base_folder=base_folder,
      leaf_folder=leaf_folder,
  ) as prof:
    yield prof

# For JAX profiler
try:
  import jax
except ImportError:
  jax = None


class ScheduledProfiler:
  """Base class for TPU profilers managing step logic."""

  def __init__(
      self,
      step_num: int,
      profile_freq: int,
      profiler_warmup: int,
      profiler_active: int,
  ) -> None:
    self.step_num = step_num
    self.profile_freq = profile_freq
    self.profiler_warmup = profiler_warmup
    self.profiler_active = profiler_active
    self.tracing = False

  def step(self):
    self.step_num += 1
    if self.profile_freq == 0:
      return
    cycle_step = self.step_num % self.profile_freq
    wait = self.profile_freq - self.profiler_warmup - self.profiler_active
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if cycle_step == wait + self.profiler_warmup:
      if local_rank == 0:
        self._start_trace()
      self.tracing = True
    elif (
        self.tracing
        and cycle_step
        == (wait + self.profiler_warmup + self.profiler_active)
        % self.profile_freq
    ):
      if local_rank == 0:
        self._stop_trace()
      self.tracing = False

  def _start_trace(self):
    raise NotImplementedError

  def _stop_trace(self):
    raise NotImplementedError


class JaxProfiler(ScheduledProfiler):
  """Context manager for JAX profiling."""

  def __init__(
      self,
      step_num: int,
      profiling_config: ProfilingConfig,
      base_folder: str,
      rank: int,
      leaf_folder: str,
  ) -> None:
    super().__init__(
        step_num,
        profiling_config.profile_freq,
        profiling_config.profiler_warmup,
        profiling_config.profiler_active,
    )
    self.base_folder = base_folder
    self.rank = rank
    self.leaf_folder = leaf_folder
    self.profiling_config = profiling_config

  def _start_trace(self):
    trace_dir = os.path.join(
        self.base_folder, self.profiling_config.save_traces_folder
    )
    curr_trace_dir_name = f"iteration_{self.step_num}"
    curr_trace_dir = os.path.join(
        trace_dir, curr_trace_dir_name, f"rank_{self.rank}", self.leaf_folder
    )
    logger.info(
        "Starting JAX profiler warmup at step %d. Trace dir: %s",
        self.step_num,
        curr_trace_dir,
    )
    os.makedirs(curr_trace_dir, exist_ok=True)
    assert jax is not None
    jax.profiler.start_trace(curr_trace_dir)

  def _stop_trace(self):
    logger.info("Stopping JAX profiler before step %d", self.step_num)
    assert jax is not None
    jax.profiler.stop_trace()


@contextlib.contextmanager
def maybe_enable_profiling(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    job_config=None,
    base_folder: str = "",
    leaf_folder: str = "",
):
  base_folder = os.environ.get("TEST_TMPDIR", base_folder)
  enable_profiling = profiling_config.enable_profiling

  tpu_config = getattr(job_config, "tpu_config", None)
  use_jax = getattr(tpu_config, "use_jax_profiler", False)
  use_internal = False

  rank = (
      torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
  )

  if enable_profiling and not use_jax:
    logger.info("Using native profiler.")
    with original_maybe_enable_profiling(
        profiling_config,
        global_step=global_step,
        base_folder=base_folder,
        leaf_folder=leaf_folder,
    ) as prof:
      yield prof

  elif use_jax and jax:
    logger.info("JAX Profiling active. Traces will be saved at %s", base_folder)
    profiler = JaxProfiler(
        global_step, profiling_config, base_folder, rank, leaf_folder
    )
    try:
      yield profiler
    finally:
      if profiler.tracing and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        jax.profiler.stop_trace()

  else:
    yield None
