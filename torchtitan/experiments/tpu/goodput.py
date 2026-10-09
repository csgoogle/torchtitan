"""Goodput recording and monitoring for TorchTitan on TPU.

Wires the ML Goodput Measurement library (`ml-goodput-measurement`) into the
TPU graph trainer. The library has two halves:

* Recording: a `GoodputRecorder` on the lead process writes start/end
  timestamps (job, TPU init, training preparation, data loading, steps and
  custom badput events) to the Cloud Logging log `goodput_<job_name>`.
* Monitoring: a `GoodputMonitor` on the lead process reads that log in a
  background process, computes goodput/badput and uploads them to TensorBoard
  and Google Cloud Monitoring (`compute.googleapis.com/workload/*`).

Both are off by default (`--goodput.enable-goodput-recording`,
`--goodput.enable-goodput-monitoring`), because monitoring starts background
GCP traffic. Everything here is a no-op when goodput is disabled, when the
library is not installed, or on non-lead processes, and a goodput failure is
logged and swallowed instead of interrupting training.
"""

import contextlib
import dataclasses
import datetime
import enum
import os
from typing import Any, Callable, Iterator, Optional

import torch.distributed as dist
from torchtitan.tools.logging import logger


@dataclasses.dataclass
class GoodputConfig:
  """Goodput options, exposed on the CLI as `--goodput.*`."""

  enable_goodput_recording: bool = False
  """Record goodput events to Cloud Logging from the lead process."""

  enable_goodput_monitoring: bool = False
  """Compute goodput from the recorded events and upload it to TensorBoard
  and Google Cloud Monitoring from the lead process."""

  job_name: str = ""
  """Goodput job name; events go to the Cloud Logging log `goodput_<job_name>`.
  Use a new name for every run, and the same name when a run restarts or
  resumes. Defaults to a timestamped name, which can't track restarts."""

  upload_interval_seconds: int = 30
  """How often the monitor recomputes and uploads cumulative goodput."""

  tensorboard_dir: str = ""
  """TensorBoard (and monitor cache) directory. Defaults to
  `<dump_folder>/goodput`. A `gs://` path also enables the GCS cache sync."""

  monitor_step_time_deviation: bool = True
  """Also compute ideal step time and step time deviation."""

  step_deviation_interval_seconds: int = 30
  """Query interval for step time deviation."""

  configured_ideal_step_time: Optional[float] = None
  """Optional ideal step time, in seconds, used for step time deviation."""

  enable_gcp_goodput_metrics: bool = True
  """Upload goodput metrics to Google Cloud Monitoring."""

  enable_gcp_step_deviation_metrics: bool = True
  """Upload step time deviation metrics to Google Cloud Monitoring."""

  enable_rolling_window_goodput: bool = False
  """Also upload goodput over rolling windows (`interval_goodput` etc.)."""

  rolling_windows_seconds: tuple[int, ...] = (3600, 86400)
  """Rolling window sizes, in seconds."""

  sync_device_before_step: bool = False
  """Wait for queued device work before recording each step start, so step
  boundaries follow TPU execution instead of host dispatch. Accurate step
  times at the cost of host run-ahead."""


class GoodputEvent(enum.Enum):
  """Recorder events; values match the library's `record_<value>_*_time`."""

  JOB = "job"
  TPU_INIT = "tpu_init"
  TRAINING_PREPARATION = "training_preparation"
  DATA_LOADING = "data_loading"


# Custom badput event types recorded by the graph trainer.
CHECKPOINT_SAVE_BADPUT = "checkpoint_save"
EVAL_BADPUT = "eval"

_recorder: Any = None
_job_name: str = ""


def _import_goodput():
  """Returns the library's `(goodput, monitoring)` modules, or None."""
  try:
    from ml_goodput_measurement import goodput  # pylint: disable=g-import-not-at-top
    from ml_goodput_measurement import monitoring  # pylint: disable=g-import-not-at-top
  except ImportError:
    return None
  return goodput, monitoring


def is_lead_process() -> bool:
  """Whether this process is global rank 0.

  Uses the process group once it exists. Before that, torchrun's `RANK` is
  used; if it is missing in a multi-process job the rank is unknown, so the
  process is not treated as the lead (avoids every rank writing events).
  """
  if dist.is_available() and dist.is_initialized():
    return dist.get_rank() == 0
  rank = os.environ.get("RANK")
  if rank is not None:
    return int(rank) == 0
  return int(os.environ.get("WORLD_SIZE", "1")) <= 1


def resolve_job_name(config: GoodputConfig) -> str:
  if config.job_name:
    return config.job_name
  name = "torchtitan_" + datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y%m%d%H%M%S"
  )
  logger.warning(
      "Goodput: --goodput.job-name is not set; using %s. Restarts of this run"
      " will be recorded as separate jobs.",
      name,
  )
  return name


def initialize(config: GoodputConfig) -> None:
  """Creates the global recorder on the lead process if recording is on."""
  global _recorder, _job_name
  _recorder = None
  _job_name = ""
  if not (config.enable_goodput_recording or config.enable_goodput_monitoring):
    return
  if not is_lead_process():
    return
  _job_name = resolve_job_name(config)
  if not config.enable_goodput_recording:
    return
  modules = _import_goodput()
  if modules is None:
    logger.warning(
        "Goodput: ml-goodput-measurement is not installed; recording disabled."
    )
    return
  goodput, _ = modules
  try:
    _recorder = goodput.GoodputRecorder(
        job_name=_job_name,
        logger_name=f"goodput_{_job_name}",
        logging_enabled=True,
    )
    logger.info("Goodput: created GoodputRecorder for job %s", _job_name)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Goodput: could not create recorder: %s", e)


def get_job_name() -> str:
  return _job_name


def enabled() -> bool:
  """Whether this process records goodput events."""
  return _recorder is not None


def verify_lead_process() -> None:
  """Drops the recorder if the process group says this is not rank 0."""
  global _recorder
  if _recorder is not None and not is_lead_process():
    logger.warning(
        "Goodput: rank %d is not the lead process; disabling its recorder.",
        dist.get_rank(),
    )
    _recorder = None


def shutdown() -> None:
  """Flushes buffered events and drops the global recorder."""
  global _recorder
  if _recorder is not None:
    try:
      _recorder.flush()
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Goodput: flush failed: %s", e)
  _recorder = None


def record(method_name: str, *args, **kwargs) -> None:
  """Calls `recorder.<method_name>` if recording; never raises."""
  if _recorder is None:
    return
  fn = getattr(_recorder, method_name, None)
  if fn is None:
    return
  try:
    fn(*args, **kwargs)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Goodput: %s failed: %s", method_name, e)


def record_start(event: GoodputEvent) -> None:
  record(f"record_{event.value}_start_time")


def record_end(event: GoodputEvent) -> None:
  record(f"record_{event.value}_end_time")


@contextlib.contextmanager
def maybe_record(event: GoodputEvent) -> Iterator[None]:
  """Records `event` start, and its end only if the block doesn't raise."""
  record_start(event)
  yield
  record_end(event)


@contextlib.contextmanager
def maybe_record_custom_badput(event_type: str) -> Iterator[None]:
  """Records a custom badput interval around the block (end only on success)."""
  record(
      "record_custom_badput_event_start_time",
      custom_badput_event_type=event_type,
  )
  yield
  record(
      "record_custom_badput_event_end_time",
      custom_badput_event_type=event_type,
  )


def record_step_start(step: int, sync_device: bool = False) -> None:
  if _recorder is None:
    return
  if sync_device:
    _synchronize_device()
  record("record_step_start_time", step)


def _synchronize_device() -> None:
  try:
    import torch  # pylint: disable=g-import-not-at-top

    torch.accelerator.synchronize()
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Goodput: device synchronize failed: %s", e)


def record_data_loading(iterator: Iterator[Any]) -> Iterator[Any]:
  """Wraps a batch iterator, recording each blocking `next()` as data loading.

  Exceptions from the wrapped iterator (e.g. dataloader exhaustion) propagate
  unchanged, without a data loading end entry.
  """
  while True:
    record_start(GoodputEvent.DATA_LOADING)
    try:
      batch = next(iterator)
    except StopIteration:
      return
    record_end(GoodputEvent.DATA_LOADING)
    yield batch


def wrap_with_custom_badput(
    fn: Callable[..., Any],
    event_type: str,
    should_record: Optional[Callable[..., bool]] = None,
) -> Callable[..., Any]:
  """Returns `fn` wrapped in a custom badput interval.

  `should_record(*args, **kwargs)`, if given, decides per call whether the call
  does blocking work worth recording (e.g. only checkpointing steps).
  """

  def wrapper(*args, **kwargs):
    if _recorder is None or (
        should_record is not None and not should_record(*args, **kwargs)
    ):
      return fn(*args, **kwargs)
    with maybe_record_custom_badput(event_type):
      return fn(*args, **kwargs)

  return wrapper


@contextlib.contextmanager
def maybe_monitor_goodput(
    config: GoodputConfig, default_tensorboard_dir: str
) -> Iterator[None]:
  """Runs the goodput uploader(s) around the block on the lead process.

  Starts the cumulative uploader (with badput breakdown and, optionally, step
  time deviation) and, if enabled, the rolling-window uploader. On exit each is
  stopped, which does a final flush. Failures to start or stop are logged and
  swallowed; exceptions from the block propagate.
  """
  if not config.enable_goodput_monitoring or not is_lead_process():
    yield
    return
  monitor = None
  try:
    monitor = _construct_monitor(config, default_tensorboard_dir)
    if monitor is not None:
      monitor.start_goodput_uploader()
      logger.info(
          "Goodput: started cumulative goodput upload to TensorBoard and GCM"
          " for job %s.",
          _job_name,
      )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Goodput: could not start goodput monitoring: %s", e)
    monitor = None
  try:
    with _maybe_monitor_rolling_window(monitor, config):
      yield
  finally:
    if monitor is not None:
      try:
        monitor.stop_goodput_uploader()
        logger.info(
            "Goodput: flushed final metrics and stopped cumulative goodput"
            " monitoring."
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Goodput: could not stop goodput monitoring: %s", e)


def _construct_monitor(config: GoodputConfig, default_tensorboard_dir: str):
  modules = _import_goodput()
  if modules is None:
    logger.warning(
        "Goodput: ml-goodput-measurement is not installed; monitoring disabled."
    )
    return None
  _, monitoring = modules
  job_name = _job_name or resolve_job_name(config)
  gcp_options = monitoring.GCPOptions(
      enable_gcp_goodput_metrics=config.enable_gcp_goodput_metrics,
      enable_gcp_step_deviation_metrics=config.enable_gcp_step_deviation_metrics,
  )
  return monitoring.GoodputMonitor(
      job_name=job_name,
      logger_name=f"goodput_{job_name}",
      tensorboard_dir=config.tensorboard_dir or default_tensorboard_dir,
      upload_interval=config.upload_interval_seconds,
      monitoring_enabled=True,
      include_badput_breakdown=True,
      include_step_deviation=config.monitor_step_time_deviation,
      configured_ideal_step_time=config.configured_ideal_step_time,
      step_deviation_interval_seconds=config.step_deviation_interval_seconds,
      gcp_options=gcp_options,
  )


@contextlib.contextmanager
def _maybe_monitor_rolling_window(monitor, config: GoodputConfig):
  if monitor is None or not config.enable_rolling_window_goodput:
    yield
    return
  started = False
  try:
    monitor.start_rolling_window_goodput_uploader(
        list(config.rolling_windows_seconds)
    )
    started = True
    logger.info(
        "Goodput: started rolling window goodput upload to TensorBoard and"
        " GCM."
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Goodput: could not start rolling window uploader: %s", e)
  try:
    yield
  finally:
    if started:
      try:
        monitor.stop_rolling_window_goodput_uploader()
        logger.info(
            "Goodput: flushed final metrics and stopped rolling window goodput"
            " monitoring."
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Goodput: could not stop rolling window uploader: %s", e
        )
