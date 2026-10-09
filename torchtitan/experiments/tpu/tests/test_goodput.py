"""Tests for goodput recording and monitoring in the TPU graph trainer."""

import dataclasses
import os
from unittest import mock

from absl.testing import absltest
import tyro

from torchtitan.config.manager import custom_registry
from torchtitan.experiments.tpu import goodput

_GoodputConfig = goodput.GoodputConfig


class _GoodputTestCase(absltest.TestCase):
  """Runs each test as global rank 0 of a fresh, uninitialized job."""

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "8"})
    )
    self.enter_context(
        mock.patch.object(goodput.dist, "is_initialized", return_value=False)
    )
    self.goodput_lib = mock.MagicMock()
    self.monitoring_lib = mock.MagicMock()
    self.enter_context(
        mock.patch.object(
            goodput,
            "_import_goodput",
            return_value=(self.goodput_lib, self.monitoring_lib),
        )
    )
    self.addCleanup(goodput.shutdown)

  @property
  def recorder(self):
    return self.goodput_lib.GoodputRecorder.return_value

  @property
  def monitor(self):
    return self.monitoring_lib.GoodputMonitor.return_value


class InitializeTest(_GoodputTestCase):

  def test_disabled_by_default(self):
    goodput.initialize(goodput.GoodputConfig())
    self.assertFalse(goodput.enabled())
    goodput.record_start(goodput.GoodputEvent.JOB)
    self.goodput_lib.GoodputRecorder.assert_not_called()

  def test_creates_recorder_on_lead_process(self):
    goodput.initialize(
        goodput.GoodputConfig(enable_goodput_recording=True, job_name="run")
    )
    self.assertTrue(goodput.enabled())
    self.assertEqual(goodput.get_job_name(), "run")
    self.goodput_lib.GoodputRecorder.assert_called_once_with(
        job_name="run", logger_name="goodput_run", logging_enabled=True
    )

  def test_no_recorder_on_other_ranks(self):
    with mock.patch.dict(os.environ, {"RANK": "3"}):
      goodput.initialize(
          goodput.GoodputConfig(enable_goodput_recording=True, job_name="run")
      )
    self.assertFalse(goodput.enabled())
    self.goodput_lib.GoodputRecorder.assert_not_called()

  def test_no_recorder_when_multi_process_rank_is_unknown(self):
    with mock.patch.dict(os.environ, {"WORLD_SIZE": "8"}):
      del os.environ["RANK"]
      goodput.initialize(goodput.GoodputConfig(enable_goodput_recording=True))
    self.assertFalse(goodput.enabled())

  def test_default_job_name_is_timestamped(self):
    goodput.initialize(goodput.GoodputConfig(enable_goodput_recording=True))
    self.assertStartsWith(goodput.get_job_name(), "torchtitan_")

  def test_missing_library_disables_recording(self):
    with mock.patch.object(goodput, "_import_goodput", return_value=None):
      goodput.initialize(goodput.GoodputConfig(enable_goodput_recording=True))
    self.assertFalse(goodput.enabled())

  def test_recorder_construction_failure_is_swallowed(self):
    self.goodput_lib.GoodputRecorder.side_effect = RuntimeError("no creds")
    goodput.initialize(goodput.GoodputConfig(enable_goodput_recording=True))
    self.assertFalse(goodput.enabled())

  def test_verify_lead_process_drops_recorder_of_other_ranks(self):
    goodput.initialize(goodput.GoodputConfig(enable_goodput_recording=True))
    with mock.patch.object(
        goodput.dist, "is_initialized", return_value=True
    ), mock.patch.object(goodput.dist, "get_rank", return_value=2):
      goodput.verify_lead_process()
    self.assertFalse(goodput.enabled())

  def test_shutdown_flushes(self):
    goodput.initialize(goodput.GoodputConfig(enable_goodput_recording=True))
    recorder = self.recorder
    goodput.shutdown()
    recorder.flush.assert_called_once()
    self.assertFalse(goodput.enabled())


class RecordTest(_GoodputTestCase):

  def setUp(self):
    super().setUp()
    goodput.initialize(
        goodput.GoodputConfig(enable_goodput_recording=True, job_name="run")
    )

  def test_record_start_and_end(self):
    goodput.record_start(goodput.GoodputEvent.TPU_INIT)
    goodput.record_end(goodput.GoodputEvent.TPU_INIT)
    self.recorder.record_tpu_init_start_time.assert_called_once()
    self.recorder.record_tpu_init_end_time.assert_called_once()

  def test_recorder_errors_are_swallowed(self):
    self.recorder.record_job_start_time.side_effect = RuntimeError("quota")
    goodput.record_start(goodput.GoodputEvent.JOB)

  def test_maybe_record_skips_end_on_exception(self):
    with self.assertRaises(ValueError):
      with goodput.maybe_record(goodput.GoodputEvent.TRAINING_PREPARATION):
        raise ValueError()
    self.recorder.record_training_preparation_start_time.assert_called_once()
    self.recorder.record_training_preparation_end_time.assert_not_called()

  def test_record_step_start(self):
    goodput.record_step_start(7)
    self.recorder.record_step_start_time.assert_called_once_with(7)

  def test_record_step_start_can_sync_device(self):
    with mock.patch.object(goodput, "_synchronize_device") as sync:
      goodput.record_step_start(7, sync_device=True)
    sync.assert_called_once()

  def test_data_loading_wraps_each_batch(self):
    batches = list(goodput.record_data_loading(iter([1, 2])))
    self.assertEqual(batches, [1, 2])
    # One start per fetch, including the final one that hits StopIteration.
    self.assertEqual(
        self.recorder.record_data_loading_start_time.call_count, 3
    )
    self.assertEqual(self.recorder.record_data_loading_end_time.call_count, 2)

  def test_data_loading_propagates_errors(self):
    def failing():
      yield 1
      raise RuntimeError("dataloader exhausted")

    iterator = goodput.record_data_loading(failing())
    self.assertEqual(next(iterator), 1)
    with self.assertRaisesRegex(RuntimeError, "exhausted"):
      next(iterator)

  def test_custom_badput_wrapper(self):
    fn = mock.Mock(return_value="saved")
    wrapped = goodput.wrap_with_custom_badput(
        fn,
        goodput.CHECKPOINT_SAVE_BADPUT,
        should_record=lambda curr_step, last_step=False: curr_step % 5 == 0,
    )
    self.assertEqual(wrapped(3), "saved")
    self.recorder.record_custom_badput_event_start_time.assert_not_called()
    self.assertEqual(wrapped(5, last_step=True), "saved")
    fn.assert_called_with(5, last_step=True)
    self.recorder.record_custom_badput_event_start_time.assert_called_once_with(
        custom_badput_event_type="checkpoint_save"
    )
    self.recorder.record_custom_badput_event_end_time.assert_called_once_with(
        custom_badput_event_type="checkpoint_save"
    )

  def test_custom_badput_wrapper_is_passthrough_when_disabled(self):
    goodput.shutdown()
    fn = mock.Mock(return_value=1)
    wrapped = goodput.wrap_with_custom_badput(fn, goodput.EVAL_BADPUT)
    self.assertEqual(wrapped(), 1)
    self.recorder.record_custom_badput_event_start_time.assert_not_called()


class MonitorTest(_GoodputTestCase):

  def _config(self, **kwargs):
    return goodput.GoodputConfig(
        enable_goodput_recording=True,
        enable_goodput_monitoring=True,
        job_name="run",
        **kwargs,
    )

  def test_disabled_monitoring_does_nothing(self):
    with goodput.maybe_monitor_goodput(goodput.GoodputConfig(), "/tmp/tb"):
      pass
    self.monitoring_lib.GoodputMonitor.assert_not_called()

  def test_starts_and_stops_cumulative_uploader(self):
    config = self._config(tensorboard_dir="gs://bucket/tb")
    goodput.initialize(config)
    with goodput.maybe_monitor_goodput(config, "/tmp/tb"):
      self.monitor.start_goodput_uploader.assert_called_once()
      self.monitor.stop_goodput_uploader.assert_not_called()
    self.monitor.stop_goodput_uploader.assert_called_once()
    self.monitor.start_rolling_window_goodput_uploader.assert_not_called()
    kwargs = self.monitoring_lib.GoodputMonitor.call_args.kwargs
    self.assertEqual(kwargs["job_name"], "run")
    self.assertEqual(kwargs["logger_name"], "goodput_run")
    self.assertEqual(kwargs["tensorboard_dir"], "gs://bucket/tb")
    self.assertTrue(kwargs["include_badput_breakdown"])

  def test_default_tensorboard_dir(self):
    config = self._config()
    goodput.initialize(config)
    with goodput.maybe_monitor_goodput(config, "/tmp/tb"):
      pass
    kwargs = self.monitoring_lib.GoodputMonitor.call_args.kwargs
    self.assertEqual(kwargs["tensorboard_dir"], "/tmp/tb")

  def test_rolling_window_uploader(self):
    config = self._config(
        enable_rolling_window_goodput=True, rolling_windows_seconds=(60, 120)
    )
    goodput.initialize(config)
    with goodput.maybe_monitor_goodput(config, "/tmp/tb"):
      self.monitor.start_rolling_window_goodput_uploader.assert_called_once_with(
          [60, 120]
      )
    self.monitor.stop_rolling_window_goodput_uploader.assert_called_once()
    self.monitor.stop_goodput_uploader.assert_called_once()

  def test_stops_uploader_when_training_fails(self):
    config = self._config()
    goodput.initialize(config)
    with self.assertRaises(ValueError):
      with goodput.maybe_monitor_goodput(config, "/tmp/tb"):
        raise ValueError()
    self.monitor.stop_goodput_uploader.assert_called_once()

  def test_start_failure_does_not_interrupt_training(self):
    self.monitor.start_goodput_uploader.side_effect = RuntimeError("no creds")
    config = self._config()
    goodput.initialize(config)
    ran = False
    with goodput.maybe_monitor_goodput(config, "/tmp/tb"):
      ran = True
    self.assertTrue(ran)
    self.monitor.stop_goodput_uploader.assert_not_called()

  def test_no_monitor_on_other_ranks(self):
    with mock.patch.dict(os.environ, {"RANK": "1"}):
      with goodput.maybe_monitor_goodput(self._config(), "/tmp/tb"):
        pass
    self.monitoring_lib.GoodputMonitor.assert_not_called()


class CommandLineTest(absltest.TestCase):

  @dataclasses.dataclass
  class _Config:
    goodput: _GoodputConfig = dataclasses.field(default_factory=_GoodputConfig)

  def test_flags(self):
    config = tyro.cli(
        self._Config,
        args=[
            "--goodput.enable-goodput-recording",
            "--goodput.enable-goodput-monitoring",
            "--goodput.job-name=my_run",
            "--goodput.tensorboard-dir=gs://bucket/tb",
            "--goodput.enable-rolling-window-goodput",
            "--goodput.rolling-windows-seconds",
            "60",
            "300",
        ],
        registry=custom_registry,
    )
    self.assertTrue(config.goodput.enable_goodput_recording)
    self.assertTrue(config.goodput.enable_goodput_monitoring)
    self.assertEqual(config.goodput.job_name, "my_run")
    self.assertEqual(config.goodput.tensorboard_dir, "gs://bucket/tb")
    self.assertTrue(config.goodput.enable_rolling_window_goodput)
    self.assertEqual(config.goodput.rolling_windows_seconds, (60, 300))

  def test_goodput_is_part_of_tpu_trainer_config(self):
    from torchtitan.experiments.tpu import tpu_job_config  # pylint: disable=g-import-not-at-top

    fields = {f.name for f in dataclasses.fields(tpu_job_config.TPUTrainerConfig)}
    self.assertIn("goodput", fields)


if __name__ == "__main__":
  absltest.main()
