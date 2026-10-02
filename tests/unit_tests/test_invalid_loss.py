# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.trainer import Trainer


class _TrainStepTestBase(unittest.TestCase):
    """Helpers to drive Trainer.train_step on a single rank with mocks."""

    def _make_trainer(self, loss_value: float, should_log: bool) -> Trainer:
        # Build a bare Trainer and inject only the collaborators train_step
        # touches on the non-distributed (single-rank) path.
        trainer = object.__new__(Trainer)

        trainer.optimizers = MagicMock()
        trainer.lr_schedulers = MagicMock()
        trainer.lr_schedulers.get_metrics.return_value = {}
        trainer.checkpointer = MagicMock()
        trainer.model_parts = []
        trainer.config = MagicMock()
        trainer.config.training.max_norm = 1.0
        trainer.device = torch.device("cpu")
        trainer.gradient_accumulation_steps = 1
        trainer.step = 1
        trainer.ntokens_seen = 0

        parallel_dims = MagicMock()
        parallel_dims.dp_enabled = False
        parallel_dims.dp_cp_enabled = False
        parallel_dims.ep_enabled = False
        parallel_dims.get_optional_mesh.return_value = None
        trainer.parallel_dims = parallel_dims

        trainer.metrics_processor = MagicMock()
        trainer.metrics_processor.should_log.return_value = should_log
        trainer.metrics_processor.config.defer_readback = False

        # Shadow the bound method so forward/backward returns a canned loss.
        trainer.forward_backward_step = MagicMock(return_value=torch.tensor(loss_value))
        return trainer

    def _data_iterator(self):
        labels = torch.tensor([1, 2, IGNORE_INDEX])
        input_dict = {"input": torch.tensor([1, 2, 3])}
        while True:
            yield input_dict, labels


class TestInvalidLoss(_TrainStepTestBase):
    """Trainer.train_step crashes on a non-finite loss, aligned with logging.

    The finiteness check reuses the device-to-host copy that logging already
    performs, so it only runs on steps where metrics are logged.
    """

    def _run_step(self, loss_value: float, should_log: bool) -> None:
        trainer = self._make_trainer(loss_value, should_log)
        # sl.* are logging side effects; clip_grad_norm_ needs real params.
        with patch("torchtitan.trainer.sl", MagicMock()), patch(
            "torchtitan.trainer.dist_utils.clip_grad_norm_",
            return_value=torch.tensor(1.0),
        ):
            trainer.train_step(self._data_iterator())

    def test_nan_loss_raises_on_log_step(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run_step(float("nan"), should_log=True)
        self.assertIn("not finite", str(ctx.exception))

    def test_inf_loss_raises_on_log_step(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run_step(float("inf"), should_log=True)
        self.assertIn("not finite", str(ctx.exception))

    def test_finite_loss_does_not_raise(self):
        self._run_step(1.5, should_log=True)

    def test_nan_loss_ignored_when_not_logging(self):
        # Detection is gated on logging, so a non-log step must not raise even
        # when the loss is NaN.
        self._run_step(float("nan"), should_log=False)

    def test_global_valid_tokens_passed_as_tensor_and_metrics_logged(self):
        trainer = self._make_trainer(1.5, should_log=True)
        trainer.config.compile.precompile_artifact_dir = None
        trainer.ntokens_seen = 7
        with patch("torchtitan.trainer.sl", MagicMock()), patch(
            "torchtitan.trainer.dist_utils.clip_grad_norm_",
            return_value=torch.tensor(0.25),
        ):
            trainer.train_step(self._data_iterator())

        kwargs = trainer.forward_backward_step.call_args.kwargs
        gvt = kwargs["global_valid_tokens"]
        self.assertIsInstance(gvt, torch.Tensor)
        self.assertEqual(gvt.dtype, torch.float32)
        self.assertEqual(gvt.item(), 2.0)  # two non-ignored labels

        args, log_kwargs = trainer.metrics_processor.log.call_args
        self.assertEqual(args[0], 1)
        self.assertAlmostEqual(args[1], 1.5)
        self.assertAlmostEqual(args[2], 1.5)
        self.assertAlmostEqual(args[3], 0.25)
        self.assertEqual(log_kwargs["extra_metrics"]["n_tokens_seen"], 7)

    def test_precompile_artifact_passes_float(self):
        trainer = self._make_trainer(1.5, should_log=False)
        trainer.config.compile.precompile_artifact_dir = "/tmp/artifact"
        with patch("torchtitan.trainer.sl", MagicMock()), patch(
            "torchtitan.trainer.dist_utils.clip_grad_norm_",
            return_value=torch.tensor(1.0),
        ):
            trainer.train_step(self._data_iterator())
        gvt = trainer.forward_backward_step.call_args.kwargs["global_valid_tokens"]
        self.assertIsInstance(gvt, float)
        self.assertEqual(gvt, 2.0)


class TestDeferredMetricsReadback(_TrainStepTestBase):
    """With `metrics.defer_readback`, step N's metrics are logged in step N+1."""

    def _make_deferred_trainer(
        self, losses: list[float], should_log: bool
    ) -> Trainer:
        trainer = self._make_trainer(losses[0], should_log)
        trainer.config.compile.precompile_artifact_dir = None
        trainer.metrics_processor.config.defer_readback = True
        trainer.forward_backward_step = MagicMock(
            side_effect=[torch.tensor(v) for v in losses]
        )
        return trainer

    def _run_steps(self, trainer: Trainer, num_steps: int) -> None:
        data_iterator = self._data_iterator()
        with patch("torchtitan.trainer.sl", MagicMock()), patch(
            "torchtitan.trainer.dist_utils.clip_grad_norm_",
            return_value=torch.tensor(0.25),
        ):
            for _ in range(num_steps):
                trainer.train_step(data_iterator)
                trainer.step += 1

    def test_metrics_logged_one_step_late(self):
        trainer = self._make_deferred_trainer([1.5, 2.5], should_log=True)
        trainer.ntokens_seen = 7

        self._run_steps(trainer, 1)
        trainer.metrics_processor.log.assert_not_called()
        self.assertIsNotNone(trainer._pending_metrics)

        self._run_steps(trainer, 1)
        args, log_kwargs = trainer.metrics_processor.log.call_args
        self.assertEqual(args[0], 1)
        self.assertAlmostEqual(args[1], 1.5)
        self.assertAlmostEqual(args[3], 0.25)
        self.assertEqual(log_kwargs["extra_metrics"]["n_tokens_seen"], 7)
        self.assertEqual(trainer._pending_metrics.step, 2)

        # The last pending step is logged by an explicit flush.
        trainer._flush_pending_metrics()
        args, _ = trainer.metrics_processor.log.call_args
        self.assertEqual(args[0], 2)
        self.assertAlmostEqual(args[1], 2.5)
        self.assertIsNone(trainer._pending_metrics)

    def test_nan_loss_raises_on_next_step(self):
        trainer = self._make_deferred_trainer([float("nan"), 1.5], should_log=True)
        self._run_steps(trainer, 1)
        with self.assertRaises(RuntimeError) as ctx:
            self._run_steps(trainer, 1)
        self.assertIn("not finite", str(ctx.exception))
        self.assertIn("step 1", str(ctx.exception))

    def test_pending_step_flushed_on_non_log_step(self):
        trainer = self._make_deferred_trainer([1.5, 2.5], should_log=True)
        self._run_steps(trainer, 1)
        trainer.metrics_processor.should_log.return_value = False
        self._run_steps(trainer, 1)
        args, _ = trainer.metrics_processor.log.call_args
        self.assertEqual(args[0], 1)
        self.assertIsNone(trainer._pending_metrics)

    def test_throughput_counters_attributed_to_logged_step(self):
        trainer = self._make_deferred_trainer([1.5, 2.5], should_log=True)
        processor = trainer.metrics_processor
        processor.ntokens_since_last_log = 0
        processor.data_loading_times = []
        counters_at_log = []

        def record_and_reset(step, *args, **kwargs):
            # Mirrors MetricsProcessor.log, which consumes and resets these.
            counters_at_log.append(
                (step, processor.ntokens_since_last_log,
                 list(processor.data_loading_times))
            )
            processor.ntokens_since_last_log = 0
            processor.data_loading_times.clear()

        processor.log.side_effect = record_and_reset
        data_iterator = self._data_iterator()
        with patch("torchtitan.trainer.sl", MagicMock()), patch(
            "torchtitan.trainer.dist_utils.clip_grad_norm_",
            return_value=torch.tensor(0.25),
        ):
            for ntokens, load_time in [(10, 0.1), (20, 0.2)]:
                # Stand-in for the batch generator's per-batch accounting.
                processor.ntokens_since_last_log += ntokens
                processor.data_loading_times.append(load_time)
                trainer.train_step(data_iterator)
                trainer.step += 1
        # Step 2's batch was loaded before step 1 was logged; it must count
        # toward step 2. The final flush then sees no new batches at all.
        trainer._flush_pending_metrics()

        self.assertEqual(counters_at_log, [(1, 10, [0.1]), (2, 20, [0.2])])
        self.assertEqual(processor.ntokens_since_last_log, 0)
        self.assertEqual(processor.data_loading_times, [])


if __name__ == "__main__":
    unittest.main()
