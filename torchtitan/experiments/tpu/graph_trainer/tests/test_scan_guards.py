# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Each guard must raise, and must say which knob to change.

The message is part of the contract, not decoration: every one of these
conditions otherwise produces a *successful* unrolled run under a scan-shaped
command line, so a reader who gets a bare `ValueError` learns nothing they
could not have learned from the exit code.
"""

import unittest
from unittest import mock

import torch
from torch import nn

from torchtitan.experiments.tpu.graph_trainer.layer_stack import (
    LayerStack,
    STACK_ATTR,
    StackedLayers,
)
from torchtitan.experiments.tpu.graph_trainer.scan_guards import (
    check_layers_are_stackable,
    check_scan_parallelism,
    check_scan_was_applied,
)


def _dims(tp=1, cp=1, pp=1):
    return mock.Mock(tp=tp, cp=cp, pp=pp)


def _layer(dim=4, extra=None, out=None, cache=None):
    layer = nn.Module()
    layer.wq = nn.Linear(dim, out if out is not None else dim, bias=False)
    if extra is not None:
        layer.wextra = nn.Linear(dim, extra, bias=False)
    if cache is not None:
        layer.register_buffer("cache", torch.as_tensor(cache))
    return layer


def _layers(*mods):
    return nn.ModuleDict({str(i): m for i, m in enumerate(mods)})


class ParallelismTest(unittest.TestCase):

    def test_plain_data_parallel_is_allowed(self):
        check_scan_parallelism(_dims())

    def test_each_degree_names_its_own_flag(self):
        for arg, kind, degree in (("tp", "tensor", 2), ("cp", "context", 4),
                                  ("pp", "pipeline", 2)):
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError) as caught:
                    check_scan_parallelism(_dims(**{arg: degree}))
                self.assertIn(
                    f"--parallelism.{kind}-parallel-degree={degree}",
                    str(caught.exception),
                )

    def test_every_conflicting_degree_is_listed_not_just_the_first(self):
        # Fixing them one run at a time, learning about the next only after
        # waiting for another launch, is the failure this avoids.
        with self.assertRaises(ValueError) as caught:
            check_scan_parallelism(_dims(tp=2, cp=2, pp=2))
        message = str(caught.exception)
        for flag in ("tensor-parallel", "context-parallel", "pipeline-parallel"):
            self.assertIn(flag, message)


class StackableTest(unittest.TestCase):

    def test_identical_layers_are_stackable(self):
        check_layers_are_stackable(_layers(_layer(), _layer(), _layer()))
        check_layers_are_stackable(_layers(_layer(cache=[1.0]), _layer(cache=[1.0])))

    def test_a_differing_parameter_set_names_the_first_layer_and_the_parameter(self):
        # Both directions -- extra and missing -- and the *first* difference,
        # not the last: fixing them one launch at a time is the cost here.
        for label, layers in (
            ("extra", _layers(_layer(), _layer(extra=8), _layer(extra=16))),
            ("missing", _layers(_layer(extra=8), _layer(), _layer())),
        ):
            with self.subTest(label):
                with self.assertRaises(ValueError) as caught:
                    check_layers_are_stackable(layers)
                self.assertIn("layer 1", str(caught.exception))
                self.assertIn("wextra.weight", str(caught.exception))

    def test_a_same_named_parameter_shows_what_differs(self):
        # A shape a set comparison would miss entirely, and -- the quietest case
        # of all -- a dtype: the stack is allocated from layer 0, so a differing
        # dtype is cast into it and the run reports nothing.
        odd_dtype = _layer()
        odd_dtype.wq.to(torch.float64)
        for expected, layers in (
            (["(4, 4)", "(6, 4)"], _layers(_layer(), _layer(out=6))),
            (["float64"], _layers(_layer(), odd_dtype)),
        ):
            with self.subTest(expected):
                with self.assertRaises(ValueError) as caught:
                    check_layers_are_stackable(layers)
                message = str(caught.exception)
                self.assertIn("wq.weight", message)
                for fragment in expected:
                    self.assertIn(fragment, message)

    def test_a_differing_buffer_is_rejected(self):
        # Differing values are the quiet case: same name, same shape, same
        # dtype, nothing structural to notice, and the body would run layer 1
        # with layer 0's 1.0 and report nothing.
        for label, layers in (
            ("values", _layers(_layer(cache=[1.0]), _layer(cache=[2.0]))),
            ("absent", _layers(_layer(cache=[1.0]), _layer())),
        ):
            with self.subTest(label):
                with self.assertRaises(ValueError) as caught:
                    check_layers_are_stackable(layers)
                message = str(caught.exception)
                self.assertIn("layer 1", message)
                self.assertIn("cache", message)

    def test_meta_buffers_are_compared_by_shape_only(self):
        # Before materialization there are no values to compare, and refusing
        # to run at that point would reject every real model.
        with torch.device("meta"):
            check_layers_are_stackable(_layers(_layer(cache=[1.0]), _layer(cache=[2.0])))


class WasAppliedTest(unittest.TestCase):

    def _model(self, *, scan: bool | None):
        """``None`` for plain layers, otherwise stacked with that ``use_scan``."""
        model = nn.Module()
        layers = _layers(_layer(), _layer())
        setattr(model, STACK_ATTR, nn.Module())
        model.layers = (
            layers
            if scan is None
            else StackedLayers(
                layers, LayerStack(layers, getattr(model, STACK_ATTR)), use_scan=scan
            )
        )
        return model

    def test_a_run_that_did_not_ask_for_scan_is_not_checked(self):
        check_scan_was_applied(self._model(scan=None), use_scan_layers=False)

    def test_a_scan_run_that_got_a_scan_passes(self):
        check_scan_was_applied(self._model(scan=True), use_scan_layers=True)

    def test_a_scan_run_that_got_nothing_raises(self):
        # The whole point: this model trains perfectly well. It is only wrong
        # because the number it will report is not the number that was asked for.
        with self.assertRaises(ValueError) as caught:
            check_scan_was_applied(self._model(scan=None), use_scan_layers=True)
        self.assertIn("use-scan-layers", str(caught.exception))
        self.assertIn("one at a time", str(caught.exception))

    def test_stacked_but_not_scanned_still_raises(self):
        # `--tpu-config.use-stacked-layers` alone reaches this same container, so the
        # stacks existing is not evidence that the scan flag was honoured.
        with self.assertRaises(ValueError):
            check_scan_was_applied(self._model(scan=False), use_scan_layers=True)

    def test_the_container_is_found_below_the_root(self):
        # Lanes are free to hang the decoder off a submodule; the guard must not
        # depend on where, or it becomes a second thing to keep in sync.
        model = nn.Module()
        model.inner = self._model(scan=True)
        check_scan_was_applied(model, use_scan_layers=True)


if __name__ == "__main__":
    unittest.main()
