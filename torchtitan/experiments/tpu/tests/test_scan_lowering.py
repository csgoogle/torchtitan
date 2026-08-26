# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device tests for how scan-over-layers lowers through torch_tpu to StableHLO.

The CPU suite (`test_scan_over_layers.py`) proves the FX graph holds one copy of
the layer body. This suite proves that survives the torch_tpu export, which is
the claim that actually matters: the layer stack must become exactly one
`stablehlo.while` whose body does not grow with depth. A regression here is
invisible in the FX graph and only shows up as compile time and HBM on a real
run.

Requires a TPU:

    python -m torchtitan.experiments.tpu.tests.test_scan_lowering
"""

import collections
import re
import subprocess
import sys

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
import torch_tpu  # pylint: disable=unused-import  (registers the `tpu` device)
from torch_tpu._internal.export.export import fx_to_mlir
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.experiments.tpu.scan import layer_stack as scan_model


_DEVICE = "tpu:0"


class _TpuOnly:
  """Skips a `BaseAcceleratorDeviceTest` unless it is running on a TPU.

  A mixin rather than a base class so the test loader does not collect it as a
  test case of its own, and so `BaseAcceleratorDeviceTest` stays a literal base
  of every class below: `.ci/discover_tpu_tests.py` classifies by base-class
  name, and that name is what routes this module to the single-TPU shard. A
  plain `parameterized.TestCase` would be scheduled on a CPU runner instead,
  where every test here fails for want of a device. The skip keeps
  `--device=cpu` honest rather than red.
  """

  def setUp(self):
    super().setUp()
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("asserts on the torch_tpu lowering; TPU only")


class _ToyBlock(nn.Module):
  """Decoder-block stand-in with the real call signature."""

  def __init__(self, dim: int):
    super().__init__()
    self.wi = nn.Linear(dim, dim, bias=False)
    self.wo = nn.Linear(dim, dim, bias=False)

  def forward(self, h, attention_masks=None, positions=None):
    del attention_masks, positions
    return h + self.wo(torch.tanh(self.wi(h)))


class _ToyModel(nn.Module):

  def __init__(self, n_layers: int, dim: int):
    super().__init__()
    self.layers = nn.ModuleDict(
        {str(i): _ToyBlock(dim) for i in range(n_layers)}
    )


def _lifted_tensors(graph: torch.fx.GraphModule) -> list[torch.Tensor]:
  """Every device buffer the traced graph consumes as a constant.

  `fx_to_mlir` needs the full argument set, not just the activation: make_fx
  leaves the stacked layer parameters as module attributes and torch_tpu
  resolves each to a device buffer it expects to be handed in. Both failure
  modes -- a missing buffer and the same buffer passed twice -- abort the
  process instead of raising, so dedupe on `data_ptr` and walk nested
  GraphModules (the scan HOP's body is one).
  """
  lifted, seen = [], set()
  for _, sub in graph.named_modules():
    for node in getattr(getattr(sub, "graph", None), "nodes", []):
      if node.op != "get_attr":
        continue
      tensor = getattr(sub, node.target, None)
      if isinstance(tensor, torch.Tensor) and tensor.data_ptr() not in seen:
        seen.add(tensor.data_ptr())
        lifted.append(tensor)
  return lifted


def _lower_to_mlir(n_layers: int, dim: int = 8) -> str:
  """Traces a scanned stack and returns its StableHLO assembly."""
  torch.manual_seed(0)
  model = _ToyModel(n_layers, dim)
  scan_model.apply_scan_over_layers(model)
  model.eval()
  stack = model.layers.to(_DEVICE)

  h = torch.randn(2, 3, dim, device=_DEVICE)
  graph = make_fx(stack, tracing_mode="real")(h)
  return fx_to_mlir(graph, [h, *_lifted_tensors(graph)]).serialize_text()


def _op_histogram(mlir: str) -> collections.Counter:
  return collections.Counter(re.findall(r"\b([a-z_]+\.[a-z_0-9]+)\b", mlir))


class ScanLoweringTest(_TpuOnly, BaseAcceleratorDeviceTest):
  """One `stablehlo.while`, and a body that does not scale with depth."""

  def test_lowers_to_exactly_one_while_loop(self):
    self.assertEqual(
        _op_histogram(_lower_to_mlir(4))["stablehlo.while"],
        1,
        "a scanned layer stack must lower to exactly one stablehlo.while",
    )

  def test_body_holds_a_single_copy_of_the_layer(self):
    """Two matmuls and one tanh for a 4-layer stack -- one layer's worth."""
    histogram = _op_histogram(_lower_to_mlir(4))
    self.assertEqual(histogram["stablehlo.dot_general"], 2)
    self.assertEqual(histogram["stablehlo.tanh"], 1)

  def test_lowered_module_is_independent_of_depth(self):
    """The compile-time win, measured at the IR the compiler actually sees.

    An unrolled stack would grow every op count with depth. Asserting the whole
    histogram rather than one op means any partial unroll shows up, not just a
    matmul one.
    """
    self.assertEqual(
        _op_histogram(_lower_to_mlir(4)),
        _op_histogram(_lower_to_mlir(16)),
        "the lowered module changed when depth went 4 -> 16; the layer body is"
        " being unrolled, not scanned",
    )


def _lower_with_short_argument_list():
  """Calls `fx_to_mlir` with an argument list the traced graph outgrows."""
  torch.manual_seed(0)
  model = _ToyModel(4, 8)
  scan_model.apply_scan_over_layers(model)
  model.eval()
  stack = model.layers.to(_DEVICE)
  h = torch.randn(2, 3, 8, device=_DEVICE)
  graph = make_fx(stack, tracing_mode="real")(h)

  # `h` alone: the stacked layer parameters the graph also consumes are
  # deliberately withheld.
  fx_to_mlir(graph, [h])


_RAISED = 0
_ACCEPTED = 1
_UNDETERMINED = 2

# `__spec__.name`, not `__name__`: under `python -m` the latter is `__main__`,
# which would not resolve in the probe's fresh interpreter.
_PROBE = f"""
from {__spec__.name} import _lower_with_short_argument_list
try:
  _lower_with_short_argument_list()
except RuntimeError as e:
  # Match the message, not just the type. A probe that never reaches the
  # validation raises RuntimeError too -- "pjrt client not initialized" when
  # another process holds the TPU, most often -- and scoring that as success
  # would report that a wheel validates when it actually aborts.
  raise SystemExit({_RAISED} if "was not provided" in str(e) else {_UNDETERMINED})
except BaseException:
  raise SystemExit({_UNDETERMINED})
raise SystemExit({_ACCEPTED})
"""

# Set by `setUpModule`; None until then, and on non-TPU runs.
_probe_result: int | None = None


def setUpModule():
  """Probes how this torch_tpu reports a bad argument list, before any test runs.

  Two constraints put this here rather than in the test that reads it. The
  probe has to be a subprocess, because the behaviour it probes for is
  `LOG(FATAL)` and an in-process probe against an older wheel would take the
  whole module down with it. And that subprocess needs a TPU of its own: once
  any test has touched the device this process owns it, and the probe can then
  only fail with "The TPU is already in use" -- an outcome that says nothing
  about argument validation. `setUpModule` runs after flag parsing and before
  the first test, which is the last moment the device is still free.
  """
  global _probe_result
  if test_utils.resolve_device_type() != device_type.AcceleratorDeviceType.TPU:
    return
  _probe_result = subprocess.run(
      [sys.executable, "-c", _PROBE], capture_output=True, check=False
  ).returncode


class ArgumentValidationTest(_TpuOnly, BaseAcceleratorDeviceTest):
  """A bad argument list must raise, not abort the process.

  `fx_to_mlir`'s `args` is caller-supplied, so a mismatch against what the
  traced graph consumes is ordinary caller error. torch_tpu used to validate it
  with `ABSL_CHECK_OK`, which turned an `INVALID_ARGUMENT` status into
  `LOG(FATAL)` -- SIGABRT, no Python exception, no traceback. Every masked
  compile failure on this lane looked identical from Python because of it.

  Skipped against a torch_tpu that still aborts, since there the assertion
  cannot be made from inside the process it kills. The call itself is never
  made in this process for the same reason -- `setUpModule` makes it in a
  subprocess and this test reads the outcome.
  """

  def test_short_argument_list_raises(self):
    if _probe_result is None or _probe_result == _UNDETERMINED:
      self.skipTest("the probe never reached fx_to_mlir's argument validation")
    if _probe_result < 0:
      self.skipTest(
          "this torch_tpu aborts on a bad fx_to_mlir argument list instead of"
          " returning the status; nothing to assert from in-process"
      )
    self.assertEqual(
        _probe_result,
        _RAISED,
        "fx_to_mlir accepted an argument list the traced graph outgrows rather"
        " than reporting it",
    )


class ScanExecutionTest(_TpuOnly, BaseAcceleratorDeviceTest):
  """The lowered loop must also run, and agree with the unrolled loop.

  `peel_first_layer` is covered here rather than in the lowering tests above:
  its purpose is keeping a Pallas kernel's `func.call` callee resolvable once
  the body is cloned into the loop region, and a toy block has no Pallas kernel
  to lose. Numerics are what a toy model can honestly check.
  """

  @parameterized.named_parameters(("plain", False), ("peel_first_layer", True))
  def test_device_forward_matches_device_unrolled(self, peel_first_layer):
    """Both sides run on device, so the only variable is scan.

    Comparing against a CPU fp32 loop instead would fail on TPU matmul
    precision alone (~1.5e-2 relative over four layers), which says nothing
    about scan. Exact fp32 equivalence is already covered on CPU by
    `test_scan_over_layers.ScanSemanticsTest`.
    """
    torch.manual_seed(0)
    n_layers, dim = 4, 8
    model = _ToyModel(n_layers, dim).to(_DEVICE)

    h = torch.randn(2, 3, dim, device=_DEVICE)
    want = h
    for i in range(n_layers):
      want = model.layers[str(i)](want)
    want = want.cpu()

    scan_model.apply_scan_over_layers(model, peel_first_layer=peel_first_layer)
    # After apply, not before: it installs a fresh ScannedLayerStack, which
    # defaults to train mode and then rejects a carry that does not require
    # grad.
    model.eval()
    got = model.layers(h).cpu()

    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  absltest.main()
