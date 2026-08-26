# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Does the scan loop survive XLA, not just torch_tpu?

`test_scan_lowering.py` asserts one `stablehlo.while` in the module torch_tpu
*emits*. That is the input to the compiler, not its output. XLA is free to
unroll a `while` -- and a `while` it unrolls costs exactly what no scan at all
costs: compile time linear in depth, and every layer's activations live at once.
Between those two IRs sit the passes that decide whether this lane's headline
claim is true, and nothing asserted anything about them.

Until now the only check on the compiled side ran at 70B on four slices, as a
grep for ` while(` in the payload's HLO dump. That is the right assertion at the
wrong scale: it needs a cluster, half an hour, and a reviewer reading a banner.
This module makes the same assertion on one chip in seconds.

The mechanism is XLA's own dump. `XLA_FLAGS=--xla_dump_to=...` has to be set
before PJRT initialises, so each compile runs in a subprocess with the flag in
its environment and the parent reads the files back.

Requires a TPU:

    python -m torchtitan.experiments.tpu.tests.test_scan_compiled_hlo
"""

import glob
import os
import re
import subprocess
import sys
import tempfile

from absl.testing import absltest
import torch
from torch import nn
import torch_tpu  # pylint: disable=unused-import  (registers the `tpu` device)
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu.scan import layer_stack as scan_model
# Imported as a module, not `from ... import BaseAcceleratorDeviceTest`. Binding
# the class into this namespace makes unittest collect the base class itself as
# a test of this run, and its inherited `test_gpu_available` then executes
# BEFORE `setUpModule` -- claiming the chip that `setUpModule`'s subprocesses
# need. `.ci/discover_tpu_tests.py` reads `ast.Attribute.attr`, so the dotted
# form still routes this module to the single-TPU shard.


_DEVICE = "tpu:0"

# The two depths the depth-invariance assertions compare. 4x apart, so a stack
# that unrolls has to grow visibly or not be unrolling.
_SHALLOW = 4
_DEEP = 16

# `while(` with the paren is the HLO call form, and the paren is load-bearing:
# `while` also appears in metadata strings and in pass names such as
# `while-loop-simplifier`, which would make a bare word match report a loop that
# a pass had just deleted.
_WHILE_RE = re.compile(r"\bwhile\(")


class _TpuOnly:
  """Skips unless running on a TPU.

  A mixin, so `BaseAcceleratorDeviceTest` stays a base of every class below --
  `.ci/discover_tpu_tests.py` classifies by base-class name and that is what
  routes this module to the TPU shard rather than a CPU runner. Same reasoning
  as `test_scan_lowering._TpuOnly`.
  """

  def setUp(self):
    super().setUp()
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("asserts on compiled XLA HLO; TPU only")


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


def compile_scanned_stack(n_layers: int, dim: int = 8, scan: bool = True):
  """Runs a layer stack on device so XLA compiles it, and dumps the HLO.

  Module-level and JSON-argument-only because it is invoked by name in a
  subprocess. The `.cpu()` at the end is what forces execution: without it the
  lazy graph is never compiled and the dump directory comes back empty, which
  would make every assertion below vacuously pass.
  """
  torch.manual_seed(0)
  model = _ToyModel(n_layers, dim).to(_DEVICE)
  if scan:
    scan_model.apply_scan_over_layers(model)
    model.eval()
  h = torch.randn(2, 3, dim, device=_DEVICE)
  out = model.layers(h) if scan else _run_unrolled(model, h, n_layers)
  return float(out.detach().cpu().sum())


def _run_unrolled(model, h, n_layers):
  for i in range(n_layers):
    h = model.layers[str(i)](h)
  return h


class _Compiled:
  """The `after_optimizations` modules from one compile, and what they contain.

  `loop_text_lines` counts only the modules that actually hold a loop. A compile
  emits a dozen unrelated modules -- RNG, dtype casts, the `.cpu()` transfer --
  whose sizes drift with anything and would swamp a whole-dump line count. The
  loop-bearing module is the one whose growth answers "was the body unrolled".
  """

  def __init__(self, texts: list[str]):
    self.while_count = sum(len(_WHILE_RE.findall(t)) for t in texts)
    self.module_count = len(texts)
    self.loop_text_lines = sum(
        t.count("\n") for t in texts if _WHILE_RE.search(t)
    )


# Filled by `setUpModule`, keyed by (n_layers, scan). Empty on non-TPU runs, and
# on a TPU run every value is present or `setUpModule` raised.
_COMPILED: dict[tuple[int, bool], _Compiled] = {}

_CASES = ((_SHALLOW, True), (_DEEP, True), (_SHALLOW, False))


def setUpModule():
  """Compiles every case up front, while the TPU is still unclaimed.

  Each compile needs a subprocess: `XLA_FLAGS=--xla_dump_to` is read when PJRT
  initialises and cannot be set afterwards. That subprocess needs a TPU of its
  own, and once any test in this process has touched the device this process
  owns it -- the child then fails with "The TPU is already in use", which was
  the original form of this module and it failed exactly that way.
  `setUpModule` runs after flag parsing and before the first test, which is the
  last moment the device is free. Same constraint, and same resolution, as
  `test_scan_lowering.setUpModule`.
  """
  if test_utils.resolve_device_type() != device_type.AcceleratorDeviceType.TPU:
    return
  for n_layers, scan in _CASES:
    _COMPILED[(n_layers, scan)] = _compile_and_read_hlo(n_layers, scan)


def _compile_and_read_hlo(n_layers: int, scan: bool) -> _Compiled:
  """Compiles in a subprocess with XLA's dump on; returns what it emitted.

  Fails loudly on an empty dump rather than returning nothing. An empty result
  would let a `while`-count assertion of 0 == 0 pass while proving nothing,
  which is the exact shape of the silent no-op this module exists to catch.
  """
  with tempfile.TemporaryDirectory() as dump_dir:
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        f"--xla_dump_to={dump_dir} --xla_dump_hlo_as_text"
        " --xla_dump_hlo_pass_re=NOTHING"
    )
    script = (
        f"import sys, importlib\n"
        f"sys.path = {sys.path!r}\n"
        f"m = importlib.import_module({__spec__.name!r})\n"
        f"m.compile_scanned_stack(n_layers={n_layers}, scan={scan!r})\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if done.returncode != 0:
      # Head as well as tail: the cause of a failed device acquisition is the
      # first PJRT line, and the traceback that ends the log is a downstream
      # "pjrt client not initialized" that names no reason.
      raise AssertionError(
          f"the n_layers={n_layers} scan={scan} compile exited"
          f" {done.returncode}\n--- stderr head ---\n{done.stderr[:3000]}"
          f"\n--- stderr tail ---\n{done.stderr[-2000:]}"
      )
    paths = sorted(glob.glob(os.path.join(dump_dir, "*after_optimizations.txt")))
    if not paths:
      raise AssertionError(
          f"XLA dumped no after_optimizations module into {dump_dir}; the"
          " compile never happened, so nothing here can be asserted"
      )
    return _Compiled([open(p).read() for p in paths])


class CompiledHloWhileTest(
    _TpuOnly, base_device_test.BaseAcceleratorDeviceTest
):
  """The loop XLA emits, not the loop torch_tpu asked for."""

  def test_compiled_hlo_retains_a_while_loop(self):
    """The assertion the 70B payload makes, at one-chip cost."""
    self.assertGreaterEqual(
        _COMPILED[(_SHALLOW, True)].while_count,
        1,
        "the scanned stack compiled to HLO with no while loop: XLA unrolled"
        " the scan, so the compile-time and HBM wins are both gone",
    )

  def test_unrolled_stack_has_no_while_loop(self):
    """The negative control.

    Without it `test_compiled_hlo_retains_a_while_loop` cannot distinguish "scan
    produced a loop" from "this toy model produces a loop no matter what", and a
    passing suite would say nothing about scan.
    """
    self.assertEqual(
        _COMPILED[(_SHALLOW, False)].while_count,
        0,
        "the UNSCANNED stack compiled to a while loop, so the positive test"
        " above is not evidence that scan did anything",
    )

  def test_compiled_while_count_is_independent_of_depth(self):
    """One loop over N layers, not one loop plus N-k peeled copies.

    A full unroll is already caught above -- it leaves no `while` at all. This
    catches the partial kind, where XLA keeps the loop and lifts some
    iterations out of it, which costs HBM in proportion to how many it lifted.
    """
    self.assertEqual(
        _COMPILED[(_SHALLOW, True)].while_count,
        _COMPILED[(_DEEP, True)].while_count,
        f"the compiled while count changed between depth {_SHALLOW} and"
        f" {_DEEP}; the layer body is being partially unrolled",
    )

  def test_loop_body_does_not_grow_with_depth(self):
    """The size check the count check cannot make.

    A peeled iteration keeps the `while` and keeps the count at one, so only
    the body's size distinguishes it. Depth here grows 4x; measured growth is
    ~1.05x (a handful of lines of loop-bound constants), and a body carrying
    even one lifted layer would land far above the 2x bar.
    """
    shallow = _COMPILED[(_SHALLOW, True)].loop_text_lines
    deep = _COMPILED[(_DEEP, True)].loop_text_lines
    self.assertGreater(shallow, 0, "no loop-bearing module to measure")
    self.assertLess(
        deep,
        2 * shallow,
        f"the loop-bearing HLO grew from {shallow} to {deep} lines while depth"
        f" went from {_SHALLOW} to {_DEEP}; the body scales with layer count,"
        " so layers are being materialised outside the loop",
    )


if __name__ == "__main__":
  absltest.main()
