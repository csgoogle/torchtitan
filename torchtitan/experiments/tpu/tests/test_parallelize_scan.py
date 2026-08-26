# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the graph_trainer lane's scan wiring and the `tpu/scan` public API.

`graph_trainer/parallelize_scan.py` is the lane the 70B target runs on, and
almost everything it does is *ordering*: the four steps it performs are
individually tested elsewhere, but the order between them is a real constraint
that nothing else checks.

  1. `install_scan_over_layers` must precede `annotate_llama` -- otherwise the
     FQN annotations describe the per-layer modules the stack replaced.
  2. it must also precede `apply_simple_fsdp`, which iterates `model.layers`'
     children; the stack has a different child structure.
  3. `apply_simple_fsdp` must receive the stacked parameters as `ignored_params`
     -- they are gathered per layer inside the scan body instead, and gathering
     them twice is the 136.9 GB this design exists to avoid.
  4. `shard_stacked_layers` must come last: the in-body gather plan can only be
     built once the mesh is known.

Getting any of these wrong produces a model that trains -- slowly, or at the
wrong memory -- rather than one that raises, so the order is asserted directly
with recording doubles instead of inferred from a trace.

No TPU is needed, but the lane's own module-level imports (`graph_trainer.
compile`, `graph_trainer.llama3.parallelize`) require a newer torch than a
plain host checkout usually has, so the four lane classes skip when the import
fails and run under `run_in_container.sh`. `ScanGuardTest` and `PublicApiTest`
depend only on `tpu/scan` and run everywhere.
"""

import dataclasses
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn

try:
  from torchtitan.experiments.tpu.graph_trainer import parallelize_scan
  _LANE_IMPORT_ERROR = None
except ImportError as e:  # torch too old for the graph_trainer tree
  parallelize_scan = None
  _LANE_IMPORT_ERROR = e

_requires_lane = unittest.skipIf(
    parallelize_scan is None,
    f"graph_trainer lane is not importable here: {_LANE_IMPORT_ERROR}",
)


@dataclasses.dataclass
class _FakeParallelDims:
  tp: int = 1
  cp: int = 1
  pp: int = 1
  seq_len_divisor: int = 1
  tp_enabled: bool = False
  cp_enabled: bool = False
  pp_enabled: bool = False


@dataclasses.dataclass
class _FakeTraining:
  seq_len: int = 8192


@dataclasses.dataclass
class _FakeParallelism:
  use_simple_fsdp: bool = True


@dataclasses.dataclass
class _FakeCompile:
  memory_policy: str = "default"


class _FakeStack(nn.Module):

  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(2, 2))

  def stacked_parameters(self):
    return [self.weight]


def _call_parallelize(calls: list, **overrides):
  """Runs `parallelize_llama_scan` with every collaborator replaced by a double.

  Returns the kwargs `apply_simple_fsdp` and `shard_stacked_layers` were called
  with, plus `calls`, which records the order the collaborators ran in.
  """
  model = nn.Module()
  stack = _FakeStack()
  captured = {}

  def record(name, result=None):
    def _fn(*args, **kwargs):
      calls.append(name)
      captured[name] = kwargs
      return result
    return _fn

  patches = {
      "install_scan_over_layers": record("install_scan_over_layers", stack),
      "annotate_llama": record("annotate_llama"),
      "apply_simple_fsdp": record("apply_simple_fsdp", model),
      "shard_stacked_layers": record("shard_stacked_layers"),
      "apply_compile": record("apply_compile", model),
      "apply_cp_to_attention": record("apply_cp_to_attention"),
      "resolve_dp_plan": lambda *a, **k: ("mesh", "mode", "policy"),
  }

  kwargs = dict(
      parallel_dims=_FakeParallelDims(),
      training=_FakeTraining(),
      parallelism=_FakeParallelism(),
      compile_config=_FakeCompile(),
      ac_config=object(),
      dump_folder="/tmp",
  )
  kwargs.update(overrides)

  with mock.patch.multiple(
      parallelize_scan,
      annotate_llama=patches["annotate_llama"],
      apply_simple_fsdp=patches["apply_simple_fsdp"],
      apply_compile=patches["apply_compile"],
      apply_cp_to_attention=patches["apply_cp_to_attention"],
      resolve_dp_plan=patches["resolve_dp_plan"],
  ), mock.patch(
      "torchtitan.experiments.tpu.scan.layer_stack.install_scan_over_layers",
      patches["install_scan_over_layers"],
  ), mock.patch(
      "torchtitan.experiments.tpu.scan.fsdp.shard_stacked_layers",
      patches["shard_stacked_layers"],
  ):
    parallelize_scan.parallelize_llama_scan(model, **kwargs)

  return captured, stack


@_requires_lane
class StepOrderTest(parameterized.TestCase):
  """The four steps, and the order between them."""

  def test_all_four_steps_run(self):
    calls = []
    _call_parallelize(calls)
    self.assertContainsSubset(
        [
            "install_scan_over_layers",
            "annotate_llama",
            "apply_simple_fsdp",
            "shard_stacked_layers",
        ],
        calls,
    )

  def test_the_steps_run_in_order(self):
    calls = []
    _call_parallelize(calls)
    ordered = [c for c in calls if c != "apply_compile"]
    self.assertEqual(
        ordered,
        [
            "install_scan_over_layers",
            "annotate_llama",
            "apply_simple_fsdp",
            "shard_stacked_layers",
        ],
    )

  def test_compile_runs_last(self):
    calls = []
    _call_parallelize(calls)
    self.assertEqual(calls[-1], "apply_compile")

  def test_fsdp_ignores_the_stacked_parameters(self):
    """Gathering them at the top level is exactly the 136.9 GB to avoid."""
    calls = []
    captured, stack = _call_parallelize(calls)
    self.assertEqual(
        captured["apply_simple_fsdp"]["ignored_params"],
        set(stack.stacked_parameters()),
    )


@_requires_lane
class MemoryPolicyTest(parameterized.TestCase):
  """`recompute_body` is derived from `memory_policy`, not from `ac_config`.

  This lane discards `ac_config` -- `to_graph_trainer_config` rewrites it to
  SelectiveAC regardless of what the user asked for -- so `memory_policy` is
  the only field here that carries user intent about activation memory.
  """

  @parameterized.named_parameters(
      ("full", "full", True),
      ("default", "default", False),
      ("none", "none", False),
  )
  def test_recompute_body_follows_memory_policy(self, policy, expected):
    calls = []
    captured, _ = _call_parallelize(
        calls, compile_config=_FakeCompile(memory_policy=policy)
    )
    self.assertEqual(
        captured["shard_stacked_layers"]["recompute_body"], expected
    )

  def test_a_compile_config_without_memory_policy_defaults_to_off(self):
    calls = []
    captured, _ = _call_parallelize(calls, compile_config=object())
    self.assertFalse(captured["shard_stacked_layers"]["recompute_body"])


@_requires_lane
class GuardTest(parameterized.TestCase):
  """The configurations this lane refuses, and refuses *before* touching the model."""

  def test_requires_simple_fsdp(self):
    calls = []
    with self.assertRaisesRegex(ValueError, "use-simple-fsdp"):
      _call_parallelize(calls, parallelism=_FakeParallelism(use_simple_fsdp=False))

  def test_the_simple_fsdp_guard_runs_before_the_model_is_rewritten(self):
    """A guard that fires after `install_scan_over_layers` leaves a half-built model."""
    calls = []
    with self.assertRaises(ValueError):
      _call_parallelize(calls, parallelism=_FakeParallelism(use_simple_fsdp=False))
    self.assertNotIn("install_scan_over_layers", calls)

  def test_rejects_a_seq_len_the_mesh_cannot_divide(self):
    calls = []
    with self.assertRaises(AssertionError):
      _call_parallelize(
          calls,
          training=_FakeTraining(seq_len=8191),
          parallel_dims=_FakeParallelDims(seq_len_divisor=8),
      )


class ScanGuardTest(parameterized.TestCase):
  """`install_scan_over_layers` rejects the parallelisms it cannot describe.

  TP / CP / PP each reshard the layer weights, and the stacked parameters carry
  a leading layer axis that none of the existing parallel plans mention. The
  guard exists so that combination raises instead of silently mis-sharding.
  """

  @parameterized.named_parameters(("tp", "tp"), ("cp", "cp"), ("pp", "pp"))
  def test_rejects_unsupported_parallelism(self, dim):
    from torchtitan.experiments.tpu.scan import layer_stack

    dims = _FakeParallelDims(**{f"{dim}_enabled": True})
    with self.assertRaisesRegex(ValueError, dim.upper()):
      layer_stack.install_scan_over_layers(
          nn.Module(), parallel_dims=dims, peel_first_layer=False
      )

  def test_the_guard_runs_before_any_torch_tpu_patch_is_installed(self):
    """Rejecting a config must not leave the process's scan HOP monkeypatched."""
    from torchtitan.experiments.tpu.scan import layer_stack

    with mock.patch(
        "torchtitan.experiments.tpu.scan.torch_tpu_fix.use_torch_tpu_scan_fix"
    ) as patch_fn:
      with self.assertRaises(ValueError):
        layer_stack.install_scan_over_layers(
            nn.Module(),
            parallel_dims=_FakeParallelDims(tp_enabled=True),
            peel_first_layer=False,
        )
    patch_fn.assert_not_called()


class PublicApiTest(parameterized.TestCase):
  """`torchtitan.experiments.tpu.scan`'s documented surface.

  The package docstring is the entry point a reader outside this lane starts
  from; these assert the names it promises actually resolve, so the example in
  it cannot rot silently.
  """

  def test_documented_entry_points_are_importable(self):
    from torchtitan.experiments.tpu import scan

    self.assertTrue(callable(scan.scan_over_layers))
    self.assertTrue(callable(scan.shard_scanned_layers))

  def test_the_aliases_are_the_underlying_functions(self):
    from torchtitan.experiments.tpu import scan
    from torchtitan.experiments.tpu.scan import fsdp, layer_stack

    self.assertIs(scan.scan_over_layers, layer_stack.install_scan_over_layers)
    self.assertIs(scan.shard_scanned_layers, fsdp.shard_stacked_layers)

  def test_everything_in_all_resolves(self):
    from torchtitan.experiments.tpu import scan

    missing = [name for name in scan.__all__ if not hasattr(scan, name)]
    self.assertEmpty(missing)

  def test_the_docstrings_worked_example_actually_binds(self):
    """The three checks above all passed while the example raised TypeError.

    Resolving a name is not the same as being callable the way the docstring
    calls it: `peel_first_layer` is keyword-only with no default, so the
    example's two-argument form did not bind. Binding is asserted rather than
    called because the real call needs a model and a mesh; the defect was in
    the signature, and that is what this catches.
    """
    import inspect
    import re

    from torchtitan.experiments.tpu import scan

    source = inspect.getdoc(scan) or ""
    match = re.search(
        r"stack = scan_over_layers\((.*?)\)\n", source, re.DOTALL
    )
    self.assertIsNotNone(
        match, "the package docstring no longer shows a scan_over_layers call"
    )

    kwargs = {}
    for part in match.group(1).split(",")[1:]:
      name, _, _ = part.partition("=")
      kwargs[name.strip()] = mock.ANY

    inspect.signature(scan.scan_over_layers).bind(mock.ANY, **kwargs)


if __name__ == "__main__":
  absltest.main()
