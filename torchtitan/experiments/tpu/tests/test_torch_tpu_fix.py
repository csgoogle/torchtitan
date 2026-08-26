# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the TEMPORARY torch_tpu scan workaround in `scan/torch_tpu_fix.py`.

The workaround re-registers a fixed `_handle_scan_impl` over torch_tpu's, and
`_handle_scan_impl_fixed` is a line-for-line mirror of one specific upstream
revision. Forcing that mirror over a torch_tpu that has moved on is worse than
not patching at all, so the module gates the patch on a source-signature check.
That check is the thing most worth testing: it is the workaround's end-of-life
detector, it fails open (patches when unsure), and when it misfires the symptom
is a wrong scan body rather than an exception.

The check is deliberately exercised against synthetic modules, not just the
installed torch_tpu, because the two torch_tpu builds in play disagree -- and
that disagreement is the point:

  - the pinned container image still has the defect, so the patch installs;
  - newer torch_tpu builds construct `body_args` from the cleaned lists, so the
    detector reports the defect gone and the patch stays off.

`ContractAgainstInstalledTorchTpu` records whichever of those two the current
environment is, so the day upstream's fix reaches every build we run on, the
test tells us the file can be deleted instead of leaving it to rot.
"""

import types
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torchtitan.experiments.tpu.scan import torch_tpu_fix


def _module_with_impl(source: str) -> types.ModuleType:
  """A stand-in torch_tpu scan module whose `_handle_scan_impl` has `source`.

  `inspect.getsource` reads the file the function was compiled from, so the
  body has to be written to a real file for the detector to see it. `exec` of a
  string would raise `OSError` instead, which is a different branch.
  """
  path = absltest.TestCase().create_tempfile(
      "fake_scan.py", content=source
  ).full_path
  module = types.ModuleType("fake_torch_tpu_scan")
  module.__file__ = path
  with open(path) as f:
    exec(compile(f.read(), path, "exec"), module.__dict__)  # pylint: disable=exec-used
  return module


_BUGGY_SOURCE = '''
def _handle_scan_impl(combine_fn, init, xs, additional_inputs, reverse=False):
  init_list = list(init)
  x_list = list(xs)
  extra_list = list(additional_inputs)
  body_args = init_list + x_list + extra_list
  return body_args
'''

_FIXED_SOURCE = '''
def _handle_scan_impl(combine_fn, init, xs, additional_inputs, reverse=False):
  init_list = list(init)
  x_list = list(xs)
  clean_inits = [t.clone() for t in init_list]
  clean_extras = [t.clone() for t in additional_inputs]
  body_args = clean_inits + x_list + clean_extras
  return body_args
'''

# Same defect, reformatted across lines. Reformatting alone must not read as a
# fix: the detector strips whitespace before matching for exactly this case.
_BUGGY_REFORMATTED_SOURCE = '''
def _handle_scan_impl(combine_fn, init, xs, additional_inputs, reverse=False):
  init_list = list(init)
  x_list = list(xs)
  extra_list = list(additional_inputs)
  body_args = (
      init_list
      + x_list
      + extra_list
  )
  return body_args
'''


class DefectDetectionTest(parameterized.TestCase):
  """`_upstream_still_has_bug_a` -- True defective, False moved on, None unknown."""

  def test_detects_the_defect(self):
    module = _module_with_impl(_BUGGY_SOURCE)
    self.assertTrue(torch_tpu_fix._upstream_still_has_bug_a(module))

  def test_reformatting_alone_does_not_read_as_a_fix(self):
    module = _module_with_impl(_BUGGY_REFORMATTED_SOURCE)
    self.assertTrue(torch_tpu_fix._upstream_still_has_bug_a(module))

  def test_detects_the_upstream_fix(self):
    """`clean_inits + x_list + clean_extras` is what the landed fix looks like."""
    module = _module_with_impl(_FIXED_SOURCE)
    self.assertFalse(torch_tpu_fix._upstream_still_has_bug_a(module))

  def test_missing_impl_is_unknown_not_fixed(self):
    """None and False both mean "do not report the defect" but differ downstream.

    None makes the caller patch anyway (fail open); False suppresses the patch.
    Confusing the two would silently stop patching a torch_tpu that still needs
    it, which is the failure this workaround exists to prevent.
    """
    self.assertIsNone(
        torch_tpu_fix._upstream_still_has_bug_a(types.ModuleType("empty"))
    )

  def test_unreadable_source_is_unknown_not_fixed(self):
    """A C extension or an `exec`-defined impl raises inside `inspect.getsource`."""
    module = types.ModuleType("builtin_impl")
    module._handle_scan_impl = len  # no Python source
    self.assertIsNone(torch_tpu_fix._upstream_still_has_bug_a(module))


class InstallTest(parameterized.TestCase):
  """`use_torch_tpu_scan_fix` -- idempotent, and honest about not installing."""

  def setUp(self):
    super().setUp()
    self._applied = torch_tpu_fix._APPLIED
    self.addCleanup(setattr, torch_tpu_fix, "_APPLIED", self._applied)

  def test_is_idempotent(self):
    torch_tpu_fix._APPLIED = True
    self.assertTrue(torch_tpu_fix.use_torch_tpu_scan_fix())

  def test_reports_false_when_torch_tpu_is_absent(self):
    """A host without torch_tpu must get a no-op, not an ImportError."""
    torch_tpu_fix._APPLIED = False
    real_import = __builtins__["__import__"] if isinstance(
        __builtins__, dict
    ) else __builtins__.__import__

    def blocked(name, *args, **kwargs):
      if name.startswith("torch_tpu"):
        raise ImportError("torch_tpu is not installed")
      return real_import(name, *args, **kwargs)

    with unittest.mock.patch("builtins.__import__", blocked):
      self.assertFalse(torch_tpu_fix.use_torch_tpu_scan_fix())
    self.assertFalse(torch_tpu_fix._APPLIED)

  def test_the_fixed_impl_mirrors_the_upstream_signature(self):
    """The replacement is dispatched positionally; a drifted signature is silent.

    `py_impl` does not check the signature, so a rename or a reordered argument
    would only surface as a wrong scan body at runtime on a TPU.
    """
    import inspect

    params = list(
        inspect.signature(torch_tpu_fix._handle_scan_impl_fixed).parameters
    )
    self.assertEqual(
        params, ["combine_fn", "init", "xs", "additional_inputs", "reverse"]
    )


class ContractAgainstInstalledTorchTpu(parameterized.TestCase):
  """What the workaround actually does against the torch_tpu in this env.

  Both outcomes are legitimate; the test asserts the module's behaviour is
  consistent with whichever torch_tpu is installed, and names the outcome so a
  failure here is read as "the environment changed", not "the code broke".
  """

  def setUp(self):
    super().setUp()
    try:
      import torch_tpu._internal.compile.scan as tt_scan
    except ImportError:
      self.skipTest("torch_tpu is not installed on this host")
    self._tt_scan = tt_scan

  def test_patch_state_matches_the_defect_state(self):
    has_bug = torch_tpu_fix._upstream_still_has_bug_a(self._tt_scan)
    if has_bug is False:
      # Upstream has moved on. The patch must NOT be forced over it, and the
      # scan HOP must still dispatch to torch_tpu's own impl.
      self.assertIsNot(
          torch.ops.higher_order.scan.py_kernels.get(
              torch._C.DispatchKey.PrivateUse1
          ),
          torch_tpu_fix._handle_scan_impl_fixed,
          "the workaround installed itself over a torch_tpu that no longer has"
          " Bug A -- re-diagnose and delete torch_tpu_fix.py if the fix landed",
      )
    else:
      # The defect is present (or undetectable): patching must succeed.
      self.assertTrue(torch_tpu_fix.use_torch_tpu_scan_fix())


if __name__ == "__main__":
  absltest.main()
