"""Unit tests for the reverse-by-index scan backward.

The patch is measured -- it is what took llama3-70B from an OOM at 98.43 G to a
61.24 GiB run -- but that evidence lives in a cluster log, so these tests state
the three properties a reader of the diff would otherwise have to take on faith:
the backward stops calling `aten.flip`, the gradients do not change when it
does, and the upstream method this file forks from is still the one upstream
has.

Everything here runs on CPU under `torch.compile`. Never eagerly: an eager scan
never materialises its stacks, so an eager test passes without exercising the
thing the patch changes.
"""

import types

from absl.testing import absltest
import torch
from torch._higher_order_ops.scan import ScanAutogradImpl, scan
from torch.utils._python_dispatch import TorchDispatchMode
import torch.utils._pytree as pytree

from torchtitan.experiments.tpu import scan_backward_patch
from torchtitan.experiments.tpu import workarounds


def _body(carry, x):
  nxt = torch.tanh(carry * 0.9 + x)
  # The clone is not decoration: a HOP body may not alias its carry to its y.
  return nxt, nxt.clone()


def _scan_of_tanh(init, xs):
  """A scan whose backward has to walk the saved stack in reverse."""
  return scan(_body, init, xs)


class _RecordOps(TorchDispatchMode):
  """The flips happen at runtime inside `call_backward`, not in any traced
  graph, so the only place to see them is the dispatcher."""

  supports_higher_order_operators = True

  def __init__(self):
    super().__init__()
    self.ops = []

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    self.ops.append(str(func))
    return func(*args, **(kwargs or {}))


class ScanBackwardPatchTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._unpatched = ScanAutogradImpl.call_backward
    self.addCleanup(setattr, ScanAutogradImpl, "call_backward", self._unpatched)
    self.addCleanup(torch._dynamo.reset)

  def _backward(self):
    """One compiled forward and backward. Returns (grads, ops seen)."""
    torch.manual_seed(0)
    init = torch.randn(3, requires_grad=True)
    xs = torch.randn(5, 3, requires_grad=True)
    torch._dynamo.reset()
    out = torch.compile(_scan_of_tanh, backend="aot_eager")(init, xs)
    loss = sum(o.square().sum() for o in pytree.tree_leaves(out))
    recorder = _RecordOps()
    with recorder:
      grads = torch.autograd.grad(loss, (init, xs))
    return grads, [op for op in recorder.ops if "flip" in op]

  def test_the_upstream_copy_is_still_current(self):
    """The canary. When torch moves, this fails before a 70B run OOMs."""
    self.assertEqual(
        scan_backward_patch._digest(self._unpatched),
        scan_backward_patch._UPSTREAM_DIGEST,
        "torch's ScanAutogradImpl.call_backward has changed. Either the"
        " upstream fix landed -- delete scan_backward_patch.py -- or it moved"
        " under us and the copy needs refreshing. Until then scan reverses by"
        " torch.flip and roughly doubles its peak.",
    )

  def test_it_refuses_to_install_over_a_changed_upstream(self):
    def not_the_method(self, *grad_fw_outputs):
      raise AssertionError("unreachable")

    ScanAutogradImpl.call_backward = not_the_method
    self.assertFalse(scan_backward_patch.apply_scan_reverse_backward_patch())
    self.assertIs(ScanAutogradImpl.call_backward, not_the_method)

  def test_a_scan_run_refuses_to_start_without_the_patch(self):
    """Refusing to install is a warning; refusing with `--use-scan-layers` is
    not. Without the patch that run reverses by copy and OOMs later, which is
    a far worse place to find out."""

    def not_the_method(self, *grad_fw_outputs):
      raise AssertionError("unreachable")

    ScanAutogradImpl.call_backward = not_the_method
    scan_on = types.SimpleNamespace(
        tpu_config=types.SimpleNamespace(use_scan_layers=True)
    )
    with self.assertRaises(RuntimeError):
      workarounds.apply_patches(torch.nn.Identity(), scan_on)

    # ... and a run that never asked for scan is unaffected.
    scan_off = types.SimpleNamespace(
        tpu_config=types.SimpleNamespace(use_scan_layers=False)
    )
    workarounds.apply_patches(torch.nn.Identity(), scan_off)

  def test_it_removes_the_flips_and_leaves_the_gradients_alone(self):
    before, flips_before = self._backward()
    # Asserted, not assumed: without this the next assertion passes for any
    # reason at all, including the test no longer reaching a scan.
    self.assertNotEqual(
        flips_before, [], "unpatched torch did not flip -- test is vacuous"
    )

    self.assertTrue(scan_backward_patch.apply_scan_reverse_backward_patch())
    after, flips_after = self._backward()

    self.assertEqual(flips_after, [], "the backward still reverses by copy")
    for name, b, a in zip(("init", "xs"), before, after):
      torch.testing.assert_close(a, b, msg=f"{name} gradient changed")


if __name__ == "__main__":
  absltest.main()
