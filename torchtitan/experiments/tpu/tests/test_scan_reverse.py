# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the reverse-scan HOP and the backward rewrite built on it.

Three claims, in dependency order:

  1. `scan_reverse_op` means what `scan/reverse.py` says it means -- last index
     first, `ys` indexed by input position. Checked against a plain Python loop,
     so the definition is pinned by something with no shared machinery.
  2. The TPU lowering (`ScanDirection.kReverse`) agrees with that definition.
     It has to be checked separately: it is a different implementation, and
     until this patch nothing in the stack exercised it.
  3. Installing the rewrite does not change a single gradient. This is the one
     that matters -- the whole point is to remove a copy, and a memory
     optimisation that perturbs gradients is not an optimisation.

Requires a TPU:

    python -m torchtitan.experiments.tpu.tests.test_scan_reverse
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torch._higher_order_ops.scan import scan as scan_public
from torch._higher_order_ops.utils import materialize_as_graph
import torch_tpu  # pylint: disable=unused-import  (registers the `tpu` device)
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.experiments.tpu.scan import reverse as scan_reverse


_DEVICE = "tpu:0"


class _TpuOnly:
  """Skips a `BaseAcceleratorDeviceTest` unless it is running on a TPU.

  A mixin rather than a base class so the test loader does not collect it as a
  test case of its own, and so `BaseAcceleratorDeviceTest` stays a literal base
  of every class below: `.ci/discover_tpu_tests.py` classifies by base-class
  name, and that name is what routes this module to the single-TPU shard. A
  plain `parameterized.TestCase` would be scheduled on a CPU runner instead,
  where the device tests fail for want of a device. The skip keeps
  `--device=cpu` honest rather than red.
  """

  def setUp(self):
    super().setUp()
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("asserts on the torch_tpu lowering; TPU only")


_WIDTH = 4


def _python_reverse_scan(combine_fn, init, xs):
  """The definition, written out. Visits N-1..0; `ys[i]` pairs with `xs[i]`."""
  carry = list(init)
  ys = [None] * xs[0].shape[0]
  for i in reversed(range(xs[0].shape[0])):
    out = combine_fn(*carry, *[x[i] for x in xs])
    carry = [out[0]]
    ys[i] = out[1]
  return carry, torch.stack(ys)


def _add_and_emit(carry, x):
  # The `clone()` is load-bearing on device. torch_tpu's meta-run passes real
  # placeholder tensors into a FakeTensorMode, and FakeTensorMode
  # constant-folds any number-binding op (`add` among them) whose arguments
  # are all non-fake -- on a placeholder that raises "cannot Materialize()".
  # One non-folding op in front makes the operand fake and the fold stops.
  # Not a property of the reverse op: the stock forward scan needs it too.
  nxt = carry.clone() + x
  return nxt, nxt.clone()


def _as_graph(fn, init, xs):
  """torch_tpu's lowering calls `fx_to_mlir`, which needs a GraphModule.

  Every real caller reaches the HOP with one already -- autograd builds the
  backward body through `materialize_as_graph`. A test that hands it a plain
  Python function is testing a path that does not exist, and fails on
  `named_modules`; the stock forward scan fails the same way.
  """
  return materialize_as_graph(fn, (*init, *[x[0] for x in xs]))


def _ramp(n):
  """`xs[i] = [i+1, i+1, ...]`, so a running sum is readable by inspection."""
  return torch.arange(1.0, n + 1.0).unsqueeze(1).expand(n, _WIDTH).contiguous()


class ScanReverseSemanticsTest(_TpuOnly, BaseAcceleratorDeviceTest):
  """Claim 1 + 2: the op, and its TPU lowering, match the written definition."""

  def test_dense_matches_python_reference(self):
    xs = [_ramp(4)]
    init = [torch.zeros(_WIDTH)]

    carry, ys = scan_reverse.scan_reverse_op(_add_and_emit, init, xs, ())
    want_carry, want_ys = _python_reverse_scan(_add_and_emit, init, xs)

    torch.testing.assert_close(carry, want_carry[0])
    torch.testing.assert_close(ys, want_ys)

  def test_ys_is_indexed_by_input_position_not_iteration_order(self):
    """The property the whole rewrite rests on, asserted on its own.

    A reverse scan over `[1,2,3,4]` accumulating sums visits 4,3,2,1 and
    produces running totals 4,7,9,10. Written in *iteration* order that is
    `[4,7,9,10]`; written in *input* order it is `[10,9,7,4]`. Only the second
    lets the caller drop the output flip, so it is worth an explicit test
    rather than being left implicit in the reference comparison above.
    """
    _, ys = scan_reverse.scan_reverse_op(
        _add_and_emit, [torch.zeros(_WIDTH)], [_ramp(4)], ()
    )
    torch.testing.assert_close(ys[:, 0], torch.tensor([10.0, 9.0, 7.0, 4.0]))

  def test_device_lowering_matches_dense(self):
    xs_cpu = [_ramp(8)]
    init_cpu = [torch.zeros(_WIDTH)]
    want_carry, want_ys = scan_reverse.scan_reverse_op(
        _add_and_emit, init_cpu, xs_cpu, ()
    )

    xs = [xs_cpu[0].to(_DEVICE)]
    init = [init_cpu[0].to(_DEVICE)]
    got_carry, got_ys = scan_reverse.scan_reverse_op(
        _as_graph(_add_and_emit, init, xs), init, xs, ()
    )

    torch.testing.assert_close(got_carry.cpu(), want_carry)
    torch.testing.assert_close(got_ys.cpu(), want_ys)

  def test_forward_scan_is_unaffected(self):
    """The sibling HOP must not have perturbed the one it sits next to."""
    xs = [_ramp(4).to(_DEVICE)]
    init = [torch.zeros(_WIDTH).to(_DEVICE)]
    _, ys = torch.ops.higher_order.scan(
        _as_graph(_add_and_emit, init, xs), init, xs, ()
    )
    torch.testing.assert_close(
        ys.cpu()[:, 0], torch.tensor([1.0, 3.0, 6.0, 10.0])
    )


class _ScannedStack(nn.Module):
  """A depth-stacked linear block, differentiable through `scan`.

  Small enough to compare gradients exactly, structured like the real layer
  stack: one set of weights per depth index, stacked on dim 0 and consumed as
  `xs`, with the activation as the carry.
  """

  def __init__(self, n_layers: int, dim: int):
    super().__init__()
    self.w = nn.Parameter(torch.randn(n_layers, dim, dim) / dim**0.5)
    self.b = nn.Parameter(torch.randn(n_layers, dim) / dim**0.5)

  def forward(self, h):
    # The public `scan` hands the combine_fn the *pytrees* of init and xs, so
    # `carry` arrives as a one-element list and `x` as the [w, b] pair.
    def body(carry, x):
      w, b = x
      nxt = torch.tanh(carry[0] @ w + b)
      return [nxt], nxt.clone()

    carry, ys = scan_public(body, [h], [self.w, self.b])
    return carry[0].sum() + ys.sum()


def _grads(device, seed=0):
  # Each call recompiles the same code object at the same shapes, which trips
  # dynamo's automatic-dynamic heuristic on the second pass; the resulting
  # SymInt `additional_inputs` are not tensors and torch_tpu's `create_scan_op`
  # rejects them. Static shapes are what the real lane traces under anyway.
  torch._dynamo.reset()
  torch.manual_seed(seed)
  model = _ScannedStack(n_layers=6, dim=4).to(device)
  h = torch.randn(2, 4, device=device, requires_grad=True)
  loss = model(h)
  loss.backward()
  return (
      loss.detach().cpu(),
      h.grad.cpu(),
      model.w.grad.cpu(),
      model.b.grad.cpu(),
  )


class BackwardRewriteTest(_TpuOnly, BaseAcceleratorDeviceTest):
  """Claim 3: the rewrite is gradient-neutral."""

  def tearDown(self):
    scan_reverse.uninstall_scan_reverse_backward()
    super().tearDown()

  @parameterized.named_parameters(("cpu", "cpu"), ("tpu", _DEVICE))
  def test_gradients_match_stock_backward(self, device):
    if device != "cpu":
      # Not a gap this patch introduced: eager autograd-through-`scan` on
      # device fails on the *unpatched* path too, with "failed to upgrade
      # versioned StableHLO" out of `h.grad.cpu()`. The same error blocks
      # TPU_SCAN_PEEL_FIRST_LAYER=0. The lane does not reach scan this way --
      # it traces a joint graph through aot_fx_trace -- so device-side
      # gradient equivalence is established by the training run's loss
      # trajectory against the same-shape baseline instead.
      self.skipTest("eager autograd-through-scan is broken on device upstream")
    want = _grads(device)
    with scan_reverse.scan_reverse_backward():
      got = _grads(device)

    names = ("loss", "grad_h", "grad_w", "grad_b")
    for name, w, g in zip(names, want, got):
      torch.testing.assert_close(g, w, msg=lambda m, n=name: f"{n}: {m}")

  def test_backward_actually_dispatches_the_reverse_scan(self):
    """Without this the parity test above would also pass on a no-op patch."""
    seen = []
    original = scan_reverse.scan_reverse_op

    def spy(*args, **kwargs):
      seen.append(args)
      return original(*args, **kwargs)

    # Swap the module attribute, not the instance's `__call__`: the patched
    # `call_backward` reaches the op through this global, and Python resolves
    # dunder methods on the type anyway.
    with scan_reverse.scan_reverse_backward():
      scan_reverse.scan_reverse_op = spy
      try:
        _grads("cpu")
      finally:
        scan_reverse.scan_reverse_op = original

    self.assertLen(seen, 1)

  def test_install_is_idempotent(self):
    scan_reverse.install_scan_reverse_backward()
    first = scan_reverse._ORIGINAL_CALL_BACKWARD
    scan_reverse.install_scan_reverse_backward()
    self.assertIs(scan_reverse._ORIGINAL_CALL_BACKWARD, first)

  def test_uninstall_restores_the_stock_backward(self):
    import importlib

    scan_mod = importlib.import_module("torch._higher_order_ops.scan")

    stock = scan_mod.ScanAutogradImpl.call_backward
    scan_reverse.install_scan_reverse_backward()
    self.assertIsNot(scan_mod.ScanAutogradImpl.call_backward, stock)
    scan_reverse.uninstall_scan_reverse_backward()
    self.assertIs(scan_mod.ScanAutogradImpl.call_backward, stock)

  def test_a_flip_inside_the_scanned_body_is_left_alone(self):
    """The two-phase guard, tested at the case that motivates it.

    The elision mode is live while `materialize_as_graph` traces the backward
    body. If it matched on shape it would eat a `torch.flip(_, (0,))` the model
    itself performs and silently return wrong gradients, so it matches on
    tensor identity instead. A body that flips its own carry is the smallest
    thing that would catch a regression to shape matching.
    """
    def flipping_body(carry, x):
      nxt = torch.flip(carry[0], (0,)) + x[0]
      return [nxt], nxt.clone()

    def run():
      torch.manual_seed(0)
      w = torch.randn(5, 3, requires_grad=True)
      h = torch.randn(3, requires_grad=True)
      carry, ys = scan_public(flipping_body, [h], [w])
      (carry[0].sum() + ys.sum()).backward()
      return h.grad.clone(), w.grad.clone()

    want = run()
    with scan_reverse.scan_reverse_backward():
      got = run()

    torch.testing.assert_close(got[0], want[0])
    torch.testing.assert_close(got[1], want[1])


if __name__ == "__main__":
  absltest.main()
