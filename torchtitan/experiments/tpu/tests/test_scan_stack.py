# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`ScannedLayerStack` is a pure refactor of the decoder's layer loop.

Three groups of invariants, all on whatever device the suite is pointed at:

  * structure -- parameters stack to `[n_layers, *shape]`, depth does not
    multiply the parameter count, and checkpoints round-trip to the unrolled
    layout in both directions;
  * semantics -- forward outputs and per-layer weight gradients match the plain
    Python loop, with and without the layer-0 peel;
  * graph shape -- the traced graph holds ONE copy of the layer body regardless
    of depth, which is the entire reason to scan.

    python -m torchtitan.experiments.tpu.tests.test_scan_stack
"""

import copy

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.experiments.tpu.scan import layer_stack
from torchtitan.experiments.tpu.tests import scan_fixtures
from torchtitan.experiments.tpu.tests.scan_fixtures import BATCH, DIM, N_LAYERS, SEQ


class ScanStructureTest(BaseAcceleratorDeviceTest):
  """Parameter stacking, layer validation, and checkpoint interchange."""

  def setUp(self):
    super().setUp()
    self.model = scan_fixtures.ToyModel(N_LAYERS, DIM).to(
        self.accelerator_device
    )

  def test_stack_layers_replaces_layers_and_is_idempotent(self):
    stack = layer_stack.stack_layers(self.model)
    self.assertIsInstance(self.model.layers, layer_stack.ScannedLayerStack)
    self.assertIs(layer_stack.stack_layers(self.model), stack)

  def test_parameters_are_stacked_on_dim0(self):
    reference = {
        fqn: tuple(p.shape)
        for fqn, p in self.model.layers["0"].named_parameters()
    }
    stack = layer_stack.stack_layers(self.model)

    self.assertCountEqual(stack.param_fqns, reference)
    for fqn, per_layer_shape in reference.items():
      stacked = stack.get_parameter(layer_stack._mangle(fqn))  # pylint: disable=protected-access
      self.assertEqual(tuple(stacked.shape), (N_LAYERS,) + per_layer_shape)

  def test_one_parameter_per_fqn_not_per_layer(self):
    """The point of the template: depth must not multiply the parameter count."""
    stack = layer_stack.stack_layers(self.model)
    self.assertLen(list(self.model.parameters()), len(stack.param_fqns))

  def test_shadow_layers_hold_no_device_memory(self):
    """Shadows survive for their initializers, but their weights are on `meta`."""
    stack = layer_stack.stack_layers(self.model)
    for shadow in stack._shadow_layers:  # pylint: disable=protected-access
      for fqn, param in shadow.named_parameters():
        self.assertEqual(param.device.type, "meta", f"{fqn} still materialized")

  def test_rejects_layers_with_mismatched_shapes(self):
    model = scan_fixtures.ToyModel(2, DIM)
    model.layers["1"] = scan_fixtures.ToyBlock(DIM + 1)
    with self.assertRaisesRegex(ValueError, "has shape"):
      layer_stack.stack_layers(model)

  def test_rejects_layers_with_mismatched_fqns(self):
    model = scan_fixtures.ToyModel(2, DIM)
    model.layers["1"].extra = nn.Parameter(torch.zeros(1))
    with self.assertRaisesRegex(ValueError, "parameter FQNs"):
      layer_stack.stack_layers(model)

  def test_stacked_to_unrolled_roundtrips(self):
    """A scanned run and an unrolled run must share checkpoints both ways."""
    before = {
        f"{i}.{fqn}": p.detach().clone()
        for i in range(N_LAYERS)
        for fqn, p in self.model.layers[str(i)].named_parameters()
    }
    stack = layer_stack.stack_layers(self.model)

    unrolled = stack.stacked_to_unrolled()
    self.assertCountEqual(unrolled, before)
    for key, want in before.items():
      torch.testing.assert_close(unrolled[key].detach().cpu(), want.cpu())

    perturbed = {k: v + 1.0 for k, v in before.items()}
    stack.load_unrolled(perturbed)
    for key, want in perturbed.items():
      torch.testing.assert_close(
          stack.stacked_to_unrolled()[key].detach().cpu(), want.cpu()
      )

  def test_load_unrolled_rejects_incomplete_state_dict(self):
    stack = layer_stack.stack_layers(self.model)
    unrolled = stack.stacked_to_unrolled()
    unrolled.pop(next(iter(unrolled)))
    with self.assertRaisesRegex(KeyError, "missing"):
      stack.load_unrolled(unrolled)


class ScanSemanticsTest(BaseAcceleratorDeviceTest):
  """Scan must be a pure refactor: same outputs, same gradients."""

  def _model_and_input(self):
    model = scan_fixtures.ToyModel(N_LAYERS, DIM).to(self.accelerator_device)
    h = torch.randn(
        BATCH, SEQ, DIM, device=self.accelerator_device, requires_grad=True
    )
    return model, h

  @parameterized.named_parameters(("plain", False), ("peel_first_layer", True))
  def test_forward_matches_unrolled_loop(self, peel_first_layer):
    model, h = self._model_and_input()
    want = scan_fixtures.unrolled_reference(copy.deepcopy(model), h)

    layer_stack.stack_layers(model, peel_first_layer=peel_first_layer)
    model.eval()
    got = model.layers(h)

    torch.testing.assert_close(got.cpu(), want.cpu(), rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(("plain", False), ("peel_first_layer", True))
  def test_backward_grads_match_unrolled_loop(self, peel_first_layer):
    """Per-layer weight gradients, not just the loss -- this guards the reverse."""
    model, h = self._model_and_input()
    reference = copy.deepcopy(model)
    scan_fixtures.unrolled_reference(reference, h).sum().backward()
    want = {
        f"{i}.{fqn}": p.grad.detach().cpu()
        for i in range(N_LAYERS)
        for fqn, p in reference.layers[str(i)].named_parameters()
    }

    stack = layer_stack.stack_layers(model, peel_first_layer=peel_first_layer)
    model.train()
    model.layers(h).sum().backward()

    for fqn in stack.param_fqns:
      stacked = stack.get_parameter(layer_stack._mangle(fqn))  # pylint: disable=protected-access
      self.assertIsNotNone(stacked.grad, f"no gradient reached {fqn}")
      for i in range(N_LAYERS):
        torch.testing.assert_close(
            stacked.grad[i].cpu(),
            want[f"{i}.{fqn}"],
            rtol=1e-5,
            atol=1e-5,
            msg=f"layer {i} gradient for {fqn} diverges from the unrolled loop",
        )

  def test_gradients_differ_across_layers(self):
    """Regression guard for the silent-zero failure mode.

    If the carry gradient is severed, every layer's weight gradient reflects
    only its own contribution and the per-layer grads collapse toward each other
    while the loss stays bit-exact. Distinct grads prove the chain ran.
    """
    model, h = self._model_and_input()
    stack = layer_stack.stack_layers(model)
    model.train()
    model.layers(h).sum().backward()

    grad = stack.get_parameter(layer_stack._mangle("wi.weight")).grad.cpu()  # pylint: disable=protected-access
    for i in range(1, N_LAYERS):
      self.assertFalse(
          torch.allclose(grad[0], grad[i]),
          f"layer 0 and layer {i} have identical gradients -- the inter-layer"
          " chain is severed (see the carry-requires-grad trap)",
      )

  def test_carry_without_grad_raises_in_training(self):
    """Fail loudly rather than return zeros for the carry gradient."""
    model, _ = self._model_and_input()
    layer_stack.stack_layers(model)
    model.train()
    h = torch.randn(BATCH, SEQ, DIM, device=self.accelerator_device)
    with self.assertRaisesRegex(RuntimeError, "does not require grad"):
      model.layers(h)


class ScanGraphTest(BaseAcceleratorDeviceTest):
  """The structural payoff: one traced body, independent of depth."""

  def _trace(self, n_layers: int):
    model, stack = scan_fixtures.stacked(self.accelerator_device, n_layers)
    model.eval()
    h = torch.randn(BATCH, SEQ, DIM, device=self.accelerator_device)
    return scan_fixtures.trace(stack, h)[0]

  def test_traced_graph_contains_exactly_one_scan_op(self):
    scan_nodes = [
        n
        for n in self._trace(4).nodes
        if n.op == "call_function" and "scan" in str(n.target).lower()
    ]
    self.assertLen(scan_nodes, 1, f"expected one scan op, got {scan_nodes}")

  def test_graph_size_is_independent_of_depth(self):
    """The compile-time win: 4 layers and 16 layers must trace to one body."""
    small = len(self._trace(4).nodes)
    large = len(self._trace(16).nodes)
    self.assertEqual(
        small,
        large,
        f"graph grew {small} -> {large} nodes when depth went 4 -> 16; the"
        " layer body is being unrolled, not scanned",
    )


if __name__ == "__main__":
  absltest.main()
