# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for scan-over-layers (`torchtitan.experiments.tpu.scan.layer_stack`).

These run on CPU and cover the invariants that are expensive to discover on a
TPU run: parameter stacking, checkpoint interchange, forward/backward numerical
equivalence against the unrolled loop, and the structural property that makes
scan worth doing at all -- the traced graph holds ONE copy of the layer body
regardless of depth.

The TPU-only half of the contract (a single `stablehlo.while` and a lowered
module that does not grow with depth) lives in `test_scan_lowering.py`, which
needs a device. The graph-trainer SAC interaction lives in
`test_scan_memory_policy.py`.
"""

import copy

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torchtitan.experiments.tpu.scan import layer_stack as scan_model


class _ToyBlock(nn.Module):
  """A decoder-block stand-in: same call signature, a fraction of the cost.

  Matches `TransformerBlock.forward(h, attention_masks, positions)` so the
  stack exercises the real `functional_call` path.
  """

  def __init__(self, dim: int):
    super().__init__()
    self.wi = nn.Linear(dim, dim, bias=False)
    self.wo = nn.Linear(dim, dim, bias=False)

  def forward(self, h, attention_masks=None, positions=None):
    del attention_masks, positions
    return h + self.wo(torch.tanh(self.wi(h)))


class _ToyModel(nn.Module):
  """Owns a `ModuleDict` of blocks keyed by index, as llama3 does."""

  def __init__(self, n_layers: int, dim: int):
    super().__init__()
    self.layers = nn.ModuleDict(
        {str(i): _ToyBlock(dim) for i in range(n_layers)}
    )

  def forward(self, h):
    for i in range(len(self.layers)):
      h = self.layers[str(i)](h)
    return h


def _sequential_reference(model: _ToyModel, h: torch.Tensor) -> torch.Tensor:
  """Runs the blocks in a plain Python loop -- the semantics scan must preserve."""
  for i in range(len(model.layers)):
    h = model.layers[str(i)](h)
  return h


class ScanStructureTest(parameterized.TestCase):
  """Parameter stacking, layer-shape validation, and checkpoint interchange."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    self.n_layers, self.dim = 4, 8
    self.model = _ToyModel(self.n_layers, self.dim)

  def test_apply_replaces_layers_and_is_idempotent(self):
    scan_model.apply_scan_over_layers(self.model)
    self.assertIsInstance(self.model.layers, scan_model.ScannedLayerStack)

    first = self.model.layers
    scan_model.apply_scan_over_layers(self.model)
    self.assertIs(self.model.layers, first, "second apply must be a no-op")

  def test_parameters_are_stacked_on_dim0(self):
    reference = {
        fqn: tuple(p.shape)
        for fqn, p in self.model.layers["0"].named_parameters()
    }
    scan_model.apply_scan_over_layers(self.model)
    stack = self.model.layers

    self.assertCountEqual(stack.param_fqns, reference)
    for fqn, per_layer_shape in reference.items():
      stacked = stack.get_parameter(scan_model._mangle(fqn))  # pylint: disable=protected-access
      self.assertEqual(
          tuple(stacked.shape),
          (self.n_layers,) + per_layer_shape,
          f"{fqn} must stack to [n_layers, *shape]",
      )

  def test_one_parameter_per_fqn_not_per_layer(self):
    """The whole point of the template: depth must not multiply parameter count."""
    scan_model.apply_scan_over_layers(self.model)
    n_params = len(list(self.model.parameters()))
    self.assertEqual(
        n_params,
        len(self.model.layers.param_fqns),
        "a scanned stack exposes one stacked tensor per FQN, not one per layer",
    )

  def test_rejects_layers_with_mismatched_shapes(self):
    model = _ToyModel(2, self.dim)
    model.layers["1"] = _ToyBlock(self.dim + 1)
    with self.assertRaisesRegex(ValueError, "has shape"):
      scan_model.apply_scan_over_layers(model)

  def test_rejects_layers_with_mismatched_fqns(self):
    model = _ToyModel(2, self.dim)
    model.layers["1"].extra = nn.Parameter(torch.zeros(1))
    with self.assertRaisesRegex(ValueError, "parameter FQNs"):
      scan_model.apply_scan_over_layers(model)

  def test_stacked_to_unrolled_roundtrips(self):
    """A scanned run and an unrolled run must share checkpoints both ways."""
    before = {
        f"{i}.{fqn}": p.detach().clone()
        for i in range(self.n_layers)
        for fqn, p in self.model.layers[str(i)].named_parameters()
    }
    scan_model.apply_scan_over_layers(self.model)

    unrolled = self.model.layers.stacked_to_unrolled()
    self.assertCountEqual(unrolled, before)
    for key, want in before.items():
      torch.testing.assert_close(unrolled[key], want)

    perturbed = {k: v + 1.0 for k, v in before.items()}
    self.model.layers.load_unrolled(perturbed)
    for key, want in perturbed.items():
      torch.testing.assert_close(
          self.model.layers.stacked_to_unrolled()[key], want
      )

  def test_load_unrolled_rejects_incomplete_state_dict(self):
    scan_model.apply_scan_over_layers(self.model)
    unrolled = self.model.layers.stacked_to_unrolled()
    unrolled.pop(next(iter(unrolled)))
    with self.assertRaisesRegex(KeyError, "missing"):
      self.model.layers.load_unrolled(unrolled)


class ScanSemanticsTest(parameterized.TestCase):
  """Scan must be a pure refactor: same outputs, same gradients."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    self.n_layers, self.dim, self.batch, self.seq = 4, 8, 2, 3

  def _inputs(self):
    return torch.randn(
        self.batch, self.seq, self.dim, dtype=torch.float32, requires_grad=True
    )

  @parameterized.named_parameters(("plain", False), ("peel_first_layer", True))
  def test_forward_matches_unrolled_loop(self, peel_first_layer):
    model = _ToyModel(self.n_layers, self.dim)
    reference_model = copy.deepcopy(model)
    h = self._inputs()

    want = _sequential_reference(reference_model, h)
    scan_model.apply_scan_over_layers(model, peel_first_layer=peel_first_layer)
    model.eval()
    got = model.layers(h)

    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(("plain", False), ("peel_first_layer", True))
  def test_backward_grads_match_unrolled_loop(self, peel_first_layer):
    """Guards the reverse-scan: per-layer weight grads, not just the loss."""
    model = _ToyModel(self.n_layers, self.dim)
    reference_model = copy.deepcopy(model)
    h = self._inputs()

    _sequential_reference(reference_model, h).sum().backward()
    want = {
        f"{i}.{fqn}": p.grad.detach().clone()
        for i in range(self.n_layers)
        for fqn, p in reference_model.layers[str(i)].named_parameters()
    }

    scan_model.apply_scan_over_layers(model, peel_first_layer=peel_first_layer)
    model.train()
    model.layers(h).sum().backward()

    for fqn in model.layers.param_fqns:
      stacked = model.layers.get_parameter(scan_model._mangle(fqn))  # pylint: disable=protected-access
      self.assertIsNotNone(stacked.grad, f"no gradient reached {fqn}")
      for i in range(self.n_layers):
        torch.testing.assert_close(
            stacked.grad[i],
            want[f"{i}.{fqn}"],
            rtol=1e-5,
            atol=1e-5,
            msg=f"layer {i} gradient for {fqn} diverges from the unrolled loop",
        )

  def test_gradients_differ_across_layers(self):
    """A regression guard for the silent-zero failure mode.

    If the carry gradient is severed, every layer's weight gradient reflects
    only its own contribution and the per-layer grads collapse toward each
    other while the loss stays bit-exact. Distinct grads prove the chain ran.
    """
    model = _ToyModel(self.n_layers, self.dim)
    scan_model.apply_scan_over_layers(model)
    model.train()
    model.layers(self._inputs()).sum().backward()

    stacked = model.layers.get_parameter(scan_model._mangle("wi.weight"))  # pylint: disable=protected-access
    for i in range(1, self.n_layers):
      self.assertFalse(
          torch.allclose(stacked.grad[0], stacked.grad[i]),
          f"layer 0 and layer {i} have identical gradients -- the inter-layer"
          " chain is severed (see the carry-requires-grad trap)",
      )

  def test_carry_without_grad_raises_in_training(self):
    """Fail loudly rather than return zeros for the carry gradient."""
    model = _ToyModel(self.n_layers, self.dim)
    scan_model.apply_scan_over_layers(model)
    model.train()
    h = torch.randn(self.batch, self.seq, self.dim, requires_grad=False)
    with self.assertRaisesRegex(RuntimeError, "does not require grad"):
      model.layers(h)


class ScanGraphTest(parameterized.TestCase):
  """The structural payoff: one traced body, independent of depth."""

  def _trace(self, n_layers: int, dim: int = 8):
    from torch.fx.experimental.proxy_tensor import make_fx  # pylint: disable=g-import-not-at-top

    torch.manual_seed(0)
    model = _ToyModel(n_layers, dim)
    scan_model.apply_scan_over_layers(model)
    model.eval()
    h = torch.randn(2, 3, dim)
    # "real", not "fake": the stack holds real parameters, and tracing the HOP
    # under a second FakeTensorMode trips "Mixing fake modes NYI".
    return make_fx(model.layers, tracing_mode="real")(h)

  def test_traced_graph_contains_a_scan_op(self):
    graph = self._trace(4)
    scan_nodes = [
        n
        for n in graph.graph.nodes
        if "scan" in str(n.target).lower() and n.op == "call_function"
    ]
    self.assertLen(
        scan_nodes, 1, f"expected exactly one scan op, got {scan_nodes}"
    )

  def test_graph_size_is_independent_of_depth(self):
    """The compile-time win: 4 layers and 16 layers must trace to one body.

    If scan silently unrolled, node count would scale with depth. This is the
    cheap local proxy for "one `stablehlo.while` instead of N inlined blocks".
    """
    small = len(self._trace(4).graph.nodes)
    large = len(self._trace(16).graph.nodes)
    self.assertEqual(
        small,
        large,
        f"graph grew {small} -> {large} nodes when depth went 4 -> 16;"
        " the layer body is being unrolled, not scanned",
    )


if __name__ == "__main__":
  absltest.main()
