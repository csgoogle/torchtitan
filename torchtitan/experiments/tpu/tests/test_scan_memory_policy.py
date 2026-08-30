# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the graph-trainer SAC pass under scan-over-layers.

The graph trainer forces `MUST_SAVE` at layer boundaries so that recomputing a
node never drags in the whole preceding layer. It finds those boundaries by
parsing an integer layer index out of each node's `module_fqn` (`layers.7.…`).

Scan replaces the N per-layer modules with a single `layer_template`, so that
integer no longer exists anywhere in the graph. Every node under the stack
reports `_NOT_IN_LAYERS` and the boundary pass can never fire -- with no error
and no log line saying so. Activation checkpointing under scan is not therefore
absent: it comes from `ScannedLayerStack.set_recompute_body`, whose ambient
`current_meta` window tags the body for the HOP's own partitioner (see
`test_scan_body_ac_shim.py`). What is inert is this pass's layer-
boundary `MUST_SAVE`, so the two mechanisms must not be assumed to compose.

Note that widening the comparison from `>` to `!=` does not fix this: both
sides of a scan boundary report the same sentinel, so both spellings are false.
The missing layer index is the defect, not the operator.

`SacUnderScanTest.test_boundary_saves_fire_under_scanned_fqns` is an
`expectedFailure`: it asserts the behaviour we want and is expected to fail
until the heuristic stops keying on FQN integers. If it ever passes
unexpectedly, the defect is fixed and the decorator should come off.
"""

import unittest

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torchtitan.experiments.graph_trainer import common_utils


def _node_with_fqn(graph: torch.fx.Graph, name: str, fqn: str) -> torch.fx.Node:
  """A placeholder node carrying the `module_fqn` annotation the pass reads."""
  node = graph.placeholder(name)
  node.meta["custom"] = {common_utils._MODULE_FQN: fqn}  # pylint: disable=protected-access
  return node


class LayerIdTest(parameterized.TestCase):
  """`_get_layer_id` is the function the boundary heuristic rests on."""

  def _layer_id(self, fqn: str) -> int:
    graph = torch.fx.Graph()
    return common_utils._get_layer_id(_node_with_fqn(graph, "x", fqn))  # pylint: disable=protected-access

  @parameterized.named_parameters(
      ("layer_0", "layers.0.attention.wq", 0),
      ("layer_7", "layers.7.feed_forward.w1", 7),
      ("layer_31", "layers.31.attention_norm", 31),
  )
  def test_unrolled_fqns_yield_the_layer_index(self, fqn, want):
    self.assertEqual(self._layer_id(fqn), want)

  @parameterized.named_parameters(
      ("embeddings", "tok_embeddings"),
      ("final_norm", "norm"),
      ("output", "output"),
  )
  def test_non_layer_fqns_are_not_in_layers(self, fqn):
    self.assertEqual(self._layer_id(fqn), common_utils._NOT_IN_LAYERS)  # pylint: disable=protected-access

  @parameterized.named_parameters(
      ("scan_stack", "layers.scan"),
      ("scan_template", "layers.scan.attention.wq"),
      ("bare_layers", "layers"),
  )
  def test_scanned_fqns_lose_the_layer_index(self, fqn):
    """Characterizes the defect: no integer, so no layer identity.

    This is the mechanism -- `int("scan")` raises, the function falls through
    to `_NOT_IN_LAYERS`, and every scanned node becomes indistinguishable.
    """
    self.assertEqual(self._layer_id(fqn), common_utils._NOT_IN_LAYERS)  # pylint: disable=protected-access


class SacUnderScanTest(parameterized.TestCase):
  """The consequence: the boundary pass cannot fire under scan."""

  def _boundary_pairs(self, producer_fqn: str, consumer_fqn: str) -> int:
    """Counts producer->consumer pairs the boundary rule would force to save.

    Mirrors the Pass-2 predicate in `memory_policy.py`: a recomputable node
    whose user sits in a strictly higher layer.
    """
    graph = torch.fx.Graph()
    producer = _node_with_fqn(graph, "producer", producer_fqn)
    consumer = _node_with_fqn(graph, "consumer", consumer_fqn)
    return int(
        common_utils._get_layer_id(consumer)  # pylint: disable=protected-access
        > common_utils._get_layer_id(producer)  # pylint: disable=protected-access
    )

  def test_boundary_saves_fire_under_unrolled_fqns(self):
    """The unrolled model: layer 3 feeding layer 4 is a boundary."""
    self.assertEqual(
        self._boundary_pairs("layers.3.feed_forward.w2", "layers.4.attention.wq"),
        1,
    )

  @unittest.expectedFailure
  def test_boundary_saves_fire_under_scanned_fqns(self):
    """EXPECTED FAILURE -- documents the live defect.

    Two consecutive scan iterations are a genuine layer boundary, but both
    carry the same template FQN, so the strict `>` comparison is never true and
    the pass forces zero saves. Observed on every scan+graph-trainer run as
    `Layer non-layer: 0 MUST_SAVE, 169 RECOMPUTE`.

    Fixing this means giving the pass a scan-aware notion of a boundary (the
    carry edge of the HOP) rather than an integer parsed from an FQN.
    """
    self.assertEqual(
        self._boundary_pairs("layers.scan.feed_forward.w2", "layers.scan.attention.wq"),
        1,
    )

  def test_scan_boundary_currently_forces_no_saves(self):
    """The same fact, asserted positively, so the suite stays green and honest.

    Delete this alongside the `expectedFailure` above once the defect is fixed.
    """
    self.assertEqual(
        self._boundary_pairs("layers.scan.feed_forward.w2", "layers.scan.attention.wq"),
        0,
    )


if __name__ == "__main__":
  absltest.main()
