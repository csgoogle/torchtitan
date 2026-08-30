# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the ambient-AC route into the scan body, and its min-cut shim.

Recomputing the in-body all-gather is the half of in-body FSDP that keeps the
memory win. Moving the gather into the scan body is not enough on its own: the
min-cut partitioner will save the *gathered* per-layer weight as a scan `ys`
residual, and at LLaMA-3 70B the 136.9 GB comes straight back, just
materialized layer-by-layer.

The mechanism is two pieces, one per test class here:

  * `ScannedLayerStack.forward` opens an `fx_traceback.current_meta` window
    around its `scan(...)` call. Every node the tracer creates while the window
    is open inherits the tag, and that includes the nodes of the HOP body's
    joint graph -- which is what `torch.utils.checkpoint` cannot reach, because
    the HOP re-traces the body through `materialize_as_graph` and drops its
    tags before the body's partitioner ever runs.
  * `install_scan_body_ac_shim` strips the tag from the two node classes the
    window necessarily over-tags and upstream's dispatch-mode tagger never
    touches: `placeholder` nodes (min-cut raises `NetworkXError` on them) and
    the outer `scan` node itself (min-cut would otherwise recompute the whole
    layer loop in backward).

`install_*` mutates a module-global in `torch._functorch.partitioners`, so
every test here installs through `_installed` and asserts the process is left
clean.
"""

import contextlib

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch._functorch.partitioners as partitioners
from torch._functorch.aot_autograd import aot_export_joint_simple
from torch.fx import traceback as fx_traceback
from torch._higher_order_ops.scan import scan, scan_op
from torchtitan.experiments.tpu.scan import fsdp as scan_fsdp

_MUST_RECOMPUTE = partitioners.CheckpointPolicy.MUST_RECOMPUTE


@contextlib.contextmanager
def _installed():
  """Installs the shim and guarantees the stock partitioner is restored."""
  scan_fsdp.install_scan_body_ac_shim()
  try:
    yield partitioners.min_cut_rematerialization_partition
  finally:
    scan_fsdp.uninstall_scan_body_ac_shim()


def _run_shim(graph: torch.fx.Graph):
  """Drives the installed shim over `graph` and returns the surviving tags.

  The shim calls through to whatever `min_cut_rematerialization_partition` was
  before the install, so the real partitioner is swapped for a stub that just
  records the graph it was handed. That keeps the test on the shim's own
  behaviour -- which tags it strips -- instead of on min-cut's response to it.
  """
  module = torch.fx.GraphModule(torch.nn.Module(), graph)

  original = partitioners.min_cut_rematerialization_partition
  partitioners.min_cut_rematerialization_partition = (
      lambda joint_module, joint_inputs, **kwargs: (None, None)
  )
  try:
    scan_fsdp.install_scan_body_ac_shim()
    partitioners.min_cut_rematerialization_partition(module, ([], {}))
  finally:
    scan_fsdp.uninstall_scan_body_ac_shim()
    partitioners.min_cut_rematerialization_partition = original

  return {
      node.name
      for node in module.graph.nodes
      if node.meta.get("recompute") is _MUST_RECOMPUTE
  }


def _tag(node):
  node.meta["recompute"] = _MUST_RECOMPUTE
  node.meta["ac_graph_id"] = 0
  return node


class AmbientWindowTest(parameterized.TestCase):
  """The `current_meta` window reaches the nodes of a scan body's joint graph.

  This is the claim the whole mechanism rests on. It is asserted against a real
  `scan` HOP rather than a hand-built graph, because the interesting part is
  precisely that the tag survives the HOP's internal re-trace.
  """

  def _traced_body_tags(self, *, window: bool):
    """Joint-traces a `scan` under an optional ambient window; returns body tags.

    The trace is `aot_export_joint_simple`, not `torch.compile`: the window is
    Python that has to actually *run* while the tracer is live, and dynamo
    would trace the `preserve_node_meta` block itself instead of executing it.
    The real caller (`ScannedLayerStack.forward` under `aot_fx_trace`) is on
    the tracing path, which is the one reproduced here.
    """
    seen = {}

    original = partitioners.min_cut_rematerialization_partition

    def capture(joint_module, joint_inputs, **kwargs):
      seen.setdefault("tags", {
          node.name: node.meta.get("recompute")
          for node in joint_module.graph.nodes
      })
      return original(joint_module, joint_inputs, **kwargs)

    class Model(torch.nn.Module):

      def forward(self, carry, xs):
        def body(c, slice_):
          out = torch.tanh(c @ slice_)
          return out, out.new_zeros(())

        def run():
          # `aot_export_joint_simple` requires a list/tuple output.
          return (scan(body, carry, xs)[0].sum(),)

        if not window:
          return run()
        with fx_traceback.preserve_node_meta():
          saved = dict(fx_traceback.current_meta)
          fx_traceback.current_meta["recompute"] = _MUST_RECOMPUTE
          fx_traceback.current_meta["ac_graph_id"] = 0
          try:
            return run()
          finally:
            fx_traceback.current_meta.clear()
            fx_traceback.current_meta.update(saved)

    carry = torch.randn(4, 4, requires_grad=True)
    xs = torch.randn(3, 4, 4, requires_grad=True)

    partitioners.min_cut_rematerialization_partition = capture
    try:
      scan_fsdp.install_scan_body_ac_shim()
      aot_export_joint_simple(Model(), (carry, xs), trace_joint=True)
    finally:
      scan_fsdp.uninstall_scan_body_ac_shim()
      partitioners.min_cut_rematerialization_partition = original

    return seen.get("tags", {})

  def test_window_tags_the_body_compute(self):
    tags = self._traced_body_tags(window=True)
    tanh = [name for name in tags if "tanh" in name]
    self.assertNotEmpty(tanh, f"no tanh node in the body's joint graph: {tags}")
    for name in tanh:
      self.assertIs(tags[name], _MUST_RECOMPUTE)

  def test_without_the_window_the_body_is_untagged(self):
    tags = self._traced_body_tags(window=False)
    self.assertEmpty([n for n, tag in tags.items() if tag is _MUST_RECOMPUTE])

  def test_the_window_does_not_outlive_the_call(self):
    """A leaked window would tag every node traced later in the process."""
    with fx_traceback.preserve_node_meta():
      before = dict(fx_traceback.current_meta)
      self._traced_body_tags(window=True)
      self.assertEqual(dict(fx_traceback.current_meta), before)


class StripTest(parameterized.TestCase):
  """Which ambient tags the shim removes before handing min-cut the graph."""

  def test_placeholder_tag_is_stripped(self):
    """A graph input is not recomputable; min-cut raises `NetworkXError` on it."""
    graph = torch.fx.Graph()
    _tag(graph.placeholder("primals_1"))
    graph.output(())

    self.assertEmpty(_run_shim(graph))

  def test_compute_tag_is_kept(self):
    """The tag on real body compute is the whole point; it must survive."""
    graph = torch.fx.Graph()
    primal = graph.placeholder("primals_1")
    mul = graph.call_function(torch.ops.aten.mul.Tensor, (primal, 2))
    mul.name = "mul"
    _tag(mul)
    graph.output((mul,))

    self.assertEqual(_run_shim(graph), {"mul"})

  def test_scan_node_tag_is_stripped(self):
    """Left tagged, min-cut recomputes the entire layer loop in backward.

    The forward then runs twice per step, silently, and the saved-set metric
    *improves* while it happens -- which is why this is asserted rather than
    left to the audit log.
    """
    graph = torch.fx.Graph()
    primal = graph.placeholder("primals_1")
    scan_node = graph.call_function(scan_op, (None, [primal], [], ()))
    scan_node.name = "scan"
    _tag(scan_node)
    graph.output((scan_node,))

    self.assertEmpty(_run_shim(graph))

  def test_untagged_nodes_are_left_alone(self):
    graph = torch.fx.Graph()
    primal = graph.placeholder("primals_1")
    mul = graph.call_function(torch.ops.aten.mul.Tensor, (primal, 2))
    mul.name = "mul"
    graph.output((mul,))

    self.assertEmpty(_run_shim(graph))


class LifecycleTest(parameterized.TestCase):
  """Install / uninstall against a process-global in `torch._functorch`."""

  def test_install_replaces_and_uninstall_restores_the_same_object(self):
    stock = partitioners.min_cut_rematerialization_partition
    with _installed() as wrapped:
      self.assertIsNot(wrapped, stock)
    self.assertIs(partitioners.min_cut_rematerialization_partition, stock)

  def test_install_is_idempotent(self):
    with _installed() as first:
      scan_fsdp.install_scan_body_ac_shim()
      self.assertIs(partitioners.min_cut_rematerialization_partition, first)

  def test_uninstall_without_install_is_a_no_op(self):
    stock = partitioners.min_cut_rematerialization_partition
    scan_fsdp.uninstall_scan_body_ac_shim()
    self.assertIs(partitioners.min_cut_rematerialization_partition, stock)

  def test_a_failed_install_does_not_leak_the_wrapper(self):
    """`uninstall` in a `finally` is the only thing between a raise and a leak."""
    stock = partitioners.min_cut_rematerialization_partition
    with self.assertRaises(RuntimeError):
      with _installed():
        raise RuntimeError("boom")
    self.assertIs(partitioners.min_cut_rematerialization_partition, stock)


if __name__ == "__main__":
  absltest.main()
