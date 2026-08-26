# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for `scan/fsdp.py`'s forced recompute of the in-body all-gather.

This is the half of in-body FSDP that keeps the memory win. Moving the gather
into the scan body is not enough on its own: the min-cut partitioner will save
the *gathered* per-layer weight as a scan `ys` residual, and at LLaMA-3 70B the
136.9 GB comes straight back, just materialized layer-by-layer.
`install_scan_collective_recompute` prevents that by tagging the gather --
and every view-like descendant of it -- `MUST_RECOMPUTE` before handing the
joint graph to the stock partitioner.

The tagging pass is pure fx-graph rewriting, so it is tested here directly on
hand-built joint graphs rather than through a real trace: the interesting cases
(a `t_copy` standing in for a tagged gather, a `reduce_scatter` that must be
left alone, a backward-region node that must not be tagged) are all awkward to
provoke from a real model and trivial to state as a graph.

`install_*` mutates a module-global in `torch._functorch.partitioners`, so every
test here installs through `_installed` and asserts the process is left clean.
"""

import contextlib

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch._functorch.partitioners as partitioners
from torchtitan.experiments.tpu.scan import fsdp as scan_fsdp

_MUST_RECOMPUTE = partitioners.CheckpointPolicy.MUST_RECOMPUTE


@contextlib.contextmanager
def _installed(recompute_body: bool = False):
  """Installs the wrapper and guarantees the stock partitioner is restored."""
  scan_fsdp.install_scan_collective_recompute(recompute_body=recompute_body)
  try:
    yield partitioners.min_cut_rematerialization_partition
  finally:
    scan_fsdp.uninstall_scan_collective_recompute()


def _run_tagging_pass(graph: torch.fx.Graph, *, recompute_body: bool = False):
  """Drives the installed wrapper over `graph` and returns the tagged names.

  The wrapper calls through to whatever `min_cut_rematerialization_partition`
  was before the install, so the real partitioner is swapped for a stub that
  just records the graph it was handed. That keeps the test on the pass's own
  behaviour -- which nodes get tagged -- instead of on min-cut's response to it.
  """
  module = torch.fx.GraphModule(torch.nn.Module(), graph)
  seen = {}

  original = partitioners.min_cut_rematerialization_partition
  partitioners.min_cut_rematerialization_partition = (
      lambda joint_module, joint_inputs, **kwargs: seen.setdefault(
          "module", joint_module
      )
      and (None, None)
  )
  try:
    scan_fsdp.install_scan_collective_recompute(recompute_body=recompute_body)
    partitioners.min_cut_rematerialization_partition(module, ([], {}))
  finally:
    scan_fsdp.uninstall_scan_collective_recompute()
    partitioners.min_cut_rematerialization_partition = original

  return {
      node.name
      for node in module.graph.nodes
      if node.meta.get("recompute") is _MUST_RECOMPUTE
  }


def _all_gather(graph: torch.fx.Graph, shard):
  """An `all_gather_into_tensor` + its `wait_tensor`, as traced in a scan body."""
  gathered = graph.call_function(
      torch.ops._c10d_functional.all_gather_into_tensor.default,
      (shard, 4, "0"),
  )
  gathered.name = "all_gather"
  wait = graph.call_function(
      torch.ops._c10d_functional.wait_tensor.default, (gathered,)
  )
  wait.name = "wait"
  return wait


class TaggingTest(parameterized.TestCase):
  """Which nodes the pass marks `MUST_RECOMPUTE`."""

  def test_gather_and_wait_are_tagged(self):
    graph = torch.fx.Graph()
    shard = graph.placeholder("shard")
    wait = _all_gather(graph, shard)
    graph.output((wait,))

    self.assertEqual(_run_tagging_pass(graph), {"all_gather", "wait"})

  def test_view_like_descendant_is_tagged_too(self):
    """The `t_copy` case: min-cut would otherwise save the transpose instead.

    Tagging the gather alone leaves the partitioner free to save
    `t_copy(gathered_weight)` -- a full-size tensor of exactly the shape the
    tag was meant to keep out of the saved set -- and it will, because a
    transpose is cheap to save and an all-gather is expensive to recompute.
    """
    graph = torch.fx.Graph()
    shard = graph.placeholder("shard")
    wait = _all_gather(graph, shard)
    transposed = graph.call_function(torch.ops.aten.t_copy.default, (wait,))
    transposed.name = "transposed"
    graph.output((transposed,))

    self.assertIn("transposed", _run_tagging_pass(graph))

  def test_view_like_closure_reaches_a_chain(self):
    """One forward sweep must reach a descendant several views deep."""
    graph = torch.fx.Graph()
    shard = graph.placeholder("shard")
    node = _all_gather(graph, shard)
    for i, op in enumerate((
        torch.ops.aten.t_copy.default,
        torch.ops.aten.view_copy.default,
        torch.ops.aten._to_copy.default,
    )):
      node = graph.call_function(op, (node,))
      node.name = f"view_{i}"
    graph.output((node,))

    tagged = _run_tagging_pass(graph)
    self.assertContainsSubset({"view_0", "view_1", "view_2"}, tagged)

  def test_compute_consumer_is_not_tagged(self):
    """A matmul on the gathered weight is real work; recomputing it is not free."""
    graph = torch.fx.Graph()
    shard = graph.placeholder("shard")
    activation = graph.placeholder("activation")
    wait = _all_gather(graph, shard)
    matmul = graph.call_function(torch.ops.aten.mm.default, (activation, wait))
    matmul.name = "matmul"
    graph.output((matmul,))

    self.assertNotIn("matmul", _run_tagging_pass(graph))

  def test_reduce_scatter_is_left_alone(self):
    """It already lives in the backward region; tagging it only confuses the cut."""
    graph = torch.fx.Graph()
    grad = graph.placeholder("grad")
    scattered = graph.call_function(
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        (grad, "sum", 4, "0"),
    )
    scattered.name = "reduce_scatter"
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default, (scattered,)
    )
    wait.name = "rs_wait"
    graph.output((wait,))

    self.assertEmpty(_run_tagging_pass(graph))

  def test_wait_on_an_untagged_collective_is_not_tagged(self):
    """`wait_tensor` is tagged for what it waits on, not for being a wait."""
    graph = torch.fx.Graph()
    shard = graph.placeholder("shard")
    reduced = graph.call_function(
        torch.ops._c10d_functional.all_reduce.default, (shard, "sum", "0")
    )
    reduced.name = "all_reduce"
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default, (reduced,)
    )
    wait.name = "ar_wait"
    graph.output((wait,))

    self.assertEmpty(_run_tagging_pass(graph))


class RecomputeBodyTest(parameterized.TestCase):
  """`recompute_body=True` -- full activation checkpointing of the scan body.

  With it, the only thing that crosses the cut is the loop carry, so the `ys`
  residual stops growing with the width of the block. `torch.utils.checkpoint`
  cannot express this: the HOP re-traces the body through
  `materialize_as_graph` and drops the recompute tags before this partitioner
  ever runs, silently.
  """

  def _joint_graph(self):
    """A joint graph: two forward computes, one backward compute off `tangents`."""
    graph = torch.fx.Graph()
    primal = graph.placeholder("primals_1")
    tangent = graph.placeholder("tangents_1")

    fwd_a = graph.call_function(torch.ops.aten.mul.Tensor, (primal, 2))
    fwd_a.name = "fwd_a"
    fwd_b = graph.call_function(torch.ops.aten.sin.default, (fwd_a,))
    fwd_b.name = "fwd_b"
    bwd = graph.call_function(torch.ops.aten.mul.Tensor, (tangent, fwd_a))
    bwd.name = "bwd"
    graph.output((fwd_b, bwd))

    for node in graph.nodes:
      if node.op == "call_function":
        node.meta["val"] = torch.empty(1)
    return graph

  def test_off_by_default_leaves_plain_compute_untagged(self):
    self.assertEmpty(_run_tagging_pass(self._joint_graph()))

  def test_on_tags_the_forward_region(self):
    tagged = _run_tagging_pass(self._joint_graph(), recompute_body=True)
    self.assertContainsSubset({"fwd_a", "fwd_b"}, tagged)

  def test_on_does_not_tag_the_backward_region(self):
    """Tagging a backward node is inert by accident, and only by accident.

    Min-cut resolves `required_bw_nodes` before it consults the policy, so a
    tagged backward node is usually ignored -- but a tagged nondeterministic op
    there would be rewritten through `run_with_rng_state`.
    """
    tagged = _run_tagging_pass(self._joint_graph(), recompute_body=True)
    self.assertNotIn("bwd", tagged)

  def test_on_still_tags_collectives(self):
    """The two mechanisms compose; `recompute_body` does not replace the gather tag."""
    graph = self._joint_graph()
    output = next(n for n in graph.nodes if n.op == "output")
    with graph.inserting_before(output):
      shard = graph.call_function(torch.ops.aten.clone.default, (
          next(n for n in graph.nodes if n.name == "fwd_a"),
      ))
      shard.meta["val"] = torch.empty(1)
      wait = _all_gather(graph, shard)
      wait.meta["val"] = torch.empty(1)

    tagged = _run_tagging_pass(graph, recompute_body=True)
    self.assertContainsSubset({"all_gather", "wait"}, tagged)


class LifecycleTest(parameterized.TestCase):
  """Install / uninstall against a process-global in `torch._functorch`."""

  def test_install_replaces_and_uninstall_restores_the_same_object(self):
    stock = partitioners.min_cut_rematerialization_partition
    with _installed() as wrapped:
      self.assertIsNot(wrapped, stock)
    self.assertIs(partitioners.min_cut_rematerialization_partition, stock)

  def test_install_is_idempotent(self):
    with _installed() as first:
      scan_fsdp.install_scan_collective_recompute(recompute_body=False)
      self.assertIs(partitioners.min_cut_rematerialization_partition, first)

  def test_uninstall_without_install_is_a_no_op(self):
    stock = partitioners.min_cut_rematerialization_partition
    scan_fsdp.uninstall_scan_collective_recompute()
    self.assertIs(partitioners.min_cut_rematerialization_partition, stock)

  def test_second_install_with_a_different_policy_warns_and_is_ignored(self):
    """The wrapper closes over `recompute_body`, so the first install wins.

    Silently handing the second caller a different activation-memory policy
    from the one it asked for is the failure this warning exists to make loud.
    Two models in one process with different `memory_policy` settings is the
    way to hit it.
    """
    with _installed(recompute_body=False):
      with self.assertLogs(level="WARNING") as logs:
        scan_fsdp.install_scan_collective_recompute(recompute_body=True)
      self.assertRegex("\n".join(logs.output), "already installed")

  def test_a_failed_install_does_not_leak_the_wrapper(self):
    """`uninstall` in a `finally` is the only thing between a raise and a leak."""
    stock = partitioners.min_cut_rematerialization_partition
    with self.assertRaises(RuntimeError):
      with _installed():
        raise RuntimeError("boom")
    self.assertIs(partitioners.min_cut_rematerialization_partition, stock)


if __name__ == "__main__":
  absltest.main()
