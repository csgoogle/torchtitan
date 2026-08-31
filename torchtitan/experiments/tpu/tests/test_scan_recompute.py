# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Moving the gather into the scan body only pays if it is recomputed.

Without `scan/recompute.py` the min-cut partitioner saves the *gathered*
per-layer weight as a scan `ys` residual and the memory win of `fsdp.py`
evaporates -- the full weight set is resident again, stacked into one residual
instead of N separate gathers.

`GatheredWeightIsNotSaved` is the measurement, on 8 real chips: it captures the
partition of the real scan body and looks at what the forward hands the
backward. It reproduces the exact failure the module exists to prevent, which is
subtler than "the gather is saved" -- what min-cut actually saves is
`t_copy(gathered_weight)`, because a transpose is free to save and an all-gather
is expensive to recompute.

The rest covers the two halves of the mechanism on their own: the ambient window
that tags the body, and the shim that strips the tag back off the two node kinds
that must not carry it.

    NPROC=8 ./run_in_container.sh \
        torchtitan.experiments.tpu.tests.test_scan_recompute --device=tpu
"""

import contextlib

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch._functorch.partitioners as partitioners
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu.scan import recompute as scan_recompute
from torchtitan.experiments.tpu.tests import scan_fixtures
from torchtitan.experiments.tpu.tests.scan_fixtures import BATCH, DIM, SEQ

_MUST_RECOMPUTE = partitioners.CheckpointPolicy.MUST_RECOMPUTE


@contextlib.contextmanager
def _installed():
  """Installs the shim and guarantees the stock partitioner comes back."""
  scan_recompute.install()
  try:
    yield partitioners.min_cut_rematerialization_partition
  finally:
    scan_recompute.uninstall()


@contextlib.contextmanager
def _capturing_partitions(saved: list):
  """Records the `(joint, fwd, bwd)` of every partition run inside the block."""
  inner = partitioners.min_cut_rematerialization_partition

  def capture(joint_module, joint_inputs, **kwargs):
    fwd, bwd = inner(joint_module, joint_inputs, **kwargs)
    saved.append((joint_module, fwd, bwd))
    return fwd, bwd

  partitioners.min_cut_rematerialization_partition = capture
  try:
    yield
  finally:
    partitioners.min_cut_rematerialization_partition = inner


# -- the measurement: what the scan body hands its backward ---------------------


def _saved_shapes(fwd) -> list:
  """Shapes of the forward graph's outputs -- what survives into backward."""
  output = next(node for node in fwd.graph.nodes if node.op == "output")
  shapes = []
  for arg in output.args[0]:
    val = getattr(arg, "meta", {}).get("val") if arg is not None else None
    if val is not None:
      shapes.append(tuple(val.shape))
  return shapes


def _run_and_capture(device, world_size, recompute: bool):
  """One sharded, scanned fwd+bwd; returns the saved shapes and the gradients."""
  _, stack = scan_fixtures.stacked(device, recompute_body=recompute)
  scan_fixtures.shard(stack, device, world_size)
  torch.manual_seed(1)
  h = torch.randn(BATCH, SEQ, DIM, device=device, requires_grad=True)

  partitions = []
  with contextlib.ExitStack() as ctx:
    if recompute:
      ctx.enter_context(_installed())
    ctx.enter_context(_capturing_partitions(partitions))
    stack(h).sum().backward()

  assert partitions, (
      "the scan body was never partitioned; the capture hook is in the wrong"
      " place and the rest of this test proves nothing"
  )
  saved = [shape for _, fwd, _ in partitions for shape in _saved_shapes(fwd)]
  grads = [p.grad.full_tensor().cpu() for p in stack.stacked_parameters()]
  return saved, grads


def _worker_gathered_weight_is_not_saved(device, rank, world_size):
  """The full weight must not survive the cut, in any spelling."""
  del rank
  full = (DIM, DIM)  # the gathered weight; a shard here is (DIM // ws, DIM).

  without, grads_without = _run_and_capture(device, world_size, recompute=False)
  assert full in without, (
      f"no {full} tensor in the saved set {without} even without recompute, so"
      " this test cannot show recompute removing one -- the toy model or the"
      " shard shape changed"
  )

  with_pass, grads_with = _run_and_capture(device, world_size, recompute=True)
  assert full not in with_pass, (
      f"the gathered {full} weight is still saved across the cut"
      f" {with_pass}; tagging the all-gather alone is not enough, min-cut"
      " saves a view of it instead"
  )

  # Recomputing changes when the gather runs, never what it computes.
  for i, (a, b) in enumerate(zip(grads_without, grads_with)):
    torch.testing.assert_close(
        a,
        b,
        rtol=1e-5,
        atol=1e-5,
        msg=lambda default, i=i: (
            f"stacked parameter {i}'s gradient changed when the gather was"
            f" recomputed\n{default}"
        ),
    )


class GatheredWeightIsNotSaved(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """The end-to-end payoff, on real chips."""

  def test_gathered_weight_is_not_saved(self):
    if self.num_devices < 2:
      self.skipTest("no gather happens at world size 1, so nothing to save")
    distributed.run_distributed(
        self.num_devices,
        self.accelerator_device_type,
        _worker_gathered_weight_is_not_saved,
    )


# -- half one: the ambient window ----------------------------------------------


def _tags(gm) -> dict:
  """Node name -> its `recompute` tag, for every node that carries one."""
  return {
      node.name: node.meta["recompute"]
      for node in gm.graph.nodes
      if "recompute" in node.meta
  }


def _traced(enabled: bool):
  """`(x * 2).sin()` traced with the window `enabled` or not."""
  with scan_recompute.body_checkpointed(enabled):
    return make_fx(lambda x: (x * 2).sin())(torch.randn(3))


class AmbientWindowTest(parameterized.TestCase):
  """`fx_traceback.current_meta` is the channel that reaches a HOP body."""

  def test_the_window_tags_every_node_traced_inside_it(self):
    tags = _tags(_traced(enabled=True))
    self.assertContainsSubset(["mul", "sin"], list(tags))
    self.assertEqual(set(tags.values()), {_MUST_RECOMPUTE})

  def test_disabled_tags_nothing(self):
    self.assertEmpty(_tags(_traced(enabled=False)))

  def test_the_window_does_not_outlive_the_block(self):
    """A leak here silently checkpoints the rest of the model, not just the body."""
    _traced(enabled=True)
    self.assertEmpty(_tags(_traced(enabled=False)))


# -- half two: the strip shim --------------------------------------------------


def _joint_graph_with_ambient_tags() -> torch.fx.GraphModule:
  """What the window leaves behind: a tagged placeholder, compute, scan node."""
  graph = torch.fx.Graph()
  primal = graph.placeholder("primals_1")
  compute = graph.call_function(torch.ops.aten.mul.Tensor, (primal, 2))
  compute.name = "compute"
  scanned = graph.call_function(torch.ops.higher_order.scan, (compute,))
  scanned.name = "scanned"
  graph.output((scanned,))
  for node in graph.nodes:
    node.meta["recompute"] = _MUST_RECOMPUTE
  return torch.fx.GraphModule(torch.nn.Module(), graph)


class StripShimTest(parameterized.TestCase):
  """Placeholders raise `NetworkXError`; a tagged scan node recompiles the loop."""

  def test_placeholder_and_scan_tags_are_stripped_and_compute_is_kept(self):
    gm = _joint_graph_with_ambient_tags()
    stock = partitioners.min_cut_rematerialization_partition
    partitioners.min_cut_rematerialization_partition = (
        lambda joint, inputs, **kwargs: (joint, None)
    )
    try:
      with _installed() as shim:
        shim(gm, [])
    finally:
      partitioners.min_cut_rematerialization_partition = stock
    tags = _tags(gm)
    self.assertNotIn("primals_1", tags)
    self.assertNotIn("scanned", tags)
    self.assertIn("compute", tags)


class LifecycleTest(parameterized.TestCase):
  """`install` patches a process-global in `torch._functorch.partitioners`."""

  def test_install_is_idempotent_and_uninstall_restores_the_stock_object(self):
    stock = partitioners.min_cut_rematerialization_partition
    with _installed() as wrapped:
      self.assertIsNot(wrapped, stock)
      scan_recompute.install()
      self.assertIs(partitioners.min_cut_rematerialization_partition, wrapped)
    self.assertIs(partitioners.min_cut_rematerialization_partition, stock)

  def test_a_raise_inside_the_block_does_not_leak_the_wrapper(self):
    stock = partitioners.min_cut_rematerialization_partition
    with self.assertRaises(RuntimeError):
      with _installed():
        raise RuntimeError("boom")
    self.assertIs(partitioners.min_cut_rematerialization_partition, stock)


if __name__ == "__main__":
  absltest.main()
