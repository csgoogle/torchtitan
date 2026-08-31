# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Neither flavour of `torch.utils.checkpoint` tags a scan body.

This is the measurement behind upstream ask 1 in `scan/README.md`, and the
reason `scan/recompute.py` exists at all. Both flavours fail, for two unrelated
reasons:

  * Vanilla AC is tagged by Dynamo (`TagActivationCheckpoint.tag_nodes`), and
    the HOP re-traces its body under `@torch._dynamo.disable(recursive=True)`.
    Dynamo never sees the body, so the tagging pass never runs.
  * Selective AC does not go through Dynamo -- it tags from a
    `TorchDispatchMode` that writes `fx_traceback.current_meta` -- so it looked
    like it might survive where vanilla AC cannot. It does not: node-meta
    preservation is off for the body re-trace, so the writes go nowhere. Turning
    preservation on from inside the body is rejected by the HOP as an unsafe
    side effect, which closes the last way around it.

Both are silent. You get min-cut's saved set with no warning and no error.

Kept as a test so the ask stays honest: if a future PyTorch fixes either
flavour, the corresponding `tags_nothing` test fails and `scan/recompute.py`
can be deleted.

This runs the *real* tracing path -- plain eager `scan(...).sum().backward()`
calls the HOP's own autograd, which is what reaches `create_partitioned_graph`
-> `materialize_as_graph`. `make_fx` alone does not exercise it. One chip is
enough; there is no collective here.

It is a separate file from `test_scan_recompute.py` on purpose: these tests
initialise the PjRt client in the parent process, which breaks the spawned
8-chip workers of `GatheredWeightIsNotSaved` if the two share a process.

    NPROC=1 ./run_in_container.sh \
        torchtitan.experiments.tpu.tests.test_scan_body_checkpoint
"""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch._functorch.partitioners as partitioners
from torch._higher_order_ops.scan import scan
from torch._higher_order_ops.scan import ScanOp
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import create_selective_checkpoint_contexts

from torchtitan.experiments.tpu.scan import recompute as scan_recompute

_MUST_RECOMPUTE = partitioners.CheckpointPolicy.MUST_RECOMPUTE

BATCH, DIM, LAYERS = 2, 4, 3


def _always_recompute(ctx, op, *args, **kwargs):
  del ctx, op, args, kwargs
  return _MUST_RECOMPUTE


def _census(joint) -> dict:
  """Counts `recompute`-tagged nodes per node kind, over every kind present."""
  tagged = {}
  for node in joint.graph.nodes:
    kind = "scan" if isinstance(node.target, ScanOp) else node.op
    tagged.setdefault(kind, 0)
    if node.meta.get("recompute") is not None:
      tagged[kind] += 1
  return tagged


def _tag_census(arm: str) -> dict:
  """Traces a tiny scan through the real HOP path; returns the tag census.

  Args:
    arm: `control` for an untagged body, `ambient` for what `recompute.py` does
      today, `sac` for `torch.utils.checkpoint` with a selective policy entered
      inside the body.
  """
  torch.manual_seed(0)
  weights = torch.randn(LAYERS, DIM, DIM, requires_grad=True)
  h = torch.randn(BATCH, DIM, requires_grad=True)

  def inner(carry, weight):
    return torch.tanh(carry @ weight) * 1.5

  def body(carry, weight):
    if arm == "sac":
      return_ = checkpoint(
          inner,
          carry,
          weight,
          use_reentrant=False,
          context_fn=functools.partial(
              create_selective_checkpoint_contexts, _always_recompute
          ),
      )
    else:
      return_ = inner(carry, weight)
    # The `ys` must not alias the carry, or the HOP rejects the body.
    return return_, return_.clone()

  seen = []
  stock = partitioners.min_cut_rematerialization_partition

  def census_then_strip_then_partition(joint, joint_inputs, **kwargs):
    # Census before the strip. The strip still runs, so every arm reaches
    # min-cut and the comparison is between tag sets rather than between a
    # graph and a crash: a tagged placeholder makes min-cut walk to a node that
    # is not in its graph and raise.
    seen.append(_census(joint))
    for node in joint.graph.nodes:
      if node.op == "placeholder" or isinstance(node.target, ScanOp):
        node.meta.pop("recompute", None)
    return stock(joint, joint_inputs, **kwargs)

  partitioners.min_cut_rematerialization_partition = (
      census_then_strip_then_partition
  )
  try:
    with scan_recompute.body_checkpointed(arm == "ambient"):
      carry, _ = scan(body, h, weights)
    carry.sum().backward()
  finally:
    partitioners.min_cut_rematerialization_partition = stock

  assert seen, (
      f"{arm}: the scan body was never partitioned, so this arm measured"
      " nothing -- the capture hook is in the wrong place"
  )
  return seen[0]


class CheckpointInsideTheBodyTest(parameterized.TestCase):
  """What does and does not reach a HOP body's joint graph."""

  def test_an_untagged_body_is_the_baseline(self):
    self.assertEqual(_tag_census("control")["call_function"], 0)

  def test_the_ambient_window_tags_the_body(self):
    census = _tag_census("ambient")
    self.assertGreater(census["call_function"], 0)
    # The node kind `recompute.install()` exists to strip. Tagging it is not a
    # bug in the window: the window is necessarily open while the outer tracer
    # creates the body's placeholders, so it cannot avoid them.
    self.assertGreater(census["placeholder"], 0)

  def test_selective_checkpointing_inside_the_body_tags_nothing(self):
    self.assertEqual(_tag_census("sac")["call_function"], 0)


if __name__ == "__main__":
  absltest.main()
