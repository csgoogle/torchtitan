# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Activation checkpointing for a scan body.

Moving the FSDP all-gather into the scan body (see `layer_stack.py`) is only
half a fix. The min-cut partitioner then saves the *gathered* per-layer weight
as a scan `ys` residual, so all 136.9 GB of LLaMA-3 70B weights come back --
stacked into one residual instead of N separate gathers, but just as resident.
The gather has to be recomputed in backward, and so do the block's activations
if the stack is to be fully checkpointed.

`torch.utils.checkpoint` around the body does not do it: the scan HOP re-traces
the body through `materialize_as_graph`, and the re-trace runs outside the
checkpoint context, so nothing tags the nodes. What *does* survive is the
tracer's ambient metadata channel. `torch/fx/proxy.py` copies every key of
`fx_traceback.current_meta` that is listed in `_COPY_META_FIELDS` -- `recompute`
and `ac_graph_id` among them -- onto every node the tracer creates. Opening that
window around the `scan` call therefore tags the body's joint-graph nodes too,
and the body's own partitioner honours the tag.

Two details make it work, and each is silent when wrong:

  * `preserve_node_meta()` is required. The copy in `proxy.py` is gated on
    `should_preserve_node_meta()`; without the context manager `current_meta` is
    populated and delivers nothing.
  * The window must not outlive the `scan` call, or every node traced afterwards
    inherits the policy. `body_checkpointed` saves and restores the whole dict
    rather than deleting the two keys it set.

The window is open while the *outer* tracer creates the `scan` node itself, and
while it creates the body's placeholders, and neither of those can carry a
`recompute` tag: `must_recompute()` upstream does not exclude placeholders, so
min-cut walks to a node that is not in its graph and raises `NetworkXError`, and
a tagged `scan` node makes the partitioner recompute the entire layer loop in
backward -- silently, while the saved-set metric it is judged by improves.
`install()` wraps `min_cut_rematerialization_partition` to strip the tag off
exactly those nodes before delegating.

The upstream ask that would delete this file is for the scan HOP to preserve
`node.meta["recompute"]` across `materialize_as_graph`; then
`torch.utils.checkpoint` would just work and neither half would be needed.

On recomputing a collective: upstream's *Note [Recomputing collectives in the
partitioner]* bans it because a pass that reorders, deletes, or duplicates a
collective inconsistently across ranks deadlocks. A user tag overrides the ban
(`ban_recomputation_if_allowed` honours it first), which is safe here because
every rank compiles the same SPMD graph, so the duplication is identical on all
of them.
"""

import contextlib
from typing import Iterator

import torch.fx.traceback as fx_traceback
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.tools.logging import logger

_original_min_cut = None


@contextlib.contextmanager
def body_checkpointed(enabled: bool = True) -> Iterator[None]:
  """Tags every node traced inside the block `MUST_RECOMPUTE`.

  Wrap the `scan` call with this to checkpoint the scan body. `install()` must
  be active for the resulting graph to partition.

  Args:
    enabled: `False` makes this a no-op, so a caller can pass its flag straight
      through instead of branching around the `with`.
  """
  if not enabled:
    yield
    return
  with fx_traceback.preserve_node_meta():
    saved = dict(fx_traceback.current_meta)
    fx_traceback.current_meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
    fx_traceback.current_meta["ac_graph_id"] = 0
    try:
      yield
    finally:
      fx_traceback.current_meta.clear()
      fx_traceback.current_meta.update(saved)


def install() -> None:
  """Wraps the partitioner to drop tags `body_checkpointed` cannot place.

  Call once before compiling; idempotent. See the module docstring for the two
  node kinds this strips and what each one breaks.
  """
  global _original_min_cut
  if _original_min_cut is not None:
    return

  import torch._functorch.partitioners as partitioners  # pylint: disable=g-import-not-at-top
  # `torch._higher_order_ops.scan` as an attribute is the exported `scan`
  # *function*, which shadows the submodule of the same name, so the class has
  # to come in by `from`-import rather than by attribute access off the package.
  from torch._higher_order_ops.scan import ScanOp  # pylint: disable=g-import-not-at-top

  _original_min_cut = partitioners.min_cut_rematerialization_partition

  def strip_untaggable_then_partition(joint_module, joint_inputs, **kwargs):
    stripped = 0
    for node in joint_module.graph.nodes:
      if "recompute" not in node.meta:
        continue
      if node.op == "placeholder" or isinstance(node.target, ScanOp):
        node.meta.pop("recompute", None)
        stripped += 1
    logger.info(
        "scan recompute: stripped ambient tags from %d placeholder/scan"
        " node(s) of %d before min-cut.",
        stripped,
        len(joint_module.graph.nodes),
    )
    return _original_min_cut(joint_module, joint_inputs, **kwargs)

  partitioners.min_cut_rematerialization_partition = (
      strip_untaggable_then_partition
  )
  logger.info("scan recompute: wrapped min_cut_rematerialization_partition.")


def uninstall() -> None:
  """Restores the stock partitioner. Idempotent.

  `install` patches a module-level function, which is process-global: without
  this a test that installs it leaks the wrapper into every later compile in
  the same process.
  """
  global _original_min_cut
  if _original_min_cut is None:
    return

  import torch._functorch.partitioners as partitioners  # pylint: disable=g-import-not-at-top

  partitioners.min_cut_rematerialization_partition = _original_min_cut
  _original_min_cut = None
  logger.info("scan recompute: restored the stock min-cut partitioner.")


__all__ = ["body_checkpointed", "install", "uninstall"]
