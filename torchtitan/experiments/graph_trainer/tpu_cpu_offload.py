# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offload the one activation a scan over decoder layers keeps alive.

Under FullAC the forward saves each layer's input, and the scan stacks those
saves into one ``[n_layers, batch, seq, dim]`` output fed straight back to the
backward scan. At 405B that is 33.8 GB at seq 8192 -- the whole of the model's
sequence-length dependence, and the reason seq 8192 does not fit at any batch
size.

Selecting it needs no layer names, just three structural facts:

  1. the node is in the forward half (``meta["autograd_backward"]`` is unset),
  2. its storage is read by a ``higher_order.scan`` node in the backward half,
  3. it is the largest such tensor.

(2) is the one that matters. With only "largest tensor the backward reads", the
debugmodel selects ``_log_softmax`` -- the loss logits, consumed immediately and
costing more to ship than to keep. Vocabulary beats depth at small scale and
loses to it at large, so a size-only rule silently does the wrong thing on
exactly the configs used to test it.

**The transfers go inside the loop bodies, not around the loop.** This is the
whole design, and getting it wrong is not a missed optimization but a 37 GB
regression: ``cpu_offload.apply_cpu_offload_pass`` places ``ao.reload``
immediately before the first backward consumer, and for the stack that consumer
*is the loop*, so all ``n_layers`` slices land on device before iteration 0.
Measured at 70B batch 4, seq 8192: 64.9 G without the policy, 101.92 G with it.

Placing them inside instead makes the forward body emit its slice already on
host, so the scan's ``dynamic-update-slice`` accumulates into a host buffer, and
the backward body's ``dynamic-slice`` reads one slice out of host and reloads
just that. The device never holds the whole stack. This is what ``nn.scan`` plus
a host-offload remat policy produces in JAX, and what ``torch_xla``'s ``scan``
produces when handed torchprime's ``remat_all_and_offload_these_inputs``
partitioner.

Two asymmetries in how the tracer builds the pair decide where the ops land,
and both are the opposite of the obvious guess:

  * The forward's saved activation is a ``ys`` output, so the body already
    emits one slice per iteration and the offload goes on that body output.
  * The backward does *not* receive it as ``xs``. It arrives in
    ``additional_inputs`` as the whole stack, and the body indexes it with the
    loop counter that ``xs`` carries. So the reload goes on the output of that
    in-body index, not on the placeholder -- reloading the placeholder would
    pull all ``n_layers`` slices back every iteration.
"""

import operator

import torch

# Registering ao::offload / ao::reload / ao::wait_tensor is an import side
# effect, and this module inserts them without going through cpu_offload.py.
import torch._functorch._activation_offloading.offload_ops  # noqa: F401
import torch.fx
from torch.fx import Node

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node
from torchtitan.experiments.graph_trainer.cpu_offload import (
    _can_offload_node,
    _has_recompute_consumer,
    _is_view,
    _tensor_bytes,
)
from torchtitan.tools.logging import logger

_SCAN_HOP = torch.ops.higher_order.scan


def _scan_body(gm: torch.fx.GraphModule, scan_node: Node) -> torch.fx.GraphModule:
    """The loop body of ``scan_node``.

    ``trace_scan`` registers the traced ``combine_fn`` as a submodule of the
    root graph and passes it as argument 0, so the body is reachable -- and
    mutable -- from a pass that only holds the parent graph.
    """
    return getattr(gm, scan_node.args[0].target)


def _find_stack_producer(node: Node) -> tuple[Node, int] | None:
    """The forward scan that produced ``node``, and which output it was.

    A scan node returns ``(*carry, *ys)`` and its body returns the same tuple,
    so the ``getitem`` index is already the body's output position.
    """
    seen: set[Node] = set()
    cur: object = node
    while isinstance(cur, Node) and cur not in seen:
        seen.add(cur)
        if cur.target is operator.getitem and isinstance(cur.args[0], Node):
            parent = cur.args[0]
            if parent.target is _SCAN_HOP and not _is_backward_node(parent):
                return parent, cur.args[1]
        if _is_view(cur) and cur.args and isinstance(cur.args[0], Node):
            cur = cur.args[0]
            continue
        return None
    return None


def _find_stack_consumer(node: Node) -> tuple[Node, int] | None:
    """The backward scan that reads ``node``, and its body placeholder index.

    ``scan``'s signature is ``(combine_fn, init, xs, additional_inputs)`` and
    the body's placeholders are ``(*carry, *xs_slices, *additional_inputs)``,
    so an operand's position in either list maps straight onto a placeholder.
    Both lists are searched because the stack arrives in ``additional_inputs``
    while ``xs`` carries only the loop counter.
    """
    seen: set[Node] = set()

    def walk(n: Node) -> tuple[Node, int] | None:
        for user in n.users:
            if user in seen:
                continue
            seen.add(user)
            if user.target is _SCAN_HOP and _is_backward_node(user):
                n_init, xs, additional = len(user.args[1]), user.args[2], user.args[3]
                for i, x in enumerate(xs):
                    if x is n:
                        return user, n_init + i
                for i, x in enumerate(additional):
                    if x is n:
                        return user, n_init + len(xs) + i
                continue
            if _is_view(user):
                found = walk(user)
                if found is not None:
                    return found
        return None

    return walk(node)


def _select_stacked_activation(gm: torch.fx.GraphModule) -> Node | None:
    """The largest forward tensor whose storage the backward scan reads."""
    best: Node | None = None
    best_bytes = 0
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        if not _can_offload_node(node):
            continue
        if _has_recompute_consumer(node):
            # AC will regenerate it; there is nothing to keep alive.
            continue
        if _find_stack_consumer(node) is None:
            continue
        nbytes = _tensor_bytes(node.meta["val"])
        if nbytes > best_bytes:
            best, best_bytes = node, nbytes
    return best


def _offload_body_output(body: torch.fx.GraphModule, index: int) -> None:
    """Emit body output ``index`` on host instead of on device."""
    output = next(n for n in body.graph.nodes if n.op == "output")
    outs = list(output.args[0])
    produced = outs[index]
    val = produced.meta["val"]

    with body.graph.inserting_after(produced):
        offload = body.graph.call_function(
            torch.ops.ao.offload.default, args=(produced,)
        )
        offload.meta["val"] = val.to("cpu")
    with body.graph.inserting_after(offload):
        # The keepalive argument holds the device storage until the D2H
        # completes. Inside the body the producer is its own last consumer.
        wait = body.graph.call_function(
            torch.ops.ao.wait_tensor.default, args=(offload, produced, produced)
        )
        wait.meta["val"] = offload.meta["val"]

    outs[index] = wait
    output.args = (tuple(outs),)
    body.graph.lint()
    body.recompile()


def _reload_body_slice(body: torch.fx.GraphModule, index: int) -> bool:
    """Reload the per-iteration slice of body placeholder ``index``.

    The placeholder is the whole stack, so the reload goes on the output of the
    op that indexes it, not on the placeholder itself. Returns False, without
    touching the graph, if the placeholder is used any other way: reloading the
    stack itself would defeat the point, and leaving offloading off and saying
    so beats doing that silently.
    """
    param = [n for n in body.graph.nodes if n.op == "placeholder"][index]
    users = list(param.users)
    if not users or any(
        u.meta["val"].numel() >= param.meta["val"].numel() for u in users
    ):
        return False

    for user in users:
        val = user.meta["val"]
        with body.graph.inserting_after(user):
            reload = body.graph.call_function(
                torch.ops.ao.reload.default,
                args=(user, val.device, list(val.size()), list(val.stride())),
            )
            reload.meta["val"] = val
        with body.graph.inserting_after(reload):
            wait = body.graph.call_function(
                torch.ops.ao.wait_tensor.default, args=(reload, user)
            )
            wait.meta["val"] = val
        for consumer in list(user.users):
            if consumer not in (reload, wait):
                consumer.replace_input_with(user, wait)
        user.meta["val"] = val.to("cpu")

    param.meta["val"] = param.meta["val"].to("cpu")
    body.graph.lint()
    body.recompile()
    return True


def offload_scan_activations(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Keep the scan's stacked layer input on host between the two loops."""
    node = _select_stacked_activation(gm)
    if node is None:
        # Distinguishing the two causes matters: no scan at all is a config
        # mistake the user can fix, while a scan the selector could not reach
        # is a bug here. Both otherwise look like a run that quietly did
        # nothing and reported a plausible number.
        scans = [n for n in gm.graph.nodes if n.target is _SCAN_HOP]
        logger.warning(
            "scan offload: nothing to offload -- this run uses the same memory "
            "as if offloading were off. Saw %d scan nodes, %d in the backward. "
            "Is --tpu-config.use-scan-layers set?",
            len(scans),
            sum(1 for n in scans if _is_backward_node(n)),
        )
        return gm

    producer = _find_stack_producer(node)
    if producer is None:
        logger.warning(
            "scan offload: %s is read by a backward scan but is not an output "
            "of a forward scan, so there is no loop body to offload from.",
            node.name,
        )
        return gm

    fwd_scan, out_index = producer
    bwd_scan, param_index = _find_stack_consumer(node)

    if not _reload_body_slice(_scan_body(gm, bwd_scan), param_index):
        logger.warning(
            "scan offload: the backward body uses %s whole rather than one "
            "slice per iteration, so offloading it would reload the entire "
            "stack every step. Leaving it on device.",
            node.name,
        )
        return gm
    _offload_body_output(_scan_body(gm, fwd_scan), out_index)

    # The stacked tensor now lives on host, so retype it and any view of it on
    # the way to the backward loop to keep ``meta["val"]`` agreeing with what
    # the backend will lower.
    cur: Node | None = node
    while cur is not None:
        cur.meta["val"] = cur.meta["val"].to("cpu")
        cur = next((u for u in cur.users if _is_view(u)), None)

    logger.info(
        "scan offload: %s %s -> host inside the loop bodies, %.2f GiB",
        node.name,
        tuple(node.meta["val"].shape),
        _tensor_bytes(node.meta["val"]) / (1024**3),
    )
    return gm
