"""Which values the scan body keeps across its forward/backward cut.

``LayerStack.run`` fully rematerializes the scan body: its backward recomputes
the whole forward from the body's inputs. That is what keeps the in-body FSDP
gather from being carried out of the loop, and it is done with an
``activation_memory_budget=0.0`` window around the ``scan`` call.

Budget 0.0 is absolute, though. The partitioner returns the inputs without
looking at any tag (``choose_saved_values_set``: ``if memory_budget == 0: return
node_info.inputs``), so *every* collective in the body is redone in backward.
For the MoE batch-wise aux loss that includes the global expert-count
all-reduce, which spans DP -- and so DCN on a multi-slice run. Its result is
consumed straight away by the router-score gradient, so in backward it is a
fully exposed, per-layer DCN round trip for 128 floats.

This module keeps the budget-0.0 behaviour for everything except those counts:

* Collectives whose operand is an expert count (``histc``/``bincount``, through
  dtype casts, views and earlier count collectives) are tagged ``MUST_SAVE``.
* Every other forward node is tagged ``MUST_RECOMPUTE``, which is what budget
  0.0 did to it anyway, the FSDP gather included.
* The partition then runs at budget 1.0, where the min-cut honours those tags.

The saved counts are the values backward would have recomputed: the same
collective over the same deterministic inputs. The math is unchanged; only
where the value comes from moves.

A body with no count collective is partitioned exactly as before, at 0.0 and
untagged.

Tags are applied to the joint graph just before it is partitioned rather than
at trace time, because the body's re-trace drops ``node.meta["custom"]`` (see
``LayerStack.run``). The hook goes in by swapping
``partitioners.min_cut_rematerialization_partition`` for the window's duration:
``HopGraphMinCutPartitioner.create_partitioned_graph`` imports it at call time.
"""

import contextlib
from collections.abc import Iterator

import torch
from torch import fx
from torch._functorch import config as functorch_config
from torch._functorch import partitioners
from torch.utils.checkpoint import CheckpointPolicy

__all__ = [
    "count_collectives",
    "scan_body_remat_window",
    "tag_scan_body_joint_graph",
]

_aten = torch.ops.aten

# Where an expert count is born.
_COUNT_SOURCES = frozenset({_aten.histc.default, _aten.bincount.default})

# Ops a count may pass through on its way to (and between) the collectives
# without becoming something else: casts, views and copies. `.to(dtype)` in
# `_batch_wise_load_balance_loss` is the one that matters.
_COUNT_PASSTHROUGH = frozenset(
    {
        _aten._to_copy.default,
        _aten.clone.default,
        _aten.alias.default,
        _aten.view.default,
        _aten.reshape.default,
        _aten._unsafe_view.default,
        _aten.detach.default,
        _aten.copy.default,
        torch.ops.prims.convert_element_type.default,
    }
)


def _is_collective(node: fx.Node) -> bool:
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace == "_c10d_functional"
    )


def count_collectives(graph: fx.Graph) -> list[fx.Node]:
    """The collectives in ``graph`` that carry expert counts, in graph order.

    Found by flowing forward from each ``histc``/``bincount`` through casts,
    views and collectives only. Anything that mixes the count with another
    value ends the flow, so a collective is picked up only if what it reduces
    *is* a count.
    """
    reached: set[fx.Node] = set()
    stack = [
        n
        for n in graph.nodes
        if n.op == "call_function" and n.target in _COUNT_SOURCES
    ]
    while stack:
        node = stack.pop()
        if node in reached:
            continue
        reached.add(node)
        for user in node.users:
            if user.op == "call_function" and (
                _is_collective(user) or user.target in _COUNT_PASSTHROUGH
            ):
                stack.append(user)
    return [n for n in graph.nodes if n in reached and _is_collective(n)]


def _depends_on_tangents(graph: fx.Graph) -> set[fx.Node]:
    """Nodes downstream of a tangent, i.e. the ones only backward can compute.

    ``HopJointGraph`` names the body's cotangent inputs ``tangents_*``.
    """
    closure: set[fx.Node] = set()
    for node in graph.nodes:
        if node.op == "placeholder":
            if str(node.target).startswith("tangents"):
                closure.add(node)
        elif any(i in closure for i in node.all_input_nodes):
            closure.add(node)
    return closure


def tag_scan_body_joint_graph(graph: fx.Graph) -> list[fx.Node]:
    """Tag a scan body's joint graph to save its expert counts, and only those.

    Returns the collectives tagged ``MUST_SAVE``. If there are none, the graph
    is left untouched: the caller should partition it as before.

    Tags already present are kept. The HOP partitioner sets ``MUST_RECOMPUTE``
    on complex SymInt expressions before calling the partition function, and
    that must stand.
    """
    to_save = count_collectives(graph)
    if not to_save:
        return []
    save = set(to_save)
    backward_only = _depends_on_tangents(graph)
    for node in graph.nodes:
        if node.op != "call_function" or node in backward_only:
            continue
        if node.meta.get("recompute") is not None:
            continue
        node.meta["recompute"] = (
            CheckpointPolicy.MUST_SAVE
            if node in save
            else CheckpointPolicy.MUST_RECOMPUTE
        )
    return to_save


@contextlib.contextmanager
def scan_body_remat_window() -> Iterator[None]:
    """Fully rematerialize a scan body, except for its expert counts.

    Wrap the ``scan`` call in this. Both the config and the partitioner are
    process-global, so it has to be a window; both are restored on exit.
    """
    inner = partitioners.min_cut_rematerialization_partition

    def partition(joint_module: fx.GraphModule, joint_inputs, *args, **kwargs):
        if not tag_scan_body_joint_graph(joint_module.graph):
            # No counts to keep: exactly the old path, at the window's 0.0.
            return inner(joint_module, joint_inputs, *args, **kwargs)
        # Every forward node now carries a tag, so the min-cut has nothing
        # left to choose; 1.0 just stops the 0.0 early return from ignoring
        # the tags.
        with functorch_config.patch(activation_memory_budget=1.0):
            return inner(joint_module, joint_inputs, *args, **kwargs)

    with functorch_config.patch(activation_memory_budget=0.0):
        partitioners.min_cut_rematerialization_partition = partition
        try:
            yield
        finally:
            partitioners.min_cut_rematerialization_partition = inner
