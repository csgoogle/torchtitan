"""FX pass: insert `torch_tpu::optimization_barrier` around the MoE
padded-combine region.

Motivation
----------
The MoE padded-combine code path
(`infra/padded_dispatcher.py::_padded_combine`) finishes the
expert-parallel combine with an fp32 argsort+sum reduction. In standalone
TPU bf16 this is bit-identical to the previous
`deterministic_scatter_add` form (see
`argsort_combine_fix/tpu_compare_combine_v3.py`), but inside the full
training graph the trajectories diverge: XLA cross-fuses the combine
with adjacent ops and the fused kernel's bf16 layout / SC offload
choices change the numerics enough to break the loss curve.

Wrapping the combine inputs+outputs in `torch_tpu::optimization_barrier`
("compiler-scheduling-barrier" in HLO) prevents that cross-fusion. The
eager-mode confirmation lives at
`opt_barrier_exp/RECOMMENDATIONS.md`:

| metric        | bare fast | barrier+fast | baseline (slow) |
|---------------|-----------|--------------|-----------------|
| loss step 30  | 11.89 BR  | 7.89         | 7.99            |
| loss step 15  | 14.58 BR  | 8.00         | 9.10            |
| step time     | 1.5 s     | 5.2 s        | 99.6 s          |
| MFU           | 21%       | 31.7%        | 0.33%           |

For compile mode the Python-source barrier breaks autograd because
`torch_tpu::optimization_barrier` has no Autograd registration in the
wheel. We therefore insert the barriers at the FX-graph level AFTER
`make_fx` (and after `_copy_fwd_metadata_to_bw_nodes` has copied the
forward annotation to backward nodes via seq_nr).

Identification
--------------
The pass relies on `torch.fx.traceback.annotate_fn({"EP": "padded_combine"})`
applied to `AllToAllTokenDispatcher.combine` in
`patch_dispatcher_with_capacity_factor`. Every node traced inside the
body carries `node.meta["custom"]["EP"] == "padded_combine"`. Backward
nodes inherit it via `seq_nr` matching (see
`torchtitan.experiments.graph_trainer.make_fx_tracer._copy_fwd_metadata_to_bw_nodes`).

Insertion strategy
------------------
For each connected "EP=padded_combine" region (forward and backward
each form their own region), insert barriers on:
  - Entries: external nodes whose users are inside the region. We
    barrier the external producer's output (replace external->region
    edges with external->barrier->getitem->region).
  - Exits: region nodes whose users are outside the region. We barrier
    the region node's output before it flows to non-region consumers.

This brackets the region exactly the way the eager-mode win did
(barriers on inputs AND outputs).

Reference: `mark_activation_checkpoints.py` (torch_tpu wheel) — same
rewrite pattern but selects on `node.meta["recompute"]` instead of
custom EP tag.
"""

from __future__ import annotations

import operator

import torch


_TAG_KEY = "EP"
_TAG_VAL = "padded_combine"


def _has_tag(node: torch.fx.Node) -> bool:
    custom = node.meta.get("custom")
    if custom is None:
        return False
    return custom.get(_TAG_KEY) == _TAG_VAL


def apply_combine_barrier_pass(
    gm: torch.fx.GraphModule,
    example_inputs,
) -> torch.fx.GraphModule:
    """Insert optimization_barrier on entries+exits of the padded-combine
    region.

    Pass signature matches the graph_trainer pass protocol:
    `(gm, example_inputs) -> gm`.
    """
    graph = gm.graph

    region = [n for n in graph.nodes if _has_tag(n)]
    if not region:
        # No annotations found — annotation infra may not be active (e.g.
        # `annotate_fn` from torch.fx.traceback unavailable on this torch,
        # or the patch hook didn't run). Silent no-op.
        print(
            "[insert_combine_barriers] no nodes with EP=padded_combine; pass is a no-op",
            flush=True,
        )
        return gm

    region_set = set(region)

    # Identify entries: external nodes that feed any region node.
    # External producer -> region consumer edge gets barrier interposed.
    external_producers: dict[torch.fx.Node, list[torch.fx.Node]] = {}
    for n in region:
        for inp in n.all_input_nodes:
            if inp in region_set:
                continue
            # Only barrier call_function / placeholder / get_attr producers
            # that yield a Tensor. We can't know "yields Tensor" cheaply, so
            # rely on the meta["val"] hint if present.
            val = inp.meta.get("val")
            if val is not None and not isinstance(val, torch.Tensor):
                continue
            external_producers.setdefault(inp, []).append(n)

    # Identify exits: region nodes with at least one external user.
    region_exits: list[torch.fx.Node] = []
    for n in region:
        for u in n.users:
            if u not in region_set:
                val = n.meta.get("val")
                if val is not None and not isinstance(val, torch.Tensor):
                    break
                region_exits.append(n)
                break

    def _insert_barrier(node: torch.fx.Node, users_to_redirect: list[torch.fx.Node]) -> None:
        """Insert barrier+getitem after ``node`` and rewrite ``users_to_redirect``
        to consume the barriered value.
        """
        if not users_to_redirect:
            return
        with graph.inserting_after(node):
            barrier = graph.call_function(
                torch.ops.torch_tpu.optimization_barrier.default,
                ([node],),
            )
        with graph.inserting_after(barrier):
            getitem = graph.create_node(
                "call_function",
                operator.getitem,
                args=(barrier, 0),
            )
        # Propagate meta so the barrier+getitem don't look "anonymous" to
        # downstream passes that inspect node.meta.
        val = node.meta.get("val")
        if val is not None:
            getitem.meta["val"] = val
        # Mark the barrier+getitem with the same EP tag as ``node`` if it
        # had one. Avoids accidental re-entry by downstream passes that
        # match on the tag.
        if "custom" in node.meta:
            getitem.meta.setdefault("custom", {}).update(node.meta["custom"])
        for u in users_to_redirect:
            u.replace_input_with(node, getitem)

    n_entry_barriers = 0
    n_exit_barriers = 0

    # Insert entry barriers first.
    for producer, consumers in external_producers.items():
        _insert_barrier(producer, consumers)
        n_entry_barriers += 1

    # Insert exit barriers.
    for exit_node in region_exits:
        external_users = [u for u in exit_node.users if u not in region_set]
        # `users` is a dict (NodeView). `replace_input_with` mutates the
        # users set on the producer, so list() it.
        _insert_barrier(exit_node, list(external_users))
        n_exit_barriers += 1

    graph.lint()

    print(
        f"[insert_combine_barriers] inserted {n_entry_barriers} entry + "
        f"{n_exit_barriers} exit optimization_barrier nodes "
        f"(region size: {len(region_set)} EP={_TAG_VAL} nodes)",
        flush=True,
    )

    return gm


__all__ = ["apply_combine_barrier_pass"]
