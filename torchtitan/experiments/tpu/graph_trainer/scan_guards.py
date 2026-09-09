# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Refuse to run when ``--tpu-config.use-scan-layers`` cannot be honoured.

Every failure mode here is otherwise **silent**: the run succeeds -- unrolled,
or stacked from layer 0 with the differences cast away -- and its log is
indistinguishable from a correct scan run. So none of these fall back; they
raise, and they name the layer, the parameter, or the flag to change.

``check_scan_was_applied`` is the important one. It runs *after*
parallelization and asks whether this run actually got a scan, rather than
enumerating the lanes that support one, so a lane added later that forgets to
thread the flag fails on its first run.
"""

import torch
from torch import nn

from torchtitan.distributed import ParallelDims


__all__ = [
    "check_scan_parallelism",
    "check_layers_are_stackable",
    "check_scan_was_applied",
]


def check_scan_parallelism(parallel_dims: ParallelDims) -> None:
    """Scan is 1-D data parallel only, for now.

    The in-body gather assumes each stacked weight is sharded on exactly one
    mesh axis. TP and CP add a second axis over the same tensor, and PP splits
    the layer range that the scan *is* the loop over: design work, not a flag.
    """
    conflicts = [
        f"--parallelism.{kind}-parallel-degree={degree}"
        for kind, degree in (
            ("tensor", parallel_dims.tp),
            ("context", parallel_dims.cp),
            ("pipeline", parallel_dims.pp),
        )
        if degree > 1
    ]
    if conflicts:
        raise ValueError(
            "--tpu-config.use-scan-layers supports 1-D data parallelism only, "
            f"but this run sets {', '.join(conflicts)}. Drop the flag, or set "
            "those degrees to 1."
        )


def check_layers_are_stackable(layers: nn.ModuleDict) -> None:
    """Every layer must be identical to layer 0, or they cannot share a stack.

    Parameters become rows of one stacked tensor, so they must agree on name,
    shape and dtype. Dtype and not just shape, because the stack is allocated
    once from layer 0 and a layer differing only in dtype would be cast into it
    rather than rejected. Buffers are not stacked at all -- the body runs every
    iteration with layer 0's -- so they must agree on *value* as well, the one
    difference no structural comparison would catch.

    So a model with a dense first layer and MoE the rest of the way fails here.
    """

    def spec(layer: nn.Module) -> dict[str, tuple]:
        named = [*layer.named_parameters(), *layer.named_buffers()]
        return {n: (tuple(t.shape), t.dtype) for n, t in named}

    template, zero_buffers = spec(layers["0"]), dict(layers["0"].named_buffers())
    for i, layer in enumerate(layers.values()):
        got = spec(layer)
        if got != template:
            # key=str: torch.dtype is not orderable.
            differences = sorted(set(template.items()) ^ set(got.items()), key=str)
        else:
            # Same names and shapes: compare buffer values. Only once
            # materialized -- on meta there is nothing to compare, and refusing
            # to run there would reject every real model.
            differences = [
                f"buffer {n} holds different values"
                for n, b in layer.named_buffers()
                if not (b.is_meta or zero_buffers[n].is_meta)
                and not torch.equal(zero_buffers[n], b)
            ]
        if differences:
            raise ValueError(
                f"layer {i} does not match layer 0, so they cannot run as one "
                f"scan: {differences}. Every layer must hold the same "
                "parameters and buffers at the same shapes and dtypes, and the "
                "same buffer values too, since buffers are shared rather than "
                "stacked. Make a per-layer value a parameter, or pass it into "
                "the body as a scan input."
            )


def check_scan_was_applied(model: nn.Module, *, use_scan_layers: bool) -> None:
    """After parallelization: did the flag actually reach anything?

    A lane that never reads the flag produces a perfectly ordinary unrolled
    run. Deliberately a property of the built model rather than a list of
    supported lanes.
    """
    if not use_scan_layers:
        return
    # Imported here, not at module scope: `layer_stack` calls
    # `check_layers_are_stackable` above, so a top-level import closes the cycle.
    from torchtitan.experiments.tpu.graph_trainer.layer_stack import StackedLayers

    # Not "are the layers stacked?" -- stacking is also reachable through
    # `--tpu-config.use-stacked-layers`, so a lane that forwards that flag and
    # drops this one would pass. Ask the container that actually runs the layers.
    if any(isinstance(m, StackedLayers) and m.use_scan for m in model.modules()):
        return
    raise ValueError(
        "--tpu-config.use-scan-layers was set, but nothing in this run "
        "consulted it: the decoder layers ran one at a time. Scan is wired up "
        "for the graph trainer's llama3 lane; check that "
        "--tpu-config.use-graph-trainer is set."
    )
