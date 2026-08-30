# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scan over transformer decoder layers on TPU.

Replaces a model's per-layer Python loop with a single `scan` higher-order op,
so N identical blocks lower to one `stablehlo.while` instead of N unrolled
copies. Compile time and program size stop growing with depth.

## Applying it to your own model

Two entry points, called at two different points of your parallelization
function. They are separate because FSDP has to run *between* them: the stack
must exist before the FQNs are annotated, and the per-layer gather plan can
only be installed once the mesh is known.

    from torchtitan.experiments.tpu.scan import (
        scan_over_layers,
        shard_scanned_layers,
    )

    # 1. Stack the layers. Must precede anything that walks `model.layers` --
    #    activation checkpointing and FSDP both iterate the ModuleDict, and the
    #    stack has a different child structure.
    stack = scan_over_layers(
        model, parallel_dims=parallel_dims, peel_first_layer=False
    )

    # 2. Your own annotation / FQN pass, if you have one.
    annotate(model)

    # 3. FSDP over everything EXCEPT the stacked parameters -- those are
    #    gathered per layer inside the scan body instead (see `fsdp.py`).
    model = apply_fsdp(model, ignored_params=set(stack.stacked_parameters()))

    # 4. Install the in-body gather plan.
    shard_scanned_layers(stack, dp_mesh=dp_mesh, dp_mode=dp_mode,
                         mp_policy=mp_policy, recompute_body=...)

`torchtitan/experiments/tpu/graph_trainer/parallelize_scan.py` is a worked
example of exactly these four steps.

## What your model must satisfy

- `model.layers` is a `ModuleDict` of blocks keyed `"0"`, `"1"`, ... All blocks
  must be structurally identical: the stack keeps one as a template and feeds
  the rest in as stacked parameters.
- Each block's forward takes `(h, attention_masks, positions)` and returns the
  new hidden state. This signature is currently fixed; a block taking different
  arguments needs `ScannedLayerStack.forward` widened.
- No complex-dtype ops that require grad in the block. The HOP re-traces the
  body functionally, where `view_as_complex` becomes `view_as_complex_copy`,
  which has no autograd formula. llama3 hits this in `ComplexRoPE`; see
  `torchtitan/experiments/tpu/rope_real_arithmetic.py` for the rewrite.
- TP, CP and PP are not supported yet -- the stacked parameters carry a leading
  layer axis the existing parallel plans do not describe. `scan_over_layers`
  raises rather than silently mis-sharding.

## In torchtitan

`--tpu-config.use-scan` wires this up for llama3 on both the fullgraph and
graph_trainer (aot_fx_trace) lanes. See `README.md` in this directory.
"""

from torchtitan.experiments.tpu.scan.fsdp import (
    install_scan_body_ac_shim,
    ScanFSDPSpec,
    shard_stacked_layers,
    shard_stacked_layers as shard_scanned_layers,
    uninstall_scan_body_ac_shim,
)
from torchtitan.experiments.tpu.scan.layer_stack import (
    apply_scan_over_layers,
    install_scan_over_layers,
    install_scan_over_layers as scan_over_layers,
    ScannedLayerStack,
)
from torchtitan.experiments.tpu.scan.policy import resolve_peel_first_layer

__all__ = [
    # The two public entry points, in call order.
    "scan_over_layers",
    "shard_scanned_layers",
    # The stack itself, for `stacked_parameters()` and checkpoint conversion.
    "ScannedLayerStack",
    # Lower-level pieces, for callers that need the steps apart.
    "apply_scan_over_layers",
    "install_scan_over_layers",
    "shard_stacked_layers",
    "ScanFSDPSpec",
    "install_scan_body_ac_shim",
    "uninstall_scan_body_ac_shim",
    # Config -> scan parameters, shared by both lanes.
    "resolve_peel_first_layer",
]
