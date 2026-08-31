# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scan over transformer decoder layers on TPU.

Replaces a model's per-layer Python loop with a single `scan` higher-order op,
so N identical blocks lower to one `stablehlo.while` instead of N unrolled
copies: compile time and program size stop growing with depth.

`over_layers` is the entry point -- see `api.py` for how to call it and what a
model must satisfy, and `README.md` for what each of the three modules
underneath it is solving. In torchtitan, `--tpu-config.use-scan` wires it up for
llama3 on the graph_trainer (aot_fx_trace) lane.
"""

from torchtitan.experiments.tpu.scan.api import (
    over_layers,
    resolve_peel_first_layer,
)
from torchtitan.experiments.tpu.scan.layer_stack import ScannedLayerStack

__all__ = [
    # The entry point.
    "over_layers",
    # The stack itself, for `stacked_parameters()`.
    "ScannedLayerStack",
    # Config -> scan parameters.
    "resolve_peel_first_layer",
]
