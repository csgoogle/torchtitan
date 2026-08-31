# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One call to put a decoder's layers under a scan.

The four pieces of this package have to run in one specific order, and two of
the orderings are silent when you get them wrong:

  1. Stack the layers, *before* anything walks `model.layers`. Activation
     checkpointing and FSDP both iterate the `ModuleDict`'s children, and the
     stack has a different child structure.
  2. The caller's own annotation and FSDP passes, with the stacked parameters
     ignored.
  3. Shard the stack on dim 1 and install the in-body gather. This needs the
     mesh, so it cannot happen at step 1.

`over_layers` is a context manager so that order is not something a caller can
get wrong: steps 1 and 3 are the enter and the exit, and the body is where the
caller's own two passes go.

    with scan.over_layers(
        model,
        parallel_dims=parallel_dims,
        dp_mesh=dp_mesh,
        dp_mode=dp_mode,
        mp_policy=mp_policy,
        recompute_body=...,
    ) as stack:
      annotate(model)
      model = apply_fsdp(model, ignored_params=set(stack.stacked_parameters()))

## What your model must satisfy

- `model.layers` is a `ModuleDict` of blocks keyed `"0"`, `"1"`, ..., all
  structurally identical: the stack keeps one as a template and feeds the rest
  in as stacked parameters.
- Each block's forward takes the hidden state first and everything after it
  loop-invariant -- `(h, attention_masks, positions, segment_ids)` for llama3.
  The trailing arguments are forwarded unchanged, so the list can grow, but a
  per-layer argument would have to become an `xs` instead.
- No complex-dtype op that requires grad in the block. The HOP re-traces the
  body functionally, where `view_as_complex` becomes `view_as_complex_copy`,
  which has no autograd formula. llama3 hits this in `ComplexRoPE`; see
  `models/common.py` for the real-arithmetic rewrite.
- TP, CP and PP are rejected: the stacked parameters carry a leading layer axis
  the existing parallel plans do not describe.
"""

import contextlib
import os
from typing import Any, Iterator

from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.experiments.graph_trainer.simple_fsdp import MixedPrecisionPolicy
from torchtitan.experiments.tpu.scan import fsdp as scan_fsdp
from torchtitan.experiments.tpu.scan import layer_stack
from torchtitan.experiments.tpu.scan import recompute as scan_recompute
from torchtitan.experiments.tpu.scan import reverse as scan_reverse
from torchtitan.tools.logging import logger


def resolve_peel_first_layer(train_config: Any) -> bool:
  """Whether layer 0 should run outside the scan, given a torchtitan config.

  A JAX-exported kernel inside the scan body lowers to a `func.call` whose
  callee torch_tpu drops when it clones the body into the `stablehlo.while`
  region, so a splash-attention block needs layer 0 peeled out to put the callee
  back in the outer module. `TPU_SCAN_PEEL_FIRST_LAYER=0|1` forces it either
  way, so the peel's cost can be measured against an otherwise identical scan
  and so the workaround can be switched off once torch_tpu carries the real fix.

  This is the one function in the package that reads a config object; everything
  else takes plain values, so a model without torchtitan's config types can
  still use it.
  """
  forced = os.environ.get("TPU_SCAN_PEEL_FIRST_LAYER")
  if forced is not None:
    return forced == "1"
  sa_config = getattr(train_config, "splash_attention_kernel", None)
  return bool(
      getattr(sa_config, "use_splash_attention_kernel", False)
      or getattr(sa_config, "use_tokamax_splash_attention_kernel", False)
  )


@contextlib.contextmanager
def over_layers(
    model: nn.Module,
    *,
    parallel_dims: Any = None,
    dp_mesh: DeviceMesh | None = None,
    dp_mode: str | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
    peel_first_layer: bool = False,
    recompute_body: bool = False,
    reverse_backward: bool = True,
) -> Iterator[layer_stack.ScannedLayerStack]:
  """Runs `model.layers` under one `scan`, sharded for FSDP on the way out.

  Args:
    model: has a `ModuleDict` at `.layers`. Modified in place.
    parallel_dims: torchtitan's `ParallelDims`, checked for TP/CP/PP. `None`
      skips the check.
    dp_mesh: the 1-D data-parallel mesh. `None` leaves the stack unsharded and
      turns the exit into a no-op, for a single-device run.
    dp_mode: SimpleFSDP mode, e.g. `"fully_shard"`. Required with `dp_mesh`.
    mp_policy: SimpleFSDP mixed-precision policy. Required with `dp_mesh`.
    peel_first_layer: run layer 0 outside the scan; see
      `resolve_peel_first_layer`.
    recompute_body: checkpoint the block, so backward recomputes it instead of
      keeping a per-layer residual. The scanned stack's activation-memory knob
      -- ordinary activation checkpointing does not reach inside a HOP body. See
      `recompute.py`.
    reverse_backward: lower the backward as a real reverse scan instead of
      flipping the residual stack. Pure memory; the values are identical. See
      `reverse.py`.

  Yields:
    The `ScannedLayerStack` now installed at `model.layers`. Pass
    `set(stack.stacked_parameters())` to your FSDP wrap as its ignored set --
    those are gathered per layer inside the body instead.

  Raises:
    ValueError: on a parallelism the stacked layout cannot express, or on a
      partial FSDP argument set.
  """
  for dim in ("tp", "cp", "pp"):
    if getattr(parallel_dims, f"{dim}_enabled", False):
      raise ValueError(
          f"scan over layers does not support {dim.upper()}: the stacked"
          " parameters carry a leading layer axis that the existing parallel"
          " plans do not describe."
      )
  if dp_mesh is not None and (dp_mode is None or mp_policy is None):
    raise ValueError(
        "dp_mesh needs dp_mode and mp_policy: they decide the gather's"
        " process group and dtypes, and the in-body gather has no default."
    )

  stack = layer_stack.stack_layers(
      model,
      peel_first_layer=peel_first_layer,
      recompute_body=recompute_body,
  )
  # Both of these patch a torch internal that only runs at compile time, so
  # installing them here rather than at the exit costs nothing and keeps every
  # global side effect of a scan in one place.
  if reverse_backward:
    scan_reverse.install()
  if recompute_body:
    scan_recompute.install()

  yield stack

  if dp_mesh is not None:
    scan_fsdp.shard_stacked_layers(
        stack, dp_mesh=dp_mesh, dp_mode=dp_mode, mp_policy=mp_policy
    )
  logger.info(
      "scan over %d layers (peel_first_layer=%s, recompute_body=%s,"
      " reverse_backward=%s, sharded=%s).",
      stack.n_layers,
      peel_first_layer,
      recompute_body,
      reverse_backward,
      dp_mesh is not None,
  )


__all__ = ["over_layers", "resolve_peel_first_layer"]
