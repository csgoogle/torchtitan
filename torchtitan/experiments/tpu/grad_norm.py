# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compiled gradient-norm helpers for TPU training."""

from collections.abc import Callable
from typing import cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
    tpu_backend_compile_pass,
)
from torchtitan.tools.logging import logger


GradSignature = tuple[
    tuple[
        torch.Size,
        torch.dtype,
        DeviceMesh | None,
        tuple[Placement, ...] | None,
    ],
    ...,
]


def _grad_signature(grads: list[torch.Tensor]) -> GradSignature:
  return tuple(
      (grad.shape, grad.dtype, grad.device_mesh, grad.placements)
      if isinstance(grad, DTensor)
      else (grad.shape, grad.dtype, None, None)
      for grad in grads
  )


def global_grad_norm(grads: list[torch.Tensor]) -> torch.Tensor:
  """Compute the global L2 norm while preserving DTensor placement semantics."""
  total_norm = torch.nn.utils.get_total_norm(
      grads,
      norm_type=2.0,
      error_if_nonfinite=False,
      foreach=True,
  )
  if isinstance(total_norm, DTensor):
    # get_total_norm leaves an internal partial scalar; full_tensor lowers its
    # global norm reduction here without gathering any gradient tensor.
    return total_norm.full_tensor()
  return total_norm


class CompiledGradNorm:
  """Compile and cache the global norm calculation for a trainer's gradients."""

  # The compiled norm is checked once against the eager DTensor norm; beyond
  # this relative error it is not trusted and every later call uses eager.
  _VALIDATION_RTOL = 1e-4

  def __init__(self) -> None:
    self._runner: Callable[[list[torch.Tensor]], torch.Tensor] | None = None
    self._signature: GradSignature | None = None
    self._validated = False
    self._use_eager = False

  @torch.no_grad()
  def clip_grad_norm_(
      self,
      parameters: list[torch.Tensor],
      max_norm: float,
  ) -> torch.Tensor:
    grads = cast(
        list[torch.Tensor],
        [parameter.grad for parameter in parameters if parameter.grad is not None],
    )
    if self._use_eager:
      total_norm = global_grad_norm(grads)
      torch.nn.utils.clip_grads_with_norm_(
          parameters, max_norm, total_norm, foreach=True
      )
      return total_norm
    signature = _grad_signature(grads)
    if self._runner is None:
      if not grads:
        raise ValueError("compile_grad_norm requires at least one gradient")
      # Gradients do not exist until backward, so their first materialized
      # shapes and placements are the earliest reliable trace inputs.
      logger.info(
          "[compiled_grad_norm] compiling global norm for %d gradients",
          len(grads),
      )
      traced = minimal_fx_tracer(global_grad_norm)(grads)
      # This is a standalone graph; is_fwd avoids backward-only AC marking and
      # selects the forward debug-dump bucket.
      traced.gm = tpu_backend_compile_pass(
          traced.gm,
          traced.example_inputs,
          is_fwd=True,
      )
      self._runner = run_traced(traced, _validate_runtime=True)
      self._signature = signature
    elif signature != self._signature:
      raise ValueError(
          "gradient metadata changed after compiling the gradient norm: "
          f"trace-time {self._signature}, runtime {signature}"
      )
    # Do not retrace a collective independently on ranks that might diverge.
    total_norm = self._runner(grads)
    if not self._validated:
      self._validated = True
      # A miscompiled reduction is silent: training runs, but clipping uses
      # the wrong norm (TorchTPU 710e117f under-reported Qwen3-30B-A3B's
      # step-1 norm as 16.69 instead of 19.09). Every rank computes the same
      # replicated scalars, so they all take the same branch.
      eager_norm = global_grad_norm(grads)
      compiled, reference = float(total_norm), float(eager_norm)
      if abs(compiled - reference) > self._VALIDATION_RTOL * max(abs(reference), 1e-12):
        logger.error(
            "[compiled_grad_norm] compiled norm %.6g != eager norm %.6g; "
            "using the eager norm for the rest of training",
            compiled,
            reference,
        )
        self._use_eager = True
        total_norm = eager_norm
      else:
        logger.info(
            "[compiled_grad_norm] compiled norm matches eager (%.6g)", compiled
        )

    # Compile the collective-heavy reduction, but leave clipping as one
    # bandwidth-bound eager foreach multiply. Capturing that mutation would
    # return every gradient and, without safe output donation, add a
    # model-sized copy-back or require rebinding every Parameter.grad.
    torch.nn.utils.clip_grads_with_norm_(
        parameters,
        max_norm,
        total_norm,
        foreach=True,
    )
    return total_norm
