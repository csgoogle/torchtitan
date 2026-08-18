"""Remove the paired expert dWeight BF16/FP32 cast round trip.

The paired Pallas backward can return values that are rounded to BF16 but
stored as FP32. AOTAutograd still coerces that output to the BF16 parameter
dtype, and SimpleFSDP immediately widens it again for FP32 reduce-scatter:

    paired_bwd FP32 -> BF16 -> view -> FP32 -> reduce_scatter

This pass runs before TPU remat scheduling, redirects reduce-scatter to the
already rounded FP32 value, and removes the dead casts. The remat scheduler
then sees the final FP32 reduce-scatter input and builds consistent barrier
metadata around it. The matcher is deliberately exact and raises if the traced
chain changes.
"""

from __future__ import annotations

import operator
import re

import torch

from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    is_reduce_scatter_tensor,
)
from torchtitan.tools.logging import logger


_TARGET_PATTERN = re.compile(
    r"tt_redacted::paired_ragged_dot_bwd_bf16_rounded_fp32_tm[0-9]+"
)


def _tensor(node: torch.fx.Node) -> torch.Tensor:
  value = node.meta.get("val")
  if not isinstance(value, torch.Tensor):
    raise ValueError(f"Expected tensor metadata on {node.name}")
  return value


def _only_user(node: torch.fx.Node) -> torch.fx.Node:
  if len(node.users) != 1:
    raise ValueError(f"Expected one user of {node.name}, got {len(node.users)}")
  return next(iter(node.users))


def _is_to_dtype(node: torch.fx.Node, dtype: torch.dtype) -> bool:
  return (
      node.op == "call_function"
      and node.target is torch.ops.aten._to_copy.default
      and _tensor(node).dtype == dtype
  )


def _is_bf16_rounded_fp32_backward(node: torch.fx.Node) -> bool:
  if node.op != "call_function":
    return False
  target_name = getattr(node.target, "name", None)
  if not callable(target_name):
    return False
  return _TARGET_PATTERN.fullmatch(target_name()) is not None


def apply_elide_paired_dweight_casts_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
) -> torch.fx.GraphModule:
  """Feed BF16-rounded FP32 paired dWeights directly to FP32 FSDP RS."""
  del example_inputs
  backward_nodes = [
      node
      for node in gm.graph.nodes
      if _is_bf16_rounded_fp32_backward(node)
  ]
  if not backward_nodes:
    raise ValueError(
        "BF16-rounded paired dWeight cast elision found no matching kernels"
    )

  rewrites = 0
  for backward in backward_nodes:
    dweight_outputs = [
        user
        for user in backward.users
        if user.op == "call_function"
        and user.target is operator.getitem
        and user.args[1] == 1
    ]
    if len(dweight_outputs) != 1:
      raise ValueError(
          f"Expected one dWeight output from {backward.name}, "
          f"got {len(dweight_outputs)}"
      )
    dweight_fp32 = dweight_outputs[0]
    dweight_meta = _tensor(dweight_fp32)
    if dweight_meta.dtype != torch.float32:
      raise ValueError(f"Expected FP32 dWeight from {backward.name}")

    downcast = _only_user(dweight_fp32)
    if not _is_to_dtype(downcast, torch.bfloat16):
      raise ValueError(f"Expected BF16 downcast after {dweight_fp32.name}")
    view = _only_user(downcast)
    if view.target is not torch.ops.aten.view.default:
      raise ValueError(f"Expected view after {downcast.name}")
    view_meta = _tensor(view)
    if (
        view_meta.shape != dweight_meta.shape
        or view_meta.stride() != dweight_meta.stride()
    ):
      raise ValueError(f"Expected metadata-preserving view after {downcast.name}")

    upcast = _only_user(view)
    if not _is_to_dtype(upcast, torch.float32):
      raise ValueError(f"Expected FP32 upcast after {view.name}")
    reduce_scatter = _only_user(upcast)
    if not is_reduce_scatter_tensor(reduce_scatter):
      raise ValueError(f"Expected reduce-scatter after {upcast.name}")
    upcast_meta = _tensor(upcast)
    if (
        upcast_meta.shape != dweight_meta.shape
        or upcast_meta.stride() != dweight_meta.stride()
    ):
      raise ValueError(f"Expected shape-preserving cast chain for {backward.name}")

    reduce_scatter.replace_input_with(upcast, dweight_fp32)
    gm.graph.erase_node(upcast)
    gm.graph.erase_node(view)
    gm.graph.erase_node(downcast)
    rewrites += 1

  if rewrites != len(backward_nodes):
    raise RuntimeError(
        f"Rewrote {rewrites} paired dWeights for {len(backward_nodes)} kernels"
    )
  gm.graph.lint()
  gm.recompile()
  logger.info(
      "[elide_paired_dweight_casts] removed %d BF16/FP32 cast round trips",
      rewrites,
  )
  return gm


__all__ = ["apply_elide_paired_dweight_casts_pass"]
