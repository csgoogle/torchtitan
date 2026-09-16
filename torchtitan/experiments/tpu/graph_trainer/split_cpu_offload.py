# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-offload support for independently compiled TPU graph children."""

import operator
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node
from torchtitan.tools.logging import logger


@dataclass(frozen=True)
class HostDonorSpec:
  shape: tuple[int, ...]
  stride: tuple[int, ...]
  dtype: torch.dtype


def _initialize_pinned_host_buffer(source: torch.Tensor) -> torch.Tensor:
  host = torch.ops.ao.offload.default(source)
  return torch.ops.ao.wait_tensor.default(host, source, source)


def compile_host_donor_initializer() -> Callable[[torch.Tensor], torch.Tensor]:
  """Compile the one-time device-to-pinned-host donor initializer."""
  from torch_tpu._internal.compile.compiler import StaticCompiler

  def backend(
      gm: torch.fx.GraphModule,
      example_inputs: list[torch.Tensor],
  ) -> Callable:
    return StaticCompiler()(gm, example_inputs, is_fwd=True)

  return torch.compile(
      _initialize_pinned_host_buffer,
      backend=backend,
      fullgraph=True,
  )


def is_ao_transfer(
    node: torch.fx.Node,
    op_name: str | None = None,
) -> bool:
  """Return whether ``node`` is an activation-offload placement op."""
  if not (
      node.op == "call_function"
      and getattr(node.target, "namespace", None) == "ao"
  ):
    return False
  return op_name is None or str(node.target).startswith(f"ao.{op_name}.")


def _first_downstream_partition(
    node: torch.fx.Node,
    partition_map: dict[torch.fx.Node, int],
) -> int:
  pending = deque(node.users)
  visited: set[torch.fx.Node] = set()
  while pending:
    user = pending.popleft()
    if user in visited:
      continue
    visited.add(user)
    if not is_ao_transfer(user) and user in partition_map:
      return partition_map[user]
    pending.extend(user.users)
  return -1


def _first_scheduled_partition(
    node: torch.fx.Node,
    partition_map: dict[torch.fx.Node, int],
) -> int:
  found = False
  for candidate in node.graph.nodes:
    if candidate is node:
      found = True
      continue
    if not found or is_ao_transfer(candidate):
      continue
    if candidate in partition_map:
      return partition_map[candidate]
  return -1


def _ao_transfer_partition(
    node: torch.fx.Node,
    partition_map: dict[torch.fx.Node, int],
) -> int:
  """Keep each asynchronous AO copy and its wait in one executable.

  Forward D2H belongs to the partition containing its final device-storage
  consumer. Backward H2D belongs where the generic prefetch pass scheduled its
  reload in FX order. In both cases the ready tensor, rather than the runtime's
  executable-local async descriptor, may cross the next split boundary.
  """
  if is_ao_transfer(node, "offload"):
    waits = [
        user
        for user in node.users
        if is_ao_transfer(user, "wait_tensor")
        and user.args
        and user.args[0] is node
    ]
    if waits:
      return _ao_transfer_partition(waits[0], partition_map)

  if is_ao_transfer(node, "wait_tensor") and not _is_backward_node(node):
    if len(node.args) > 2 and isinstance(node.args[2], torch.fx.Node):
      partition = partition_map.get(node.args[2], -1)
      if partition >= 0:
        return partition

  if is_ao_transfer(node, "reload") and _is_backward_node(node):
    partition = _first_scheduled_partition(node, partition_map)
    if partition >= 0:
      return partition

  if is_ao_transfer(node, "wait_tensor") and _is_backward_node(node):
    producer = node.args[0] if node.args else None
    if isinstance(producer, torch.fx.Node) and is_ao_transfer(
        producer, "reload"
    ):
      return _ao_transfer_partition(producer, partition_map)

  return _first_downstream_partition(node, partition_map)


def assign_ao_transfer_partitions(
    gm: torch.fx.GraphModule,
    partition_map: dict[torch.fx.Node, int],
) -> None:
  """Assign AO pairs without carrying an async descriptor across children."""
  for node in gm.graph.nodes:
    if not is_ao_transfer(node):
      continue
    partition = _ao_transfer_partition(node, partition_map)
    if partition < 0:
      raise ValueError(
          f"Cannot assign AO transfer {node.name!r} to a split partition"
      )
    partition_map[node] = partition


def move_forward_offloads_to_partition_entry(
    gm: torch.fx.GraphModule,
    partition_map: dict[torch.fx.Node, int],
) -> None:
  """Move D2H starts to the entry of the child containing their wait."""
  nodes = list(gm.graph.nodes)
  node_order = {node: index for index, node in enumerate(nodes)}
  first_node_by_partition: dict[int, torch.fx.Node] = {}
  for node in nodes:
    if node.op in ("placeholder", "output") or is_ao_transfer(node):
      continue
    partition = partition_map.get(node)
    if partition is not None and partition not in first_node_by_partition:
      first_node_by_partition[partition] = node

  moved = False
  for node in nodes:
    if not is_ao_transfer(node, "offload"):
      continue
    anchor = first_node_by_partition.get(partition_map[node])
    if anchor is None or node_order[node] >= node_order[anchor]:
      continue
    if any(
        input_node.op != "placeholder"
        and node_order[input_node] >= node_order[anchor]
        for input_node in node.all_input_nodes
    ):
      continue
    anchor.prepend(node)
    moved = True

  if moved:
    gm.graph.lint()
    gm.recompile()


def validate_ao_transfer_pairs(
    chunked_gm: torch.fx.GraphModule,
    graph_name: str,
) -> None:
  """Reject async activation copies whose wait crossed a child boundary."""
  for child_name, child in chunked_gm.named_children():
    if not isinstance(child, torch.fx.GraphModule):
      continue
    child_nodes = set(child.graph.nodes)
    producers = [
        node
        for node in child_nodes
        if is_ao_transfer(node, "offload")
        or is_ao_transfer(node, "reload")
    ]
    waits = [
        node for node in child_nodes if is_ao_transfer(node, "wait_tensor")
    ]
    for wait in waits:
      producer = wait.args[0] if wait.args else None
      if not (
          isinstance(producer, torch.fx.Node)
          and producer in child_nodes
          and (
              is_ao_transfer(producer, "offload")
              or is_ao_transfer(producer, "reload")
          )
      ):
        raise ValueError(
            f"TPU {graph_name} child {child_name!r} contains AO wait "
            f"{wait.name!r} without its async-copy producer"
        )
    for producer in producers:
      if not any(
          is_ao_transfer(user, "wait_tensor") and user in child_nodes
          for user in producer.users
      ):
        raise ValueError(
            f"TPU {graph_name} child {child_name!r} contains AO producer "
            f"{producer.name!r} without its wait"
        )


def _as_tuple(value: Any) -> tuple[Any, ...]:
  if isinstance(value, tuple):
    return value
  if isinstance(value, list):
    return tuple(value)
  return (value,)


def _child_output_source(
    parent: torch.fx.GraphModule,
    value: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.GraphModule, int] | None:
  call = value
  output_index = 0
  if (
      value.op == "call_function"
      and value.target is operator.getitem
      and len(value.args) == 2
      and isinstance(value.args[0], torch.fx.Node)
      and isinstance(value.args[1], int)
  ):
    call = value.args[0]
    output_index = value.args[1]
  if call.op != "call_module":
    return None
  child = parent.get_submodule(str(call.target))
  if not isinstance(child, torch.fx.GraphModule):
    return None
  child_output = next(
      node for node in child.graph.nodes if node.op == "output"
  )
  outputs = _as_tuple(child_output.args[0])
  if value is call and len(outputs) != 1:
    return None
  if output_index < 0 or output_index >= len(outputs):
    return None
  if not isinstance(outputs[output_index], torch.fx.Node):
    return None
  return call, child, output_index


def thread_host_offload_donors(
    parent: torch.fx.GraphModule,
) -> tuple[
    tuple[HostDonorSpec, ...],
    tuple[int, ...],
    tuple[torch.Tensor, ...],
]:
  """Thread previous pinned-host outputs into D2H children for donation."""
  parent_output = next(
      node for node in parent.graph.nodes if node.op == "output"
  )
  parent_outputs = _as_tuple(parent_output.args[0])
  first_parent_op = next(
      node for node in parent.graph.nodes if node.op != "placeholder"
  )
  specs = []
  output_positions = []
  examples = []
  total_bytes = 0
  wired_outputs: set[tuple[torch.fx.Node, int]] = set()

  for parent_output_position, value in enumerate(parent_outputs):
    if not isinstance(value, torch.fx.Node):
      continue
    resolved = _child_output_source(parent, value)
    if resolved is None:
      continue
    call, child, child_output_position = resolved
    key = (call, child_output_position)
    if key in wired_outputs:
      continue
    child_output = next(
        node for node in child.graph.nodes if node.op == "output"
    )
    child_outputs = _as_tuple(child_output.args[0])
    offload_wait = child_outputs[child_output_position]
    if not (
        isinstance(offload_wait, torch.fx.Node)
        and is_ao_transfer(offload_wait, "wait_tensor")
        and not _is_backward_node(offload_wait)
    ):
      continue
    value_meta = offload_wait.meta.get("val")
    if not (
        isinstance(value_meta, torch.Tensor)
        and value_meta.device.type == "cpu"
    ):
      continue

    donor_index = len(specs)
    first_child_op = next(
        node for node in child.graph.nodes if node.op != "placeholder"
    )
    with child.graph.inserting_before(first_child_op):
      child_donor = child.graph.placeholder(f"host_donor_{donor_index}")
    child_donor.meta = dict(offload_wait.meta)
    child_donor.meta["tpu_force_donate"] = True
    wait_args = list(offload_wait.args)
    while len(wait_args) < 3:
      wait_args.append(None)
    wait_args[1] = child_donor
    offload_wait.args = tuple(wait_args)

    with parent.graph.inserting_before(first_parent_op):
      parent_donor = parent.graph.placeholder(f"host_donor_{donor_index}")
    parent_donor.meta = dict(child_donor.meta)
    call.args = (*call.args, parent_donor)

    specs.append(
        HostDonorSpec(
            shape=tuple(int(dim) for dim in value_meta.shape),
            stride=tuple(int(dim) for dim in value_meta.stride()),
            dtype=value_meta.dtype,
        )
    )
    output_positions.append(parent_output_position)
    examples.append(value_meta)
    total_bytes += value_meta.numel() * value_meta.element_size()
    wired_outputs.add(key)

  for child in parent.children():
    if isinstance(child, torch.fx.GraphModule):
      child.graph.lint()
      child.recompile()
  parent.graph.lint()
  parent.recompile()
  logger.info(
      "Threaded %d pinned-host outputs as recurrent donated inputs (%.2f GiB)",
      len(specs),
      total_bytes / (1024**3),
  )
  return tuple(specs), tuple(output_positions), tuple(examples)
