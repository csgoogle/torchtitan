# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Split a GraphTrainer joint FX graph into independently compiled modules.

``use_graph_split`` first divides the mandatory joint graph into forward and
backward GraphModules connected by explicit residual tensors. The forward graph
can then be divided into sequential layer ranges. The backward graph can remain
whole, group adjacent rematerialization regions sequentially, or form staggered
FSDP all-gather/compute/reduce-scatter pipeline stages. These subdivisions are
therefore always nested under the forward/backward split.

Each rematerialization region is one transformer layer. Combining regions in
one executable changes only the compiler scheduling scope; it preserves the
per-layer activation-checkpoint boundaries and just-in-time recompute order.
Multi-layer rematerialization instead trades fewer saved or offloaded boundaries
for longer-lived intermediates or repeated recompute and is outside this pass.

The result wrappers retain both the executable GraphModule and the node-to-
partition metadata used to validate that semantic regions and collective chains
remain indivisible.
"""

import operator
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch
from torch import nn
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor,
    is_all_reduce_tensor,
    is_all_to_all_tensor,
    is_reduce_scatter_tensor,
    is_wait_tensor,
)
from torch.fx.passes.split_module import split_module

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    _collect_bucketed_fsdp_comms,
)
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
  from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
  from torchtitan.experiments.tpu.tpu_job_config import (
      GraphSplitterConfig,
      TPUTrainerConfig,
  )


def _enqueue_forward_graph_outputs(values: list[torch.Tensor]) -> None:
  values = [value for value in values if value.device.type == "tpu"]
  if not values:
    return
  # TorchTPU is not installed in every TorchTitan host-test environment.
  from torch_tpu._internal.sync import synchronize

  synchronize(values, wait=False)


def _as_tuple(value: Any) -> tuple[Any, ...]:
  if isinstance(value, tuple):
    return value
  if isinstance(value, list):
    return tuple(value)
  return (value,)


class SplitForwardBackwardModule(nn.Module):

  def __init__(
      self,
      forward_graph: torch.fx.GraphModule,
      backward_graph: torch.fx.GraphModule,
      *,
      forward_output_positions: tuple[int, ...],
      backward_output_positions: tuple[int, ...],
      constant_outputs: dict[int, Any],
      num_direct_forward_outputs: int,
      num_outputs: int,
  ) -> None:
    super().__init__()
    self.forward_graph = forward_graph
    self.backward_graph = backward_graph
    self.forward_output_positions = forward_output_positions
    self.backward_output_positions = backward_output_positions
    self.constant_outputs = constant_outputs
    self.num_direct_forward_outputs = num_direct_forward_outputs
    self.num_outputs = num_outputs

  def forward(self, *flat_inputs: Any) -> tuple[Any, ...]:
    forward_outputs = _as_tuple(self.forward_graph(*flat_inputs))
    direct_forward_outputs = forward_outputs[
        : self.num_direct_forward_outputs
    ]
    residuals = forward_outputs[self.num_direct_forward_outputs :]
    # A compiled child may return lazy outputs without enqueueing its executable.
    # Residual dependencies cover only the forward children that produce those
    # residuals; direct outputs such as loss may come from other children. Enqueue
    # every tensor output before invoking backward so all forward children precede
    # backward collectives. wait=False preserves asynchronous host execution.
    _enqueue_forward_graph_outputs(list(forward_outputs))
    backward_outputs = _as_tuple(
        self.backward_graph(*flat_inputs, *residuals)
    )

    outputs = [None] * self.num_outputs
    positioned_outputs = chain(
        zip(
            self.forward_output_positions,
            direct_forward_outputs,
            strict=True,
        ),
        zip(
            self.backward_output_positions,
            backward_outputs,
            strict=True,
        ),
        self.constant_outputs.items(),
    )
    for position, value in positioned_outputs:
      outputs[position] = value
    return tuple(outputs)


@dataclass
class SplitForwardBackwardGraph:
  module: SplitForwardBackwardModule
  forward_graph: torch.fx.GraphModule
  backward_graph: torch.fx.GraphModule
  forward_example_inputs: tuple[Any, ...]
  backward_example_inputs: tuple[Any, ...]
  residual_nodes: tuple[torch.fx.Node, ...]


@dataclass
class SequentialChunkedGraph:
  module: torch.fx.GraphModule
  partition_map: dict[torch.fx.Node, int]
  unit_to_partition: dict[int, int]


@dataclass
class FSDPPipelineChunkedGraph:
  module: torch.fx.GraphModule
  partition_map: dict[torch.fx.Node, int]
  compute_region_to_partition: dict[int, int]
  ag_layer_to_partition: dict[int, int]
  rs_layer_to_partition: dict[int, int]


@dataclass
class _UnitChunkedGraph(SequentialChunkedGraph):
  collective_layer_to_partition: dict[str, dict[int, int]]


@dataclass(frozen=True)
class GraphSplitter:
  """Validate, partition, and compile one GraphTrainer trace."""

  enabled: bool
  config: "GraphSplitterConfig"
  n_layers: int

  @classmethod
  def from_trainer_config(
      cls,
      config: "TPUTrainerConfig",
      *,
      optimizer_in_graph: bool,
  ) -> "GraphSplitter":
    enabled = bool(config.tpu_config.use_graph_split)
    split_config = config.tpu_config.graph_split
    if split_config.forward_layers_per_executable == 0 or (
        split_config.forward_layers_per_executable < -1
    ):
      raise ValueError(
          "forward_layers_per_executable must be -1 or positive"
      )
    if split_config.backward_partition_mode not in (
        "none",
        "sequential",
        "fsdp_pipeline",
        "fsdp_region_local",
    ):
      raise ValueError(
          "backward_partition_mode must be none, sequential, fsdp_pipeline, "
          "or fsdp_region_local"
      )
    if split_config.backward_regions_per_executable < 1:
      raise ValueError(
          "backward_regions_per_executable must be at least 1"
      )
    if split_config.backward_partition_mode != "none" and not enabled:
      raise ValueError("backward_partition_mode requires use_graph_split")
    if enabled and config.tpu_config.graph_trainer_fsdp_schedule != "none":
      raise ValueError(
          "Graph splitting does not currently compose with "
          "graph_trainer_fsdp_schedule"
      )
    # Optimizer-in-graph needs a third explicit phase rather than treating its
    # state reads, mutations, and parameter updates as backward.
    if enabled and optimizer_in_graph:
      raise ValueError(
          "TPU GraphTrainer graph split does not support optimizer_in_graph"
      )
    # This artifact format stores one transformed FX graph. TorchTPU's XLA and
    # Tier-2 executable caches remain supported for every compiled child.
    if enabled and config.compile.precompile_artifact_dir:
      raise ValueError(
          "TPU GraphTrainer graph split does not support serialized "
          "precompiled FX artifacts"
      )
    return cls(
        enabled=enabled,
        config=split_config,
        n_layers=len(config.model_spec.model.layers) if enabled else 0,
    )

  def split_and_compile(self, traced: "TracedResult") -> torch.fx.GraphModule:
    """Split one transformed joint graph and compile its child executables."""
    if not self.enabled:
      raise RuntimeError("GraphSplitter is disabled")

    from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
        tpu_backend_compile_partitioned_pass,
        tpu_backend_compile_pass,
    )

    split = split_forward_backward_graph(traced.gm, traced.example_inputs)

    if self.config.forward_layers_per_executable > 0:
      forward = split_forward_graph_by_layer(
          split.forward_graph,
          self.config.forward_layers_per_executable,
          isolate_prefix=self.config.forward_isolate_prefix,
      )
      split.forward_graph = forward.module
      split.module.forward_graph = forward.module
      tpu_backend_compile_partitioned_pass(
          forward.module,
          split.forward_example_inputs,
          donate_internal_buffers=self.config.donate_internal_buffers,
      )
    else:
      tpu_backend_compile_pass(
          split.forward_graph,
          split.forward_example_inputs,
      )

    mode = self.config.backward_partition_mode
    if mode in ("fsdp_pipeline", "fsdp_region_local"):
      backward = split_backward_graph_by_fsdp_pipeline(
          split.backward_graph,
          n_layers=self.n_layers,
          regions_per_executable=(
              self.config.backward_regions_per_executable
          ),
          collective_ownership=(
              "region_local" if mode == "fsdp_region_local" else "staggered"
          ),
      )
      split.backward_graph = backward.module
      split.module.backward_graph = backward.module
      tpu_backend_compile_partitioned_pass(
          backward.module,
          split.backward_example_inputs,
          is_fwd=False,
          donate_internal_buffers=self.config.donate_internal_buffers,
      )
    elif mode == "sequential":
      backward = split_backward_graph_by_remat_region(
          split.backward_graph,
          self.config.backward_regions_per_executable,
      )
      split.backward_graph = backward.module
      split.module.backward_graph = backward.module
      tpu_backend_compile_partitioned_pass(
          backward.module,
          split.backward_example_inputs,
          is_fwd=False,
          donate_internal_buffers=self.config.donate_internal_buffers,
      )
    else:
      tpu_backend_compile_pass(
          split.backward_graph,
          split.backward_example_inputs,
          is_fwd=False,
      )
    return split.module


def _is_getitem(node: torch.fx.Node) -> bool:
  return node.op == "call_function" and (
      node.target == operator.getitem
      or getattr(node.target, "__name__", "") == "getitem"
  )


def _collective_producer(node: torch.fx.Node) -> torch.fx.Node | None:
  if not node.args or not isinstance(node.args[0], torch.fx.Node):
    return None
  producer = node.args[0]
  while _is_getitem(producer):
    if not producer.args or not isinstance(producer.args[0], torch.fx.Node):
      return None
    producer = producer.args[0]
  if _is_collective(producer):
    return producer
  return None


def _is_collective(node: torch.fx.Node) -> bool:
  return (
      is_all_gather_into_tensor(node)
      or is_reduce_scatter_tensor(node)
      or is_all_reduce_tensor(node)
      or is_all_to_all_tensor(node)
  )


def _is_spmd_region_marker(node: torch.fx.Node, marker: str) -> bool:
  return (
      node.op == "call_function"
      and str(getattr(node.target, "overloadpacket", node.target))
      == f"torch_tpu.{marker}"
  )


def _collect_collective_chains(
    gm: torch.fx.GraphModule,
) -> dict[torch.fx.Node, set[torch.fx.Node]]:
  """Collect indivisible collective and FSDP dtype-conversion chains.

  Collective classification comes from TorchInductor's bucketing helpers. For
  each all-gather launch, repeatedly follow tensor inputs through upstream
  ``call_function`` nodes toward its parameter placeholder. Retain only nodes
  used exclusively by that launch chain. This keeps layout preparation with its
  all-gather without maintaining an op allowlist or absorbing shared compute.
  A BF16 FSDP reduction has this shape (the first cast is omitted when the
  computed gradient is already BF16):

    compute partition                 RS partition
    dWeight -> [to(BF16)]  |  -> RS(BF16) -> wait -> to(FP32) -> grad output

  The post-RS cast retains its module-layer metadata. If it is treated as layer
  compute, that metadata assigns it to the earlier compute partition even
  though its input is produced by the later RS partition. The FX graph is
  acyclic, but the resulting executable graph is not:

    compute -> RS -> compute

  Keep the post-RS cast with the launch/wait chain to preserve a single
  forward dependency between the two executables.
  """
  chains = {
      node: {node}
      for node in gm.graph.nodes
      if _is_collective(node)
  }
  for node in gm.graph.nodes:
    if _is_getitem(node) or is_wait_tensor(node):
      producer = _collective_producer(node)
      if producer is not None:
        chains[producer].add(node)
        if is_reduce_scatter_tensor(producer):
          chains[producer].update(
              user
              for user in node.users
              if user.op == "call_function"
              and user.target is torch.ops.aten._to_copy.default
              and len(user.all_input_nodes) == 1
          )
  for producer, chain in chains.items():
    if not is_all_gather_into_tensor(producer):
      continue
    candidates: set[torch.fx.Node] = set()
    pending = list(producer.all_input_nodes)
    while pending:
      node = pending.pop()
      if (
          node in chain
          or node in candidates
          or node.op != "call_function"
          or _is_collective(node)
          or is_wait_tensor(node)
      ):
        continue
      candidates.add(node)
      pending.extend(node.all_input_nodes)
    while candidates:
      shared = {
          node
          for node in candidates
          if any(
              user not in candidates and user not in chain
              for user in node.users
          )
      }
      if not shared:
        break
      candidates.difference_update(shared)
    chain.update(candidates)
  return chains


def _split_graph_by_units(
    gm: torch.fx.GraphModule,
    *,
    unit_id_fn: Callable[[torch.fx.Node], int],
    unit_name: str,
    descending: bool,
    graph_name: Literal["forward", "backward"],
    isolate_prefix: bool,
    units_per_executable: int,
    n_layers: int | None = None,
    fsdp_collective_schedule: (
        Literal["staggered", "region_local"] | None
    ) = None,
) -> _UnitChunkedGraph:
  """Split semantic units, optionally scheduling FSDP collectives around them.

  A semantic unit is the indivisible group of body nodes for which
  ``unit_id_fn(node)`` returns the same non-negative ID; a negative ID marks
  unowned prefix or dependency-glue work. For example, ``_get_layer_id`` keeps
  every FX op attributed to ``layers.3`` together without coupling this
  algorithm to module-FQN metadata. The callback lets another grouping policy
  reuse the same collective-safety and GraphModule materialization logic.

  Sequential mode (``fsdp_collective_schedule=None``) backs forward layer
  splitting and ``backward_partition_mode=sequential``. It groups adjacent
  semantic units in graph order. When the next unit belongs to a later target
  partition, the traversal switches partitions only after every launched
  collective has reached its wait and it has left any TPU SPMD-safe region.
  Thus neither structure is cut between executables.

  FSDP-scheduled mode backs ``backward_partition_mode=fsdp_pipeline`` and
  ``fsdp_region_local``. It identifies each layer's complete AG and RS chain,
  then either staggers those chains around compute or assigns them to the same
  executable as that layer's compute.

  Algorithm:
  1. Apply ``unit_id_fn`` to each body node and identify complete collective
     chains, such as ``parameter -> cast -> AG -> wait`` and
     ``dWeight -> cast -> RS -> wait -> cast``.
  2. Derive forward or backward execution order from the collected unit IDs.
     FSDP scheduling also verifies that every model layer maps to exactly one
     compute unit and that those units cover non-overlapping layer ranges.
  3. Assign, validate, and normalize partitions. When
     ``fsdp_collective_schedule`` is set, it reassigns each complete AG/RS
     chain as described in detail above before assigning compute and glue;
     otherwise the sequential traversal advances only at a safe graph edge.
  4. Pass the node-to-partition mapping to PyTorch ``split_module``, which
     creates the parent GraphModule, child GraphModules, and lifted boundary
     inputs/outputs. Compilation happens later in ``GraphSplitter``.
  """
  if units_per_executable < 1:
    raise ValueError(
        "units_per_executable must be at least 1, got "
        f"{units_per_executable}"
    )
  if fsdp_collective_schedule not in (None, "staggered", "region_local"):
    raise ValueError(
        "fsdp_collective_schedule must be staggered, region_local, or None"
    )
  uses_fsdp_schedule = fsdp_collective_schedule is not None
  if uses_fsdp_schedule and (n_layers is None or n_layers < 1):
    raise ValueError(f"n_layers must be positive, got {n_layers}")
  if uses_fsdp_schedule and graph_name != "backward":
    raise ValueError("FSDP collective scheduling is backward-only")

  # Stage 1: label body nodes and identify indivisible collective chains.
  body_nodes = [
      node
      for node in gm.graph.nodes
      if node.op not in ("placeholder", "output")
  ]
  node_unit_ids = {
      node: unit_id_fn(node) for node in body_nodes
  }
  collective_chains = (
      _collect_collective_chains(gm) if uses_fsdp_schedule else {}
  )
  collective_nodes: set[torch.fx.Node] = set().union(
      *collective_chains.values()
  )
  unit_nodes: dict[int, set[torch.fx.Node]] = defaultdict(set)
  layer_to_unit: dict[int, int] = {}

  # Stage 2: derive execution order and validate scheduled layer ownership.
  if uses_fsdp_schedule:
    assert n_layers is not None
    for node in body_nodes:
      if node in collective_nodes:
        continue
      layer_id = _get_layer_id(node)
      if layer_id < 0:
        continue
      unit_id = node_unit_ids[node]
      if unit_id < 0:
        continue
      previous = layer_to_unit.setdefault(layer_id, unit_id)
      if previous != unit_id:
        raise ValueError(
            f"Layer {layer_id} spans {unit_name}s {previous} and {unit_id}"
        )
      unit_nodes[unit_id].add(node)

    missing_layers = sorted(set(range(n_layers)) - layer_to_unit.keys())
    if missing_layers:
      raise ValueError(
          f"FSDP {graph_name} split requires compute metadata for every "
          f"layer; missing layers {missing_layers}"
      )
    unit_layers: dict[int, set[int]] = defaultdict(set)
    for layer_id, unit_id in layer_to_unit.items():
      unit_layers[unit_id].add(layer_id)
    unit_ids = sorted(
        unit_layers,
        key=lambda unit_id: (
            max(unit_layers[unit_id])
            if descending
            else min(unit_layers[unit_id])
        ),
        reverse=descending,
    )
    for first, second in zip(unit_ids, unit_ids[1:]):
      first_layers = unit_layers[first]
      second_layers = unit_layers[second]
      overlaps = (
          min(first_layers) <= max(second_layers)
          if descending
          else max(first_layers) >= min(second_layers)
      )
      if overlaps:
        raise ValueError(
            f"FSDP {graph_name} split requires "
            f"{'descending' if descending else 'ascending'} {unit_name}s"
        )
  else:
    unit_ids = sorted(
        {unit_id for unit_id in node_unit_ids.values() if unit_id >= 0},
        reverse=descending,
    )
    for node, unit_id in node_unit_ids.items():
      if unit_id >= 0:
        unit_nodes[unit_id].add(node)

  if not unit_ids:
    raise ValueError(
        f"Cannot split {graph_name}: no {unit_name} metadata was found"
    )

  # Stage 3: assign every body node according to the selected mode.
  partition_map: dict[torch.fx.Node, int] = {}
  target_num_executables: int | None = None
  has_prefix = False
  raw_collective_layer_partitions: dict[str, dict[int, int]] = {}

  if uses_fsdp_schedule:
    assert n_layers is not None
    assert fsdp_collective_schedule is not None
    expected_collectives = ("ag", "rs")
    num_units = len(unit_ids)

    def compute_stage(index: int) -> int:
      if fsdp_collective_schedule == "staggered":
        return 2 + index // units_per_executable
      return 1 + index // units_per_executable

    def collective_stage(kind: str, index: int) -> int:
      if fsdp_collective_schedule == "region_local":
        return compute_stage(index)
      if kind == "ag":
        if index == 0:
          return 1
        return 2 + (index - 1) // units_per_executable
      if index == num_units - 1:
        return 3 + (num_units - 1) // units_per_executable
      return 2 + (index + 1) // units_per_executable

    unit_to_stage = {
        unit_id: compute_stage(index)
        for index, unit_id in enumerate(unit_ids)
    }
    collective_unit_to_stage = {
        kind: {
            unit_id: collective_stage(kind, index)
            for index, unit_id in enumerate(unit_ids)
        }
        for kind in expected_collectives
    }

    def assign_nodes_to_partition(
        nodes: set[torch.fx.Node], partition: int, owner: str
    ) -> None:
      conflicts = {
          node.name: partition_map[node]
          for node in nodes
          if node in partition_map and partition_map[node] != partition
      }
      if conflicts:
        raise ValueError(f"{owner} has conflicting ownership: {conflicts}")
      partition_map.update(dict.fromkeys(nodes, partition))

    _, analyzed_comms = _collect_bucketed_fsdp_comms(
        gm, n_layers=n_layers
    )
    analyzed_owners = {
        comm.launch: (comm.kind, comm.logical_index)
        for comm in analyzed_comms
        if comm.direction == "bwd"
        and comm.kind in expected_collectives
        and comm.logical_index is not None
        and comm.logical_index < n_layers
    }
    comm_counts = {
        layer_id: dict.fromkeys(expected_collectives, 0)
        for layer_id in range(n_layers)
    }
    for producer, chain in collective_chains.items():
      owner = analyzed_owners.get(producer)
      if owner is None:
        continue
      kind, layer_id = owner
      unit_id = layer_to_unit[layer_id]
      assign_nodes_to_partition(
          chain,
          collective_unit_to_stage[kind][unit_id],
          f"FSDP {kind} layer {layer_id}",
      )
      comm_counts[layer_id][kind] += 1

    missing_comms = {
        layer_id: tuple(kind for kind, count in counts.items() if count == 0)
        for layer_id, counts in comm_counts.items()
        if 0 in counts.values()
    }
    if missing_comms:
      expected_names = " and ".join(kind.upper() for kind in expected_collectives)
      raise ValueError(
          f"FSDP {graph_name} split requires {expected_names} ownership for "
          f"every layer; missing {missing_comms}"
      )

    for unit_id, nodes in unit_nodes.items():
      assign_nodes_to_partition(
          nodes,
          unit_to_stage[unit_id],
          f"{unit_name} {unit_id}",
      )

    for node in body_nodes:
      if node not in partition_map:
        partition_map[node] = max(
            (
                partition_map[input_node]
                for input_node in node.all_input_nodes
                if input_node in partition_map
            ),
            default=0,
        )

    for producer, chain in collective_chains.items():
      partitions = {partition_map[node] for node in chain}
      if len(partitions) != 1:
        raise ValueError(
            f"Collective {producer.name!r} spans FSDP {graph_name} "
            f"partitions {sorted(partitions)}"
        )
    for node, partition in partition_map.items():
      for input_node in node.all_input_nodes:
        input_partition = partition_map.get(input_node)
        if input_partition is not None and input_partition > partition:
          raise ValueError(
              f"FSDP {graph_name} split has a reverse dependency: "
              f"{input_node.name} ({input_partition}) -> "
              f"{node.name} ({partition})"
          )

    raw_collective_layer_partitions = {
        kind: {
            layer_id: collective_unit_to_stage[kind][unit_id]
            for layer_id, unit_id in layer_to_unit.items()
        }
        for kind in expected_collectives
    }
  else:
    first_unit_index = next(
        (
            index
            for index, node in enumerate(body_nodes)
            if node_unit_ids[node] >= 0
        ),
        None,
    )
    has_prefix = isolate_prefix and first_unit_index is not None and any(
        node.op == "call_function"
        for node in body_nodes[:first_unit_index]
    )
    unit_partition_offset = int(has_prefix)
    num_unit_executables = (
        len(unit_ids) + units_per_executable - 1
    ) // units_per_executable
    target_num_executables = (
        num_unit_executables + unit_partition_offset
    )
    unit_to_target_partition = {
        unit_id: unit_partition_offset + index // units_per_executable
        for index, unit_id in enumerate(unit_ids)
    }
    node_to_target_partition = {
        node: unit_to_target_partition[unit_id]
        for unit_id, nodes in unit_nodes.items()
        for node in nodes
    }

    current_partition = 0
    target_partition = 0
    current_unit_id = -1
    collective_wait_counts = Counter(
        producer
        for node in body_nodes
        if is_wait_tensor(node)
        and (producer := _collective_producer(node)) is not None
    )
    pending_collectives: dict[torch.fx.Node, int] = {}
    spmd_safe_depth = 0
    for node in body_nodes:
      node_unit_id = node_unit_ids[node]
      starts_unit = node_unit_id >= 0 and node_unit_id != current_unit_id
      if starts_unit:
        current_unit_id = node_unit_id
      if node in node_to_target_partition:
        target_partition = max(
            target_partition, node_to_target_partition[node]
        )
      if starts_unit and not pending_collectives and spmd_safe_depth == 0:
        current_partition = target_partition

      collective_producer = _collective_producer(node)
      if is_wait_tensor(node):
        if collective_producer not in partition_map:
          raise ValueError(
              f"Collective producer for {node.name!r} is not partitioned"
          )
        partition_map[node] = partition_map[collective_producer]
        remaining_waits = pending_collectives[collective_producer] - 1
        if remaining_waits:
          pending_collectives[collective_producer] = remaining_waits
        else:
          del pending_collectives[collective_producer]
      elif _is_getitem(node) and collective_producer is not None:
        if collective_producer not in partition_map:
          raise ValueError(
              f"Collective producer for {node.name!r} is not partitioned"
          )
        partition_map[node] = partition_map[collective_producer]
      else:
        partition_map[node] = current_partition
        if _is_collective(node):
          pending_collectives[node] = max(1, collective_wait_counts[node])

      if _is_spmd_region_marker(node, "enter_spmd_safe_region"):
        spmd_safe_depth += 1
      elif _is_spmd_region_marker(node, "exit_spmd_safe_region"):
        spmd_safe_depth -= 1
        if spmd_safe_depth < 0:
          raise ValueError("Unbalanced TPU SPMD safe-region exit")

    if pending_collectives:
      raise ValueError(
          f"{graph_name.capitalize()} graph has "
          f"{len(pending_collectives)} unwaited collectives"
      )
    if spmd_safe_depth:
      raise ValueError("Unbalanced TPU SPMD safe-region entry")

  # Stage 3 continued: normalize IDs and verify each unit remains atomic.
  actual_partitions = sorted(set(partition_map.values()))
  normalized_partitions = {
      partition: index for index, partition in enumerate(actual_partitions)
  }
  partition_map = {
      node: normalized_partitions[partition]
      for node, partition in partition_map.items()
  }
  unit_to_partition = {}
  for unit_id, nodes in unit_nodes.items():
    unit_partitions = {partition_map[node] for node in nodes}
    if len(unit_partitions) != 1:
      raise ValueError(
          f"{graph_name.capitalize()} {unit_name} {unit_id} spans "
          f"partitions {sorted(unit_partitions)}"
      )
    unit_to_partition[unit_id] = next(iter(unit_partitions))
  collective_layer_to_partition = {
      kind: {
          layer_id: normalized_partitions[partition]
          for layer_id, partition in layer_partitions.items()
      }
      for kind, layer_partitions in raw_collective_layer_partitions.items()
  }

  actual_num_executables = len(actual_partitions)
  if (
      target_num_executables is not None
      and actual_num_executables != target_num_executables
  ):
    logger.warning(
        "Targeted %d TPU %s executables, produced %d after preserving %s, "
        "collective, and SPMD-safe boundaries",
        target_num_executables,
        graph_name,
        actual_num_executables,
        unit_name,
    )

  # Stage 4: materialize child GraphModules and retain audit metadata.
  chunked_gm = split_module(gm, None, lambda node: partition_map[node])
  chunked_gm.graph.lint()
  chunked_gm.recompile()
  if uses_fsdp_schedule:
    logger.info(
        "Split TPU %s into %d FSDP-scheduled executables with at most %d "
        "%ss per executable: compute=%s collectives=%s",
        graph_name,
        actual_num_executables,
        units_per_executable,
        unit_name,
        unit_to_partition,
        collective_layer_to_partition,
    )
  else:
    node_counts = [0] * actual_num_executables
    chunk_units: list[set[int]] = [
        set() for _ in range(actual_num_executables)
    ]
    for node, partition in partition_map.items():
      node_counts[partition] += 1
      unit_id = node_unit_ids[node]
      if unit_id >= 0:
        chunk_units[partition].add(unit_id)
    unit_ranges = [
        (min(units), max(units)) if units else None
        for units in chunk_units
    ]
    logger.info(
        "Split TPU %s graph into %d executables (prefix=%s): nodes=%s "
        "%ss=%s",
        graph_name,
        actual_num_executables,
        has_prefix,
        node_counts,
        unit_name,
        unit_ranges,
    )
  return _UnitChunkedGraph(
      module=chunked_gm,
      partition_map=partition_map,
      unit_to_partition=unit_to_partition,
      collective_layer_to_partition=collective_layer_to_partition,
  )


def split_backward_graph_by_fsdp_pipeline(
    gm: torch.fx.GraphModule,
    *,
    n_layers: int,
    regions_per_executable: int = 1,
    collective_ownership: Literal["staggered", "region_local"] = "staggered",
) -> FSDPPipelineChunkedGraph:
  """Create AG / compute / RS pipeline executables for backward.

  Rematerialization regions and collective chains remain atomic. Staggered
  ownership overlaps each region's compute with adjacent AG and RS work;
  region-local ownership keeps all three operations in the same executable.
  """
  chunked = _split_graph_by_units(
      gm,
      unit_id_fn=_get_layer_id,
      unit_name="remat region",
      descending=True,
      graph_name="backward",
      isolate_prefix=True,
      units_per_executable=regions_per_executable,
      n_layers=n_layers,
      fsdp_collective_schedule=collective_ownership,
  )
  return FSDPPipelineChunkedGraph(
      module=chunked.module,
      partition_map=chunked.partition_map,
      compute_region_to_partition=chunked.unit_to_partition,
      ag_layer_to_partition=chunked.collective_layer_to_partition["ag"],
      rs_layer_to_partition=chunked.collective_layer_to_partition["rs"],
  )


def split_backward_graph_by_remat_region(
    gm: torch.fx.GraphModule,
    regions_per_executable: int,
) -> SequentialChunkedGraph:
  """Split backward into descending, complete rematerialization regions."""
  return _split_graph_by_units(
      gm,
      unit_id_fn=_get_layer_id,
      unit_name="remat region",
      descending=True,
      graph_name="backward",
      isolate_prefix=True,
      units_per_executable=regions_per_executable,
  )


def split_forward_graph_by_layer(
    gm: torch.fx.GraphModule,
    layers_per_executable: int,
    *,
    isolate_prefix: bool = False,
) -> SequentialChunkedGraph:
  """Split forward into ascending, complete transformer-layer ranges."""
  return _split_graph_by_units(
      gm,
      unit_id_fn=_get_layer_id,
      unit_name="layer",
      descending=False,
      graph_name="forward",
      isolate_prefix=isolate_prefix,
      units_per_executable=layers_per_executable,
  )


def split_forward_backward_graph(
    gm: torch.fx.GraphModule,
    example_inputs: tuple[Any, ...],
) -> SplitForwardBackwardGraph:
  """Partition a make_fx joint graph using ``autograd_backward`` metadata."""
  nodes = list(gm.graph.nodes)
  placeholders = [node for node in nodes if node.op == "placeholder"]
  if len(placeholders) != len(example_inputs):
    raise ValueError(
        "Graph input count does not match example inputs: "
        f"{len(placeholders)} != {len(example_inputs)}"
    )

  output_node = next(node for node in nodes if node.op == "output")
  original_outputs = _as_tuple(output_node.args[0])
  backward_node_set: set[torch.fx.Node] = set()
  for node in nodes:
    if node.op == "output":
      continue
    if _is_backward_node(node) or any(
        input_node in backward_node_set for input_node in node.all_input_nodes
    ):
      backward_node_set.add(node)
  backward_nodes = [node for node in nodes if node in backward_node_set]
  if not backward_nodes:
    raise ValueError("Cannot split a graph with no autograd_backward nodes")

  residual_nodes: list[torch.fx.Node] = []
  residual_set: set[torch.fx.Node] = set()
  backward_recomputed_nodes: set[torch.fx.Node] = set()
  missing = object()

  def make_available_to_backward(node: torch.fx.Node) -> None:
    if node in backward_node_set or node.op in ("placeholder", "get_attr"):
      return
    value = node.meta.get("val", missing)
    if value is missing:
      raise ValueError(
          f"Graph split boundary node {node.name!r} has no example value"
      )
    if isinstance(value, torch.Tensor):
      if node not in residual_set:
        residual_set.add(node)
        residual_nodes.append(node)
      return
    if node in backward_recomputed_nodes:
      return
    backward_recomputed_nodes.add(node)
    for input_node in node.all_input_nodes:
      make_available_to_backward(input_node)

  for node in backward_nodes:
    for input_node in node.all_input_nodes:
      make_available_to_backward(input_node)

  forward_output_positions: list[int] = []
  backward_output_positions: list[int] = []
  constant_outputs: dict[int, Any] = {}
  forward_output_nodes: list[torch.fx.Node] = []
  backward_output_nodes: list[torch.fx.Node] = []
  for position, output in enumerate(original_outputs):
    if not isinstance(output, torch.fx.Node):
      constant_outputs[position] = output
    elif (
        "val" in output.meta
        and not isinstance(output.meta["val"], torch.Tensor)
    ):
      constant_outputs[position] = output.meta["val"]
    elif output in backward_node_set:
      backward_output_positions.append(position)
      backward_output_nodes.append(output)
    else:
      forward_output_positions.append(position)
      forward_output_nodes.append(output)

  forward_roots = [*forward_output_nodes, *residual_nodes]
  forward_needed: set[torch.fx.Node] = set(placeholders)

  def add_forward_dependencies(node: torch.fx.Node) -> None:
    if node in forward_needed:
      return
    if node in backward_node_set:
      raise ValueError(
          f"Forward output {node.name!r} depends on a backward graph node"
      )
    forward_needed.add(node)
    for input_node in node.all_input_nodes:
      add_forward_dependencies(input_node)

  for root in forward_roots:
    add_forward_dependencies(root)

  forward_fx = torch.fx.Graph()
  forward_env: dict[torch.fx.Node, torch.fx.Node] = {}
  for node in nodes:
    if node.op == "output" or node not in forward_needed:
      continue
    forward_env[node] = forward_fx.node_copy(node, forward_env.__getitem__)
  forward_fx.output(
      tuple(forward_env[node] for node in forward_roots)
  )
  forward_gm = torch.fx.GraphModule(gm, forward_fx)
  forward_gm.graph.lint()
  forward_gm.recompile()

  backward_fx = torch.fx.Graph()
  backward_env: dict[torch.fx.Node, torch.fx.Node] = {}
  for node in placeholders:
    backward_env[node] = backward_fx.node_copy(node, lambda _: None)
  backward_get_attrs = {
      input_node
      for copied_node in [*backward_recomputed_nodes, *backward_nodes]
      for input_node in copied_node.all_input_nodes
      if input_node.op == "get_attr"
  }
  for node in nodes:
    if node in backward_get_attrs:
      backward_env[node] = backward_fx.node_copy(node, backward_env.__getitem__)
  for residual in residual_nodes:
    placeholder = backward_fx.placeholder(f"residual_{residual.name}")
    placeholder.meta = dict(residual.meta)
    backward_env[residual] = placeholder
  for node in nodes:
    if node in backward_recomputed_nodes or node in backward_node_set:
      backward_env[node] = backward_fx.node_copy(node, backward_env.__getitem__)
  backward_fx.output(tuple(backward_env[node] for node in backward_output_nodes))
  backward_gm = torch.fx.GraphModule(gm, backward_fx)
  backward_gm.graph.lint()
  backward_gm.recompile()

  residual_examples = tuple(node.meta["val"] for node in residual_nodes)
  logger.info(
      "Split TPU graph into %d forward and %d backward nodes with %d tensor "
      "residuals",
      len(list(forward_gm.graph.nodes)),
      len(list(backward_gm.graph.nodes)),
      len(residual_nodes),
  )
  module = SplitForwardBackwardModule(
      forward_gm,
      backward_gm,
      forward_output_positions=tuple(forward_output_positions),
      backward_output_positions=tuple(backward_output_positions),
      constant_outputs=constant_outputs,
      num_direct_forward_outputs=len(forward_output_nodes),
      num_outputs=len(original_outputs),
  )
  return SplitForwardBackwardGraph(
      module=module,
      forward_graph=forward_gm,
      backward_graph=backward_gm,
      forward_example_inputs=example_inputs,
      backward_example_inputs=(*example_inputs, *residual_examples),
      residual_nodes=tuple(residual_nodes),
  )


__all__ = [
    "FSDPPipelineChunkedGraph",
    "GraphSplitter",
    "SequentialChunkedGraph",
    "SplitForwardBackwardGraph",
    "SplitForwardBackwardModule",
    "split_backward_graph_by_fsdp_pipeline",
    "split_backward_graph_by_remat_region",
    "split_forward_backward_graph",
    "split_forward_graph_by_layer",
]
