# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration and validation owned by TPU GraphTrainer."""

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    GraphTrainerMemoryPolicy,
)
from torchtitan.experiments.tpu.graph_trainer.layer_stack import (
    with_tpu_stacked_layers,
)
from torchtitan.experiments.tpu.graph_trainer.scan_guards import (
    check_scan_parallelism,
)
from torchtitan.experiments.tpu.layer_loop import builds_layer_loop

if TYPE_CHECKING:
  from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
      TpuGraphTrainer,
  )


GraphTrainerFSDPBarrierPolicy = Literal["none", "prefetch", "interleave"]
BackwardPartitionMode = Literal["none", "by_layer", "fsdp_pipeline"]
TpuGraphTrainerMemoryPolicy = Literal[
    GraphTrainerMemoryPolicy, "full_and_offload", "full_and_scan_offload"
]


@dataclass(kw_only=True, slots=True)
class GraphSplitterConfig:
  """Partition a joint forward/backward trace into TPU executables.

  Splitting is disabled by default. When ``enabled`` is false, all other
  fields in this config are ignored. When enabled with
  ``backward_partition_mode='none'``, GraphSplitter still makes separate
  top-level forward and backward executables; the mode only controls finer
  subdivision of the backward executable.

  ``GraphSplitterConfig()`` performs no split because ``enabled`` defaults to
  false. Enabling it with all other defaults produces one forward executable
  and one backward executable. ``backward_partition_mode='by_layer'`` or
  ``'fsdp_pipeline'`` additionally partitions backward, using
  ``backward_layers_per_executable`` as the grouping size.

  Layerwise remat barriers may run before splitting and can still help inside
  a child containing multiple layers. ``fsdp_barrier_policy`` is rejected with
  GraphSplitter: it moves collectives across layers on the complete graph and
  is not yet aware of the later executable boundaries.
  """

  enabled: bool = False
  # Maximum number of complete transformer layers assigned to one forward
  # executable. -1 keeps forward as one executable.
  forward_layers_per_executable: int = -1
  # Assign non-layer work before the first transformer layer to its own partition.
  forward_isolate_prefix: bool = False
  # Backward subdivision applied after the top-level forward/backward split.
  # ``by_layer`` groups adjacent layer backward computations and co-locates any
  # attributable FSDP collective chains with their layer. ``fsdp_pipeline``
  # instead moves each layer's AG earlier and RS later to form overlap stages.
  backward_partition_mode: BackwardPartitionMode = "none"
  # Maximum number of complete transformer layers assigned to one backward
  # executable. Grouping layers widens the compiler scheduling window without
  # changing activation-checkpoint boundaries.
  backward_layers_per_executable: int = 1
  # Reuse dead internal split-boundary buffers through XLA input donation.
  # This remains False until TorchTPU includes donation in its compilation
  # cache key (torch_tpu#3614).
  donate_internal_buffers: bool = False
  # Number of complete backward child executables by which to advance each H2D
  # reload pair relative to its first consumer.
  cpu_offload_prefetch_executables: int = 0


@dataclass(kw_only=True, slots=True)
class TpuGraphTrainerCompileConfig(GraphTrainerCompileConfig):
  """Compilation and graph-transformation policy for TPU GraphTrainer."""

  enable: bool = False
  """Enable the upstream GraphTrainer compilation path.

  This field is inherited from ``CompileConfig`` and repeated here to make the
  public TPU surface explicit. ``mode`` selects JIT or joint-step AOT tracing.
  """
  backend: str = "tpu"
  pass_pipeline: str = "tpu"
  memory_policy: TpuGraphTrainerMemoryPolicy = "default"
  # Compile AdamW into its own FX graph, separate from the joint training graph.
  compile_optimizer: bool = False
  # Compile the DTensor-aware global gradient norm as a separate static graph.
  compile_grad_norm: bool = False
  # Compile the optimizer step into the joint forward/loss/backward graph.
  optimizer_in_graph: bool = False
  # Stack decoder-layer parameters and execute the layers with one scan.
  # Checkpoint keys remain per-layer; optimizer checkpoint names change.
  use_scan_layers: bool = False
  # Make each layer's local parameter gradients a dependency of the next
  # layer's recomputation. This prevents XLA from starting the next layer and
  # retaining both layers' recompute workspaces at once. It can lower peak HBM
  # for an unsplit or multi-layer executable, but can reduce scheduling freedom;
  # a one-layer GraphSplitter child already supplies the same lifetime boundary.
  # ``memory_policy`` independently decides what is saved or recomputed, so
  # these barriers do nothing when rematerialization is disabled.
  enable_remat_barriers: bool = False
  # Bound recomputed SimpleFSDP parameter lifetimes and place their collectives
  # near dense backward work. ``prefetch`` permits AG/compute overlap within a
  # bounded window. ``interleave`` additionally makes each next AG wait for the
  # prior RS, reducing HBM at the cost of AG/RS concurrency. This requires
  # ``enable_remat_barriers`` and is not yet partition-aware, so it cannot be
  # combined with GraphSplitter.
  fsdp_barrier_policy: GraphTrainerFSDPBarrierPolicy = "none"
  # Release layer L's recomputed parameters at backward layer L+N.
  fsdp_barrier_prefetch_distance: int = 1
  graph_split: GraphSplitterConfig = field(default_factory=GraphSplitterConfig)
  # Lower selected FSDP gradient reduce-scatters as all-reduce plus rank-local
  # slice. Patterns match parameter FQNs; an empty tuple disables the rewrite.
  fsdp_allreduce_slice_parameter_patterns: tuple[str, ...] = ()
  fsdp_allreduce_slice_max_input_bytes: int = 0


def _validate_fsdp_allreduce_slice(config: "TpuGraphTrainer.Config") -> None:
  patterns = config.compile.fsdp_allreduce_slice_parameter_patterns
  if not patterns:
    return

  if config.compile.fsdp_allreduce_slice_max_input_bytes <= 0:
    raise ValueError(
        "fsdp all-reduce-plus-slice requires a positive maximum input size"
    )

  parallelism = config.parallelism
  if not parallelism.use_simple_fsdp:
    raise ValueError("fsdp all-reduce-plus-slice requires SimpleFSDP")
  if (
      parallelism.data_parallel_replicate_degree != 1
      or parallelism.data_parallel_shard_degree != -1
      or parallelism.tensor_parallel_degree != 1
      or parallelism.context_parallel_degree != 1
      or parallelism.pipeline_parallel_degree != 1
      or parallelism.expert_parallel_degree != 1
  ):
    raise ValueError(
        "fsdp all-reduce-plus-slice currently requires full-world FSDP "
        "with DP-replicate/TP/CP/PP/EP degrees all equal to 1"
    )
  if config.training.mixed_precision_reduce != "float32":
    raise ValueError(
        "fsdp all-reduce-plus-slice currently requires FP32 FSDP reductions"
    )


def apply_tpu_graph_trainer_model_transforms(
    config: "TpuGraphTrainer.Config",
) -> None:
  """Validate TPU graph options and apply CLI-selected model transforms.

  Recipes construct the final ``TpuGraphTrainer.Config`` before CLI parsing,
  but scan is itself a model-class choice and the CLI value is only known
  afterward. TpuGraphTrainer calls this once, before its base class builds the
  model, so ``compile.use_scan_layers`` can select the scanned model class.
  """
  _validate_fsdp_allreduce_slice(config)

  compile_config = config.compile
  if compile_config.optimizer_in_graph and compile_config.compile_optimizer:
    raise ValueError(
        "compile.compile_optimizer cannot be combined with "
        "compile.optimizer_in_graph"
    )
  if compile_config.optimizer_in_graph and compile_config.compile_grad_norm:
    raise ValueError(
        "compile.compile_grad_norm cannot be combined with "
        "compile.optimizer_in_graph"
    )
  if compile_config.fsdp_barrier_policy != "none" and (
      compile_config.memory_policy in (None, "none")
      or not compile_config.enable_remat_barriers
  ):
    raise ValueError(
        "compile.fsdp_barrier_policy requires GraphTrainer "
        "rematerialization and compile.enable_remat_barriers"
    )
  if (
      compile_config.fsdp_barrier_policy != "none"
      and compile_config.fsdp_barrier_prefetch_distance < 1
  ):
    raise ValueError(
        "compile.fsdp_barrier_prefetch_distance must be positive"
    )
  if (
      config.parallelism.fsdp_reshard_after_forward == "always"
      and not compile_config.use_scan_layers
      and not compile_config.graph_split.enabled
      and compile_config.fsdp_barrier_policy == "none"
  ):
    raise ValueError(
        "TPU GraphTrainer ZeRO-3 requires compile.use_scan_layers, "
        "compile.graph_split.enabled, or compile.fsdp_barrier_policy="
        "'prefetch'/'interleave'; without one, XLA can merge forward and "
        "recomputed all-gathers and retain full parameters through backward."
    )

  memory_policy = compile_config.memory_policy
  if (
      compile_config.use_scan_layers
      and memory_policy.endswith("_and_offload")
      and memory_policy != "full_and_scan_offload"
  ):
    raise ValueError(
        f"generic offload policy {memory_policy!r} is unsupported with scan; "
        "use the scan-specific 'full_and_scan_offload' policy"
    )
  if compile_config.cpu_offload_budget_gb < 0:
    raise ValueError("CPU-offload budget must be nonnegative")

  if not compile_config.use_scan_layers:
    return

  check_scan_parallelism(config.parallelism)
  if config.parallelism.expert_parallel_degree > 1:
    raise ValueError(
        "--compile.use-scan-layers cannot be combined with "
        "--parallelism.expert-parallel-degree > 1 "
        "(https://github.com/google-pytorch/torchtitan/issues/395)."
    )
  assert config.model_spec is not None
  config.model_spec = dataclasses.replace(
      config.model_spec,
      model=with_tpu_stacked_layers(config.model_spec.model, use_scan=True),
  )
  if not builds_layer_loop(config.model_spec.model):
    raise AssertionError("--compile.use-scan-layers did not produce a scan")


__all__ = [
    "GraphSplitterConfig",
    "GraphTrainerFSDPBarrierPolicy",
    "TpuGraphTrainerCompileConfig",
    "TpuGraphTrainerMemoryPolicy",
    "apply_tpu_graph_trainer_model_transforms",
]
