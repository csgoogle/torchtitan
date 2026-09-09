from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch import nn
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    annotate_module_fqns,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _full_memory_policy_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.experiments.tpu.graph_trainer import graph_split as graph_split_lib
from torchtitan.experiments.tpu.graph_trainer.graph_split import (
    split_backward_graph_by_fsdp_pipeline,
    split_backward_graph_by_remat_region,
    split_forward_graph_by_layer,
    split_forward_backward_graph,
)


class _ToyTransformer(nn.Module):

  def __init__(self, n_layers: int) -> None:
    super().__init__()
    self.layers = nn.ModuleList(
        nn.Linear(8, 8, bias=False) for _ in range(n_layers)
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    inputs = torch.sin(inputs)
    for layer in self.layers:
      inputs = torch.relu(layer(inputs))
    return inputs


class GraphSplitTest(absltest.TestCase):

  def _build_toy_model(self, n_layers: int) -> _ToyTransformer:
    """Build the shared prefix-plus-transformer-layer test model."""
    model = _ToyTransformer(n_layers)
    annotate_module_fqns(model)
    return model

  def _trace_toy_model(
      self,
      *,
      model: nn.Module,
      inputs: torch.Tensor,
      gradient_epilogue: bool = False,
      use_shape_scalar: bool = False,
  ):
    """Trace loss and explicit gradients for the shared test model."""

    def step(inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
      loss = model(inputs).float().sum()
      gradients = torch.autograd.grad(
          loss, (*tuple(model.parameters()), inputs)
      )
      if gradient_epilogue:
        gradients = tuple(
            torch.ops.aten.alias.default(gradient) for gradient in gradients
        )
      if use_shape_scalar:
        gradients = tuple(gradient * inputs.shape[0] for gradient in gradients)
      return (loss, *gradients)

    return minimal_fx_tracer(step, module=model)(inputs)

  def _apply_passes_and_split(
      self,
      traced,
      *,
      fsdp_reshard_after_forward: str,
  ):
    """Apply full remat, then split the real joint trace at fwd/bwd."""
    _full_memory_policy_pass(
        traced.gm,
        config=SimpleNamespace(
            parallelism=SimpleNamespace(
                fsdp_reshard_after_forward=fsdp_reshard_after_forward,
                pipeline_parallel_degree=1,
            )
        ),
    )
    selective_activation_remat_pass(traced.gm)
    return split_forward_backward_graph(traced.gm, traced.example_inputs)

  def _trace_simple_fsdp(
      self,
      *,
      n_layers: int,
      reduce_dtype: torch.dtype = torch.float32,
  ):
    """Trace the shared model through real SimpleFSDP with fake collectives."""

    self.assertFalse(dist.is_initialized())
    dist.init_process_group(
        "fake", store=FakeStore(), rank=0, world_size=2
    )
    try:
      mesh = DeviceMesh("cpu", [0, 1])
      with torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True):
        model = self._build_toy_model(n_layers)
        model = data_parallel(
            model,
            device_mesh=mesh,
            mode="fully_shard",
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=reduce_dtype,
            ),
        )
        inputs = torch.randn(
            2, 8, dtype=torch.bfloat16, requires_grad=True
        )

        traced = self._trace_toy_model(model=model, inputs=inputs)
    finally:
      dist.destroy_process_group()

    return self._apply_passes_and_split(
        traced,
        fsdp_reshard_after_forward="always",
    )

  def _assert_collective_chains_are_atomic(
      self,
      gm: torch.fx.GraphModule,
      partition_map: dict[torch.fx.Node, int],
  ) -> None:
    chains = graph_split_lib._collect_collective_chains(gm)
    self.assertNotEmpty(chains)
    for launch, nodes in chains.items():
      self.assertEqual(
          {partition_map[node] for node in nodes},
          {partition_map[launch]},
          f"collective chain rooted at {launch.name} was split",
      )

  def test_forward_backward_split_preserves_outputs_and_ordering(self):
    """Preserve loss and every gradient while running forward before backward.

    The traced result is ``(loss, parameter gradients..., input gradient)``.
    Splitting must preserve every value, keep autograd work in backward, and
    enqueue all forward children before any backward collective can run.
    """
    model = self._build_toy_model(n_layers=2)
    inputs = torch.randn(6, 8, requires_grad=True)
    traced = self._trace_toy_model(model=model, inputs=inputs)
    expected = run_traced(traced, module=model)(inputs)

    split = self._apply_passes_and_split(
        traced,
        fsdp_reshard_after_forward="never",
    )
    traced.gm = split.module
    with mock.patch.object(
        graph_split_lib, "_enqueue_forward_graph_outputs"
    ) as enqueue:
      actual = run_traced(traced, module=model)(inputs)

    self.assertLen(actual, len(expected))
    output_names = (
        "loss",
        *(f"parameter gradient {i}" for i, _ in enumerate(model.parameters())),
        "input gradient",
    )
    for name, expected_value, actual_value in zip(
        output_names, expected, actual, strict=True
    ):
      with self.subTest(output=name):
        torch.testing.assert_close(actual_value, expected_value)
    self.assertLen(output_names, len(actual))
    self.assertTrue(
        all(
            not node.meta.get("autograd_backward", False)
            for node in split.forward_graph.graph.nodes
        )
    )
    self.assertNotEmpty(
        [
            node
            for node in split.backward_graph.graph.nodes
            if node.meta.get("autograd_backward", False)
        ]
    )
    enqueue.assert_called_once()
    enqueued_outputs = enqueue.call_args.args[0]
    self.assertLen(
        enqueued_outputs,
        split.module.num_direct_forward_outputs + len(split.residual_nodes),
    )
    self.assertIs(enqueued_outputs[0], actual[0])

  def test_forward_backward_boundary_edge_cases(self):
    """Keep gradient epilogues and dynamic scalar uses on the backward side."""
    model = self._build_toy_model(n_layers=2)
    with self.subTest("untagged gradient epilogue stays in backward"):
      inputs = torch.randn(2, 8, requires_grad=True)
      traced = self._trace_toy_model(
          model=model,
          inputs=inputs,
          gradient_epilogue=True,
      )
      split = self._apply_passes_and_split(
          traced,
          fsdp_reshard_after_forward="never",
      )
      aliases = [
          node
          for node in traced.gm.graph.nodes
          if node.op == "call_function"
          and node.target is torch.ops.aten.alias.default
      ]
      backward_names = {
          node.name for node in split.backward_graph.graph.nodes
      }
      self.assertNotEmpty(aliases)
      self.assertTrue(
          all(alias.name in backward_names for alias in aliases),
      )

    with self.subTest("dynamic scalar is recomputed instead of saved"):
      inputs = torch.randn(2, 8, requires_grad=True)
      torch._dynamo.mark_dynamic(inputs, 0)
      traced = self._trace_toy_model(
          model=model,
          inputs=inputs,
          use_shape_scalar=True,
      )
      split = self._apply_passes_and_split(
          traced,
          fsdp_reshard_after_forward="never",
      )
      size_nodes = [
          node
          for node in traced.gm.graph.nodes
          if node.target is torch.ops.aten.sym_size.int
      ]
      self.assertLen(size_nodes, 1)
      self.assertNotIn(size_nodes[0], split.residual_nodes)
      self.assertIn(
          size_nodes[0].name,
          {node.name for node in split.backward_graph.graph.nodes},
      )

  def test_partition_layouts_reuse_one_simple_fsdp_trace(self):
    """Check sequential, pipelined, and region-local four-layer layouts.

    Forward visits ``L0 -> L1 -> L2 -> L3``; backward visits the layers in
    reverse. ``AGn``, layer ``Ln`` compute, ``RSn``, and every collective's
    launch/getitem/wait chain must retain the ownership shown below.
    """
    split = self._trace_simple_fsdp(n_layers=4)
    forward = split.forward_graph
    backward = split.backward_graph
    self.assertFalse(
        any("fsdp_bucket" in node.meta for node in backward.graph.nodes)
    )

    # Sequential backward: [prefix] [L3,L2] [L1,L0].
    sequential = split_backward_graph_by_remat_region(backward, 2)
    self.assertEqual(
        sequential.unit_to_partition,
        {3: 1, 2: 1, 1: 2, 0: 2},
    )
    partitionable = [
        node
        for node in backward.graph.nodes
        if node.op not in ("placeholder", "output")
    ]
    first_layer_index = next(
        index
        for index, node in enumerate(partitionable)
        if _get_layer_id(node) >= 0
    )
    prefix_nodes = partitionable[:first_layer_index]
    self.assertNotEmpty(prefix_nodes)
    self.assertTrue(
        all(sequential.partition_map[node] == 0 for node in prefix_nodes)
    )
    self._assert_collective_chains_are_atomic(
        backward, sequential.partition_map
    )

    forward_scenarios = (
        # [prefix,L0,L1] [L2,L3]
        (2, False, {0: 0, 1: 0, 2: 1, 3: 1}),
        # [prefix] [L0] [L1] [L2] [L3]
        (1, True, {0: 1, 1: 2, 2: 3, 3: 4}),
    )
    for layers_per_executable, isolate_prefix, expected in forward_scenarios:
      with self.subTest(
          direction="forward",
          layers_per_executable=layers_per_executable,
          isolate_prefix=isolate_prefix,
      ):
        chunked = split_forward_graph_by_layer(
            forward,
            layers_per_executable,
            isolate_prefix=isolate_prefix,
        )
        self.assertEqual(chunked.unit_to_partition, expected)

    local_one = (1, 2, 3, 4)
    local_three = (1, 1, 1, 2)
    pipeline_scenarios = (
        # [AG3] [AG2,C3] [AG1,RS3,C2] ... [RS0]
        (
            "staggered-one-layer",
            1,
            "staggered",
            (2, 3, 4, 5),
            (1, 2, 3, 4),
            (3, 4, 5, 6),
        ),
        # Coalesce pairs of compute stages while preserving pipeline fill/drain.
        (
            "staggered-two-layers",
            2,
            "staggered",
            (2, 2, 3, 3),
            (1, 2, 2, 3),
            (2, 3, 3, 4),
        ),
        # Each layer owns its AG, compute, and RS.
        (
            "region-local-one-layer",
            1,
            "region_local",
            local_one,
            local_one,
            local_one,
        ),
        # Region-local ownership accepts widths >1 and an incomplete tail.
        (
            "region-local-three-layers",
            3,
            "region_local",
            local_three,
            local_three,
            local_three,
        ),
    )
    for name, width, ownership, compute, ag, rs in pipeline_scenarios:
      with self.subTest(name):
        chunked = split_backward_graph_by_fsdp_pipeline(
            backward,
            n_layers=4,
            regions_per_executable=width,
            collective_ownership=ownership,
        )
        layer_ids = (3, 2, 1, 0)
        self.assertEqual(
            chunked.compute_region_to_partition,
            dict(zip(layer_ids, compute, strict=True)),
        )
        self.assertEqual(
            chunked.ag_layer_to_partition,
            dict(zip(layer_ids, ag, strict=True)),
        )
        self.assertEqual(
            chunked.rs_layer_to_partition,
            dict(zip(layer_ids, rs, strict=True)),
        )
        self._assert_collective_chains_are_atomic(
            backward, chunked.partition_map
        )

    # Missing layer ownership must fail instead of silently cutting by position.
    for node in backward.graph.nodes:
      node.meta.pop("custom", None)
    with self.assertRaisesRegex(ValueError, "no remat region metadata"):
      split_backward_graph_by_remat_region(backward, 1)

  def test_backward_fsdp_pipeline_keeps_bf16_reduce_cast_with_wait(self):
    """Keep ``dW -> RS(BF16) -> wait -> cast(FP32)`` in one executable."""
    gm = self._trace_simple_fsdp(
        n_layers=2,
        reduce_dtype=torch.bfloat16,
    ).backward_graph

    chunked = split_backward_graph_by_fsdp_pipeline(gm, n_layers=2)

    reduce_casts = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.aten._to_copy.default
        and len(node.all_input_nodes) == 1
        and graph_split_lib.is_wait_tensor(node.all_input_nodes[0])
    ]
    self.assertLen(reduce_casts, 2)
    for cast in reduce_casts:
      wait = cast.all_input_nodes[0]
      self.assertEqual(
          chunked.partition_map[cast], chunked.partition_map[wait]
      )


if __name__ == "__main__":
  absltest.main()
