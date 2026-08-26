# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the in-body FSDP gather that makes scan-over-layers fit at 70B.

`test_scan_lowering.py` proves the layer stack becomes one `stablehlo.while`.
That is only half the claim. The other half is *where the FSDP all-gather
happens*: SimpleFSDP installs its gather as a class property on the parameter
name, so a single `getattr` on a stacked parameter gathers all N layers at
graph-build time — 136.9 GB of bf16 weights at LLaMA-3 70B against a 96 GB
budget, i.e. OOM before the first activation. The gather has to happen one layer
at a time, *inside* the loop body. See `torchtitan/experiments/tpu/scan/fsdp.py`.

Three invariants are checked here, all of them silent-failure modes — each one
regresses into a working run that is merely 80x too large, or into a run whose
per-iteration slice crosses a sharded axis:

1. the stacked parameter is sharded on dim 1, never on dim 0 (the scan axis);
2. `plan_stacked_gathers` rejects, loudly, every placement the in-body path
   cannot express;
3. the all-gathers land inside the scan body and there are none above it.

CPU-only: every invariant here is structural, so a gloo world resolves the
placements, the group name, and the traced collective exactly as an 8-rank TPU
world does. Invariants 1 and 2 run in a world of one, which is enough for them.
Invariant 3 — `InBodyCollectiveTest`, the one that actually guards the 136.9 GB
regression — spawns a real world of **four**, and the size is load-bearing, not
incidental: on a mesh of one, SimpleFSDP's `redistribute(Shard(1) -> Replicate)`
is a no-op that emits no collective at all, so a single-rank version of these
three checks passes even against the broken `getattr` read. That four-rank spawn
is also why this module is the slowest in the CPU set.

    python -m torchtitan.experiments.tpu.tests.test_scan_fsdp_lowering
"""

import collections
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard
from torch.distributed.tensor import distribute_tensor
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.experiments.graph_trainer.simple_fsdp import (
    MixedPrecisionPolicy as SimpleFSDPMixedPrecisionPolicy,
    data_parallel as simple_fsdp_data_parallel,
)
from torchtitan.experiments.tpu.scan import fsdp as scan_fsdp
from torchtitan.experiments.tpu.scan import layer_stack as scan_model


def setUpModule():
  if not dist.is_initialized():
    dist.init_process_group(
        backend="gloo", store=dist.HashStore(), rank=0, world_size=1
    )


def tearDownModule():
  if dist.is_initialized():
    dist.destroy_process_group()


class _ToyBlock(nn.Module):
  """Decoder-block stand-in with the real call signature."""

  def __init__(self, dim: int):
    super().__init__()
    self.wi = nn.Linear(dim, dim, bias=False)
    self.wo = nn.Linear(dim, dim, bias=False)

  def forward(self, h, attention_masks=None, positions=None):
    del attention_masks, positions
    return h + self.wo(torch.tanh(self.wi(h)))


class _ToyModel(nn.Module):

  def __init__(self, n_layers: int, dim: int):
    super().__init__()
    self.layers = nn.ModuleDict(
        {str(i): _ToyBlock(dim) for i in range(n_layers)}
    )


def _scanned_stack(n_layers: int = 4, dim: int = 8):
  """A `ScannedLayerStack` over `n_layers` toy blocks, before any sharding."""
  torch.manual_seed(0)
  model = _ToyModel(n_layers, dim)
  scan_model.apply_scan_over_layers(model)
  return model.layers


def _data_parallel(stack, mesh, param_dtype=None, reduce_dtype=None, shard_dim=1):
  """Applies SimpleFSDP the way `parallelize_llama` applies it to the stack.

  `shard_dim` is restated here rather than read from `parallelize_llama`, so
  these tests check the contract (`shard_dim=1` produces a dim-1-sharded stack)
  and not the call site. The call site has its own guard, at runtime rather
  than here: a regression to the default `shard_dim=0` makes
  `plan_stacked_gathers` raise on the first step, which is what
  `PlanValidationTest.test_rejects_sharding_on_the_scan_axis` covers.

  Planning is left to the caller — the tests that assert `plan_stacked_gathers`
  *rejects* a placement need the sharding without the plan.
  """
  simple_fsdp_data_parallel(
      stack,
      mesh,
      mode="fully_shard",
      mp_policy=SimpleFSDPMixedPrecisionPolicy(
          param_dtype=param_dtype, reduce_dtype=reduce_dtype
      ),
      shard_dim=shard_dim,
  )


def _shard(stack, mesh, param_dtype=None, reduce_dtype=None, shard_dim=1):
  """`data_parallel` + `plan_stacked_gathers` + `set_fsdp_plan`, as wired."""
  _data_parallel(stack, mesh, param_dtype, reduce_dtype, shard_dim)
  plan = scan_fsdp.plan_stacked_gathers(
      stack, param_dtype=param_dtype, reduce_dtype=reduce_dtype
  )
  stack.set_fsdp_plan(plan)
  return plan


def _mesh(*shape: int):
  return init_device_mesh("cpu", shape or (1,))


def _subgraphs(graph: torch.fx.GraphModule):
  """`(root_graph, [nested graphs])` — the scan body is one of the nested ones."""
  nested = [
      sub.graph
      for name, sub in graph.named_modules()
      if name and hasattr(sub, "graph")
  ]
  return graph.graph, nested


def _collectives(fx_graph) -> collections.Counter:
  return collections.Counter(
      node.target._schema.name.split("::")[-1]  # pylint: disable=protected-access
      for node in fx_graph.nodes
      if isinstance(node.target, torch._ops.OpOverload)  # pylint: disable=protected-access
      and node.target.namespace == "_c10d_functional"
  )


class StackedPlacementTest(parameterized.TestCase):
  """Dim 0 is the scan axis and must survive SimpleFSDP untouched."""

  def test_stack_is_sharded_on_dim_one(self):
    stack = _scanned_stack()
    _shard(stack, _mesh(1))
    for fqn, param in zip(stack.param_fqns, stack.stacked_parameters()):
      self.assertIsInstance(param, DTensor, f"{fqn} was not sharded at all")
      self.assertEqual(param.placements, (Shard(1),), f"wrong placement for {fqn}")

  def test_layer_axis_is_never_sharded(self):
    """The invariant, stated as the thing that must never be true.

    A `Shard(0)` here makes every scan iteration slice a sharded axis, which
    turns the per-layer read into a dynamic collective. It is also what a
    copy-paste of the default `shard_dim=0` call would produce, so it is worth
    asserting separately from the positive check above.
    """
    stack = _scanned_stack()
    _shard(stack, _mesh(1))
    for fqn, param in zip(stack.param_fqns, stack.stacked_parameters()):
      placement = param.placements[0]
      self.assertFalse(
          isinstance(placement, Shard) and placement.dim == 0,
          f"{fqn} is sharded on the scan axis",
      )
      self.assertEqual(
          param.shape[0],
          stack.n_layers,
          f"{fqn} lost its layer axis: {tuple(param.shape)}",
      )

  def test_forward_does_not_read_through_the_parametrization(self):
    """The 136.9 GB regression test.

    `forward` must read `self._parameters[...]`; a `getattr` instead goes
    through SimpleFSDP's property and gathers all N layers before the loop
    starts. Both spellings produce a numerically correct run, so nothing but a
    counter distinguishes them.
    """
    stack = _scanned_stack()
    _shard(stack, _mesh(1))

    parametrization = stack._simple_fsdp_parametrization_list[0]  # pylint: disable=protected-access
    calls = []
    stack._simple_fsdp_parametrization_list = [  # pylint: disable=protected-access
        lambda p: (calls.append(p.shape), parametrization(p))[1]
    ]

    h = torch.randn(2, 3, 8, requires_grad=True)
    stack(h)

    self.assertEmpty(
        calls,
        "forward gathered the whole stack through SimpleFSDP's parametrization"
        f" property (shapes {calls}); it must read self._parameters and gather"
        " per layer inside the scan body",
    )


class PlanValidationTest(parameterized.TestCase):
  """`plan_stacked_gathers` must reject what the in-body path cannot express."""

  def test_rejects_sharding_on_the_scan_axis(self):
    stack = _scanned_stack()
    _data_parallel(stack, _mesh(1), shard_dim=0)
    with self.assertRaisesRegex(ValueError, "sharded on dim 0"):
      scan_fsdp.plan_stacked_gathers(stack, None, None)

  def test_rejects_a_multi_dimensional_mesh(self):
    """hybrid_shard would need one collective per mesh dim inside the body."""
    stack = _scanned_stack()
    mesh = _mesh(1, 1)
    for mangled, param in list(stack._parameters.items()):  # pylint: disable=protected-access
      stack._parameters[mangled] = nn.Parameter(  # pylint: disable=protected-access
          distribute_tensor(param.detach(), mesh, (Replicate(), Shard(1)))
      )
    with self.assertRaisesRegex(ValueError, "1-D dp mesh only"):
      scan_fsdp.plan_stacked_gathers(stack, None, None)

  def test_sharded_placement_plans_a_gather(self):
    stack = _scanned_stack()
    plan = _shard(
        stack, _mesh(1), param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    self.assertLen(plan, len(stack.param_fqns))
    for spec in plan.values():
      self.assertTrue(spec.sharded)
      self.assertEqual(spec.param_dtype, torch.bfloat16)
      self.assertEqual(spec.reduce_dtype, torch.float32)
      self.assertNotEmpty(spec.group_name)

  def test_replicated_placement_plans_a_cast_only(self):
    """SimpleFSDP downgrades Shard to Replicate when the extent is too small.

    The plan has to mirror that downgrade: gathering a replicated parameter
    would produce a tensor `mesh.size()` times too long, silently, because the
    concatenation axis is a real weight axis.
    """
    stack = _scanned_stack()
    mesh = _mesh(1)
    for mangled, param in list(stack._parameters.items()):  # pylint: disable=protected-access
      stack._parameters[mangled] = nn.Parameter(  # pylint: disable=protected-access
          distribute_tensor(param.detach(), mesh, (Replicate(),))
      )
    plan = scan_fsdp.plan_stacked_gathers(stack, torch.bfloat16, torch.float32)
    for spec in plan.values():
      self.assertFalse(spec.sharded)
      self.assertEqual(spec.param_dtype, torch.bfloat16)

  def test_unsharded_parameter_still_honours_the_compute_dtype(self):
    """A single-device run must agree numerically with a sharded one."""
    stack = _scanned_stack()
    plan = scan_fsdp.plan_stacked_gathers(stack, torch.bfloat16, torch.float32)
    for spec in plan.values():
      self.assertFalse(spec.sharded)
      self.assertEqual(spec.param_dtype, torch.bfloat16)
      self.assertEqual(
          scan_fsdp.gather_slice(torch.randn(4, 4), spec).dtype, torch.bfloat16
      )


# -- multi-rank collective placement -------------------------------------------
# These run in a real gloo world of 4, not the single-rank world the rest of the
# module uses, and the size is load-bearing rather than incidental: on a mesh of
# one, SimpleFSDP's `redistribute(Shard(1) -> Replicate)` is a no-op that emits
# no collective at all. A hoisted whole-stack gather — the exact 136.9 GB
# regression — is therefore *invisible* at world size 1. Verified by mutation:
# rewriting `forward`'s `self._parameters[...]` read as `getattr` leaves every
# single-rank collective assertion green.


def _trace_scanned(world_size: int, n_layers: int = 4, dim: int = 8):
  stack = _scanned_stack(n_layers, dim)
  _shard(stack, init_device_mesh("cpu", (world_size,)))
  h = torch.randn(2, 3, dim, requires_grad=True)
  # "real", not "fake": the stack holds real parameters, and tracing the HOP
  # under a second FakeTensorMode trips "Mixing fake modes NYI".
  return stack, make_fx(stack, tracing_mode="real")(h)


def _check_no_collective_above_the_scan(world_size: int) -> None:
  """An all-gather in the root graph is the 136.9 GB failure, in FX form."""
  _, graph = _trace_scanned(world_size)
  root, _ = _subgraphs(graph)
  found = _collectives(root)
  assert not found, (
      f"{dict(found)} traced outside the scan body; the whole stack is being"
      " gathered before the loop"
  )


def _check_one_gather_per_sharded_parameter(world_size: int) -> None:
  stack, graph = _trace_scanned(world_size)
  _, nested = _subgraphs(graph)
  gathers = sum(_collectives(sub)["all_gather_into_tensor"] for sub in nested)
  assert gathers == len(stack.param_fqns), (
      f"{gathers} in-body all-gathers for {len(stack.param_fqns)} stacked"
      " parameters; expected exactly one each"
  )


def _check_gathers_do_not_scale_with_depth(world_size: int) -> None:
  """Per-layer gathering means one gather in the body, not one per layer.

  If the count tracked depth, the gather would have been hoisted out of the
  body and unrolled — the same regression as
  `_check_no_collective_above_the_scan`, seen from the other side.
  """

  def body_gathers(n_layers):
    _, graph = _trace_scanned(world_size, n_layers=n_layers)
    _, nested = _subgraphs(graph)
    return sum(_collectives(sub)["all_gather_into_tensor"] for sub in nested)

  small, large = body_gathers(4), body_gathers(16)
  assert small == large, (
      f"in-body all-gathers went {small} -> {large} when depth went 4 -> 16;"
      " the gather is being unrolled, not scanned"
  )


_CHECKS = {
    "no_collective_above_the_scan": _check_no_collective_above_the_scan,
    "one_gather_per_sharded_parameter": _check_one_gather_per_sharded_parameter,
    "gathers_do_not_scale_with_depth": _check_gathers_do_not_scale_with_depth,
}


def _rank_main(rank: int, world_size: int, init_method: str, check: str):
  dist.init_process_group(
      backend="gloo", init_method=init_method, rank=rank, world_size=world_size
  )
  try:
    _CHECKS[check](world_size)
  finally:
    dist.destroy_process_group()


class InBodyCollectiveTest(parameterized.TestCase):
  """The all-gathers belong in the loop body, and nowhere else."""

  WORLD_SIZE = 4

  def _spawn(self, check: str):
    with tempfile.TemporaryDirectory() as tmpdir:
      try:
        mp.spawn(
            _rank_main,
            args=(
                self.WORLD_SIZE,
                f"file://{os.path.join(tmpdir, 'store')}",
                check,
            ),
            nprocs=self.WORLD_SIZE,
            join=True,
        )
      except Exception as e:  # pylint: disable=broad-except
        # The child's assertion text is the whole diagnostic; re-raising the
        # spawn wrapper would bury it under a process-exit code.
        self.fail(f"{check} failed in a rank:\n{e}")

  @parameterized.named_parameters(
      (name, name) for name in sorted(_CHECKS)
  )
  def test_in_body(self, check):
    self._spawn(check)


if __name__ == "__main__":
  absltest.main()
