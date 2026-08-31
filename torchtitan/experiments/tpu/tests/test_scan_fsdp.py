# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The FSDP all-gather belongs inside the scan body, and nowhere else.

`test_scan_stack.py` proves the layer loop becomes one `scan`. That is half the
claim. The other half is *where the all-gather happens*: SimpleFSDP installs its
gather as a property on the parameter name, so one read of a stacked parameter
gathers all N layers before the loop starts -- 136.9 GB of bf16 weights at
LLaMA-3 70B against a 96 GB budget. Per-layer, inside the body, it is 1.71 GB.

Four checks, each guarding a failure that is silent (a correct run that is
merely 80x too large, or a per-iteration slice that crosses a sharded axis):

  * placement -- the stack shards on dim 1, never on dim 0, the scan axis;
  * plan validation -- every placement the in-body path cannot express is
    rejected loudly;
  * graph shape -- the all-gathers trace into the scan body and none above it;
  * numerics -- sharded-and-scanned matches the plain unrolled loop, forward
    and per-layer gradient.

Every check runs in a real world of `--num_devices`, and the world size is
load-bearing rather than incidental: on a mesh of one, SimpleFSDP's
`redistribute(Shard(1) -> Replicate)` is a no-op that emits no collective at
all, so a single-rank version of all of this passes even against the broken
whole-stack gather. Verified by mutation: rewriting `forward`'s
`self._parameters[...]` read as `getattr` leaves every single-rank assertion
green.

    NPROC=8 ./run_in_container.sh \
        torchtitan.experiments.tpu.tests.test_scan_fsdp --device=tpu
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard

from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu.scan import fsdp as scan_fsdp
from torchtitan.experiments.tpu.scan import layer_stack
from torchtitan.experiments.tpu.tests import scan_fixtures
from torchtitan.experiments.tpu.tests.scan_fixtures import BATCH, DIM, N_LAYERS, SEQ


def _traced(device, world_size: int, n_layers: int = N_LAYERS):
  """`(stack, root graph, [body graphs])` for a sharded, scanned stack."""
  _, stack = scan_fixtures.stacked(device, n_layers=n_layers)
  scan_fixtures.shard(stack, device, world_size)
  h = torch.randn(BATCH, SEQ, DIM, device=device, requires_grad=True)
  root, bodies = scan_fixtures.trace(stack, h)
  return stack, root, bodies


# -- workers -------------------------------------------------------------------
# Each runs in every rank of a real world; `run_distributed` spawns them. They
# assert rather than use unittest, because a rank is not a test case; the
# assertion text is what surfaces when a rank fails.


def _worker_placement(device, rank, world_size):
  """Dim 0 is the scan axis and must survive SimpleFSDP untouched."""
  del rank
  _, stack = scan_fixtures.stacked(device)
  plan = scan_fixtures.shard(
      stack, device, world_size, torch.bfloat16, torch.float32
  )

  assert len(plan) == len(stack.param_fqns), (
      f"planned {len(plan)} gathers for {len(stack.param_fqns)} stacked"
      " parameters"
  )
  for fqn, param in zip(stack.param_fqns, stack.stacked_parameters()):
    assert isinstance(param, DTensor), f"{fqn} was not sharded at all"
    assert param.placements == (Shard(1),), (
        f"{fqn} has placements {param.placements}; sharding dim 0 would make"
        " every scan iteration slice a sharded axis"
    )
    assert param.shape[0] == stack.n_layers, (
        f"{fqn} lost its layer axis: {tuple(param.shape)}"
    )
    spec = plan[layer_stack._mangle(fqn)]  # pylint: disable=protected-access
    assert spec.sharded and spec.group_name, f"{fqn} planned no gather: {spec}"
    assert spec.param_dtype == torch.bfloat16, spec
    assert spec.reduce_dtype == torch.float32, spec


def _worker_plan_rejections(device, rank, world_size):
  """`plan_gathers` must reject what the in-body path cannot express."""
  del rank
  policy = MixedPrecisionPolicy()

  # Depth >= world size, or `data_parallel` downgrades the dim-0 Shard it is
  # being asked for to a Replicate and there is nothing left to reject.
  _, stack = scan_fixtures.stacked(device, n_layers=max(world_size, N_LAYERS))
  data_parallel(
      stack,
      init_device_mesh(device.type, (world_size,)),
      mode="fully_shard",
      mp_policy=policy,
      shard_dim=0,
  )
  try:
    scan_fsdp.plan_gathers(stack, policy)
  except ValueError as e:
    assert "sharded on dim 0" in str(e), e
  else:
    raise AssertionError("sharding the scan axis was accepted")

  # hybrid_shard would need one collective per mesh axis inside the body.
  _, stack = scan_fixtures.stacked(device)
  mesh = init_device_mesh(device.type, (2, world_size // 2))
  for name, param in list(stack._parameters.items()):  # pylint: disable=protected-access
    stack._parameters[name] = nn.Parameter(  # pylint: disable=protected-access
        distribute_tensor(param.detach(), mesh, (Replicate(), Shard(1)))
    )
  try:
    scan_fsdp.plan_gathers(stack, policy)
  except ValueError as e:
    assert "1-D dp mesh only" in str(e), e
  else:
    raise AssertionError("a 2-D mesh was accepted")


def _worker_graph_shape(device, rank, world_size):
  """The gathers trace into the body, one per parameter, none above."""
  del rank
  stack, root, bodies = _traced(device, world_size)

  above = scan_fixtures.collectives(root)
  assert not above, (
      f"{dict(above)} traced outside the scan body; the whole stack is being"
      " gathered before the loop starts"
  )
  count = lambda gs: sum(
      scan_fixtures.collectives(g)["all_gather_into_tensor"] for g in gs
  )
  in_body = count(bodies)
  assert in_body == len(stack.param_fqns), (
      f"{in_body} in-body all-gathers for {len(stack.param_fqns)} stacked"
      " parameters; expected exactly one each"
  )

  # If the count tracked depth, the gather would have been hoisted out of the
  # body and unrolled -- the same regression seen from the other side.
  deeper = count(_traced(device, world_size, n_layers=4 * N_LAYERS)[2])
  assert in_body == deeper, (
      f"in-body all-gathers went {in_body} -> {deeper} when depth went"
      f" {N_LAYERS} -> {4 * N_LAYERS}; the gather is being unrolled"
  )


def _worker_matches_unrolled(device, rank, world_size):
  """Sharded and scanned must equal the plain loop over full weights.

  The forward is the easy half. The gradient is the one that catches a bad
  in-body collective: `gather_slice` differentiates to a reduce-scatter, so a
  wrong group, axis, or dtype shows up as a per-layer weight gradient that is
  off by a factor of the world size rather than as an error.
  """
  # Same seed as `scan_fixtures.stacked`, so the reference holds exactly the
  # weights the stack was built from -- no copy of a half-transformed module.
  torch.manual_seed(0)
  reference = scan_fixtures.ToyModel(N_LAYERS, DIM).to(device)
  _, stack = scan_fixtures.stacked(device)
  scan_fixtures.shard(stack, device, world_size)

  # Real data parallelism: one global batch, a different slice per rank. The
  # reduce-scatter in `gather_slice`'s backward *sums* across ranks -- DTensor's
  # Partial -> Shard reduction -- so the reference must be the whole global
  # batch. Feeding every rank the same slice instead would leave the gradient
  # world_size times too large and the comparison would have to fudge it.
  torch.manual_seed(1)
  h = torch.randn(
      world_size * BATCH, SEQ, DIM, device=device, requires_grad=True
  )
  mine = slice(rank * BATCH, (rank + 1) * BATCH)

  got = stack(h[mine])
  got.sum().backward()
  want = scan_fixtures.unrolled_reference(reference, h)
  want.sum().backward()

  torch.testing.assert_close(
      got.detach().cpu(),
      want[mine].detach().cpu(),
      rtol=1e-4,
      atol=1e-4,
      msg=lambda default: f"rank {rank}'s slice of the forward differs\n{default}",
  )
  for fqn, param in zip(stack.param_fqns, stack.stacked_parameters()):
    assert isinstance(param.grad, DTensor), (
        f"the gradient of {fqn} came back as a plain tensor: the shard was"
        " detached from its DTensor on the way into the body"
    )
    full = param.grad.full_tensor()
    for i in range(stack.n_layers):
      torch.testing.assert_close(
          full[i].cpu(),
          reference.layers[str(i)].get_parameter(fqn).grad.cpu(),
          rtol=1e-4,
          atol=1e-4,
          msg=lambda default, fqn=fqn, i=i: (
              f"layer {i} gradient for {fqn} diverges from the unrolled"
              f" loop\n{default}"
          ),
      )


_WORKERS = {
    "placement": _worker_placement,
    "plan_rejections": _worker_plan_rejections,
    "graph_shape": _worker_graph_shape,
    "matches_unrolled": _worker_matches_unrolled,
}


class ScanFSDPTest(base_distributed_device_test.BaseDistributedDeviceTest):
  """Runs each worker across the whole world."""

  def setUp(self):
    super().setUp()
    if self.num_devices < 2:
      self.skipTest(
          "the in-body gather is invisible at world size 1: SimpleFSDP's"
          " Shard(1) -> Replicate redistribute emits no collective there"
      )

  @parameterized.named_parameters((name, name) for name in sorted(_WORKERS))
  def test_in_body_gather(self, worker):
    distributed.run_distributed(
        self.num_devices, self.accelerator_device_type, _WORKERS[worker]
    )


if __name__ == "__main__":
  absltest.main()
