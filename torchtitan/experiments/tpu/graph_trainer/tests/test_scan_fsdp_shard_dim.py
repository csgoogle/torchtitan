"""One shard axis per stack, not one per holder.

Sharding every stack on dim 1 breaks once the mesh is wider than dim 1 of some
stack: ``data_parallel`` quietly replicates that parameter, and the run OOMs
with nothing pointing at the cause. Qwen3-235B hits this at 129 ranks, where
the expert stacks' dim 1 is ``num_experts = 128``.

A fake process group is enough: the failure is about mesh size, not devices.
"""

import contextlib
import unittest
import unittest.mock

import torch
import torch.distributed as dist
from absl.testing import absltest
from torch import nn
from torch.distributed.tensor import (
    distribute_tensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.experiments.tpu.graph_trainer.layer_stack import (
    _shard_dim,
    LayerStackHolder,
)
from torchtitan.experiments.tpu.graph_trainer import scan_fsdp
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import plan_stacked_gathers


N_LAYERS = 94

# Qwen3-235B-A22B's stacked parameters, layer axis in front.
Q235_STACKS = {
    "moe__experts__w1_EDF": (N_LAYERS, 128, 4096, 1536),
    "moe__experts__w2_EFD": (N_LAYERS, 128, 1536, 4096),
    "moe__experts__w3_EDF": (N_LAYERS, 128, 4096, 1536),
    "moe__router__gate__weight": (N_LAYERS, 128, 4096),
    "attention__qkv_linear__wqkv": (N_LAYERS, 9216, 4096),
    "attention_norm__weight": (N_LAYERS, 4096),
    "attention__q_norm__weight": (N_LAYERS, 128),
}


def _pick(shape, degree):
    return _shard_dim(torch.empty(shape, device="meta"), degree)


@contextlib.contextmanager
def _sharded_holder(world, shapes=Q235_STACKS, shard_dim=None):
    """The stacks, wrapped by SimpleFSDP on a `world`-wide fake mesh."""
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=world)
    try:
        holder = LayerStackHolder()
        for name, shape in shapes.items():
            holder.register_parameter(
                name, nn.Parameter(torch.empty(shape, device="meta"))
            )
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("fsdp",))
        if shard_dim is None:
            shard_dim = holder.plan_shard_dims(world)
        data_parallel(holder, mesh, "fully_shard", shard_dim=shard_dim)
        yield holder
    finally:
        dist.destroy_process_group()


class TestShardDimPicker(unittest.TestCase):

    def test_dim_one_is_kept_whenever_it_divides(self):
        # Every published 235B number was taken at 128 ranks on dim 1. Moving a
        # stack off dim 1 there would make the comparison not one.
        for name, shape in Q235_STACKS.items():
            if len(shape) > 2 and shape[1] % 128 == 0:
                self.assertEqual(_pick(shape, 128), 1, name)

    def test_a_per_layer_vector_is_replicated(self):
        # A norm weight sharded 128 ways is 32 elements a rank, and the scan
        # body then gathers them back with a latency-bound collective on
        # every layer. Replicating it costs nothing worth counting.
        for name, shape in Q235_STACKS.items():
            if len(shape) == 2:
                self.assertIsNone(_pick(shape, 128), name)

    def test_every_expert_stack_takes_the_same_axis_at_512_ranks(self):
        # Both hidden axes of every expert stack divide 512, and the unrolled
        # path puts all three on the same one (its dim 1, this dim 2). Taking
        # the widest instead would send w2_EFD to dim 3 by itself.
        for name in Q235_STACKS:
            if name.startswith("moe__experts__"):
                self.assertEqual(_pick(Q235_STACKS[name], 512), 2, name)

    def test_an_expert_stack_moves_off_num_experts_past_128_ranks(self):
        # Dim 1 is the 128 experts. By 2048 only w2_EFD's minor axis still
        # divides, so it is the one stack that has to leave dim 2.
        self.assertEqual(_pick(Q235_STACKS["moe__experts__w1_EDF"], 2048), 2)
        self.assertEqual(_pick(Q235_STACKS["moe__experts__w2_EFD"], 2048), 3)

    def test_a_gather_off_dim_zero_is_transposed_onto_dim_zero(self):
        # `funcol`'s fast path needs every axis before the gather axis to be
        # size 1, and a layer slice starts with `num_experts`. Off dim 0 it
        # would fall back to `cat(chunk(t, world_size))`; the transpose has to
        # both avoid that and reassemble on the axis the caller asked for.
        shard = torch.arange(3 * 4 * 6).reshape(3, 4, 6).float()
        ranks = 4

        def fake_all_gather(x, gather_dim, group):
            self.assertEqual(gather_dim, 0, "gathered off dim 0")
            return torch.cat([x] * ranks, dim=0)

        with unittest.mock.patch.object(
            scan_fsdp.funcol, "all_gather_tensor", fake_all_gather
        ):
            out = scan_fsdp._gather_on(shard, 2, "group")
        self.assertTrue(torch.equal(out, torch.cat([shard] * ranks, dim=2)))

    def test_the_layer_axis_is_never_chosen(self):
        for name, shape in Q235_STACKS.items():
            for degree in (128, 1024, 2048, 4096):
                self.assertNotEqual(_pick(shape, degree), 0, f"{name} @ {degree}")
        # And a stack with no other axis is refused, not sharded on dim 0.
        with self.assertRaises(ValueError):
            _pick((N_LAYERS,), 128)


class TestQ235Placements(unittest.TestCase):
    """What ``data_parallel`` and the scan body's plan end up with."""

    def test_a_dict_places_exactly_like_the_int_it_repeats(self):
        # Every other `data_parallel` caller passes an int, so the dict has to
        # be a superset of that path and not a second one beside it.
        def placements(shard_dim):
            with _sharded_holder(128, shard_dim=shard_dim) as holder:
                return {n: p.placements for n, p in holder._parameters.items()}

        self.assertEqual(
            placements(1), placements({name: 1 for name in Q235_STACKS})
        )

    def test_a_none_in_the_dict_replicates_that_parameter(self):
        with _sharded_holder(128) as holder:
            placements = {n: p.placements for n, p in holder._parameters.items()}
        for name, shape in Q235_STACKS.items():
            expected = (Replicate(),) if len(shape) == 2 else (Shard(1),)
            self.assertEqual(placements[name], expected, name)

    def test_an_already_distributed_parameter_is_refused(self):
        # A DTensor here means EP or TP ran first, and `ReplicateComputation`
        # would re-wrap it with the module-level axis, losing the per-stack one.
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=8)
        try:
            mesh = init_device_mesh("cpu", (8,), mesh_dim_names=("fsdp",))
            holder = LayerStackHolder()
            holder.register_parameter(
                "w",
                nn.Parameter(
                    distribute_tensor(
                        torch.empty(4, 16, device="meta"), mesh, (Shard(1),)
                    )
                ),
            )
            with self.assertRaises(NotImplementedError):
                data_parallel(holder, mesh, "fully_shard", shard_dim={"w": 1})
        finally:
            dist.destroy_process_group()

    def test_hsdp_plans_off_the_shard_axis_not_the_whole_mesh(self):
        # `dp_replicate=16, dp_shard=128` is how 235B reaches 8k chips. Only the
        # shard axis splits the tensor, so only its width may pick the dim.
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2048)
        try:
            mesh = init_device_mesh(
                "cpu", (16, 128), mesh_dim_names=("dp_replicate", "fsdp")
            )
            holder = LayerStackHolder()
            for name, shape in Q235_STACKS.items():
                holder.register_parameter(
                    name, nn.Parameter(torch.empty(shape, device="meta"))
                )
            plan = holder.plan_shard_dims(mesh.size(mesh.ndim - 1))
            data_parallel(holder, mesh, "hybrid_shard", shard_dim=plan)
            for name, spec in plan_stacked_gathers(holder).items():
                if len(Q235_STACKS[name]) == 2:
                    self.assertFalse(spec.sharded, name)  # replicated vector
                    continue
                self.assertTrue(spec.sharded, name)
                self.assertEqual(spec.gather_dim, 0, name)  # dim 1 still divides
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    absltest.main()
