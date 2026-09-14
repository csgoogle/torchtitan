"""Stacking must not change what the model computes, on the real device.

The structural claims -- FQNs, views, placements, checkpoints -- hold wherever
the tensors live, so ``test_layer_stack`` asserts them on CPU. Numerics do not:
the stacked layers run as one batched matmul per op instead of ``n_layers``
separate ones, which is a different kernel and a different accumulation order.
So the parity check is repeated here on ``--device``, across ``--num_devices``
ranks, with the stacks actually sharded.
"""

from absl.testing import absltest
import torch
from torch.distributed.tensor import Shard

from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu.graph_trainer.layer_stack import STACK_ATTR
from torchtitan.experiments.tpu.graph_trainer.tests.test_layer_stack import (
    _build,
    SEQ_LEN,
)


def _models(device, world_size):
    """A plain model and a stacked one, built identically on ``device``."""
    # SimpleFSDP reads this to place its mesh, and a stack is a DTensor, so it
    # has to name the device the weights are actually on.
    import torchtitan.distributed.parallel_dims as parallel_dims  # pylint: disable=g-import-not-at-top

    parallel_dims.device_type = device.type
    plain = _build(stacked=False, device=device, world_size=world_size)
    stacked = _build(stacked=True, device=device, world_size=world_size)
    return plain, stacked


def forward_matches_unrolled(device, rank, world_size):
    """One batched matmul per op must give the same bits as n separate ones."""
    del rank
    plain, stacked = _models(device, world_size)
    tokens = torch.randint(0, plain.config.vocab_size, (2, SEQ_LEN), device=device)
    torch.testing.assert_close(stacked(tokens), plain(tokens), rtol=0, atol=0)


def stacks_are_sharded_off_the_layer_axis(device, rank, world_size):
    """Sharding dim 0 would split one layer's weight across ranks.

    Worth repeating on a real mesh: with ``n_layers`` below the rank count,
    which is the usual case, a dim-0 shard silently falls back to ``Replicate``.
    """
    del rank
    _, stacked = _models(device, world_size)
    holder = getattr(stacked, STACK_ATTR)
    for name, stack in holder.named_parameters(recurse=False):
        assert stack.placements == (Shard(1),), f"{name} is {stack.placements}"


class TestLayerStackDistributed(
    base_distributed_device_test.BaseDistributedDeviceTest
):
    """Each check runs inside the spawned workers, so it asserts there too.

    The launcher returns nothing, so a failure arrives as the child's re-raised
    exception rather than as a diff printed here.
    """

    def _on_devices(self, worker):
        distributed.run_distributed(
            self.num_devices, self.accelerator_device_type, worker
        )

    def test_forward_matches_unrolled(self):
        self._on_devices(forward_matches_unrolled)

    def test_stacks_are_sharded_off_the_layer_axis(self):
        self._on_devices(stacks_are_sharded_off_the_layer_axis)


if __name__ == "__main__":
    absltest.main()
