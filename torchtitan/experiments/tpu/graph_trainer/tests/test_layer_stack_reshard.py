# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A checkpoint written by a stacked model must reshard like any other.

``test_layer_stack.py`` pins the saved *format* -- same keys, same bytes as the
unrolled model -- but does it at world size 1, where nothing is actually
sharded. The claim this file tests is the one an owner of a 70B checkpoint
cares about: save on eight ranks, come back on four, get the same weights, in
either layout.

That needs two real process groups of different widths, so each case spawns its
own. These tests are therefore slow (seconds, not milliseconds) and live apart
from the single-process ones.
"""

import contextlib
import os
import socket
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.models.llama3 import model_registry


SEED = 1234


def _build(world_size: int, *, stacked: bool) -> torch.nn.Module:
    """The trainer's build order, at an arbitrary data-parallel width."""
    # ParallelDims reads the accelerator type at import; there is none here.
    import torchtitan.distributed.parallel_dims as parallel_dims_module

    parallel_dims_module.device_type = "cpu"

    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=world_size,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    with torch.device("meta"):
        model = model_registry("debugmodel", attn_backend="sdpa").model.build()
    model = apply_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=TrainingConfig(
            mixed_precision_param="float32", mixed_precision_reduce="float32"
        ),
        stack_layers=stacked,
    )
    model.to_empty(device="cpu")
    torch.manual_seed(SEED)
    model.init_weights()
    return model


def _gathered(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The state dict as whole tensors. Collective: every rank must call it."""
    return {k: v.full_tensor() for k, v in model.state_dict().items()}


@contextlib.contextmanager
def _process_group(rank: int, world_size: int, port: int):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        world_size=world_size,
        rank=rank,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def _save_worker(rank: int, world_size: int, port: int, path: str, stacked: bool):
    with _process_group(rank, world_size, port):
        model = _build(world_size, stacked=stacked)
        dcp.save(model.state_dict(), checkpoint_id=os.path.join(path, "ckpt"))
        # Written from the source of truth so the load side has something to be
        # wrong against; a load that quietly no-ops would otherwise pass.
        whole = _gathered(model)
        if rank == 0:
            torch.save(whole, os.path.join(path, "expected.pt"))


def _load_worker(rank: int, world_size: int, port: int, path: str, stacked: bool):
    with _process_group(rank, world_size, port):
        model = _build(world_size, stacked=stacked)
        # Perturb first: `init_weights` is seeded, so without this the model
        # already holds the saved values and any load at all would pass.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        state_dict = model.state_dict()
        dcp.load(state_dict, checkpoint_id=os.path.join(path, "ckpt"))
        model.load_state_dict(state_dict)

        got = _gathered(model)
        if rank != 0:
            return
        want = torch.load(os.path.join(path, "expected.pt"), weights_only=True)
        assert set(got) == set(want), (set(got) ^ set(want))
        for key, expected in want.items():
            torch.testing.assert_close(got[key], expected, rtol=0, atol=0, msg=key)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


class ReshardTest(unittest.TestCase):
    """Eight ranks out, four ranks back, across both layouts."""

    def _round_trip(self, *, saved_stacked: bool, loaded_stacked: bool):
        with tempfile.TemporaryDirectory() as path:
            mp.spawn(
                _save_worker, args=(8, _free_port(), path, saved_stacked), nprocs=8
            )
            mp.spawn(
                _load_worker, args=(4, _free_port(), path, loaded_stacked), nprocs=4
            )

    def test_stacked_to_stacked(self):
        self._round_trip(saved_stacked=True, loaded_stacked=True)

    def test_stacked_to_unrolled(self):
        """Nothing traps a run that started with scan into keeping it."""
        self._round_trip(saved_stacked=True, loaded_stacked=False)

    def test_unrolled_to_stacked(self):
        """The migration this work exists to make safe: an old checkpoint in."""
        self._round_trip(saved_stacked=False, loaded_stacked=True)


if __name__ == "__main__":
    unittest.main()
