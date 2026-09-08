# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from tests.unit_tests.test_tpu_packed_qkv import _build_fused_and_packed
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel


def _full_tensor(tensor):
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _worker(rank: int, init_file: str, checkpoint_dir: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        mesh = DeviceMesh("cpu", [0, 1])
        fused, packed = (
            data_parallel(module, mesh, mode="fully_shard")
            for module in _build_fused_and_packed()
        )
        for module in (fused, packed):
            torch.manual_seed(1234)
            module.init_states()

        fused_state = fused.state_dict()
        packed_state = packed.state_dict()
        assert fused_state.keys() == packed_state.keys()
        for key in fused_state:
            torch.testing.assert_close(
                _full_tensor(packed_state[key]),
                _full_tensor(fused_state[key]),
                rtol=0,
                atol=0,
            )
        [(name, parameter)] = packed.named_parameters()
        assert name == "wqkv_packed.weight"
        assert isinstance(parameter, DTensor)

        dcp.save({"model": packed_state}, checkpoint_id=checkpoint_dir)
        dist.barrier()
        restored, _ = _build_fused_and_packed()
        restored = data_parallel(restored, mesh, mode="fully_shard")
        load_state = {"model": restored.state_dict()}
        dcp.load(load_state, checkpoint_id=checkpoint_dir)
        restored.load_state_dict(load_state["model"])
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(
                _full_tensor(value),
                _full_tensor(packed_state[key]),
                rtol=0,
                atol=0,
            )
    finally:
        dist.destroy_process_group()


def test_two_rank_simple_fsdp_checkpoint_roundtrip():
    fd, init_file = tempfile.mkstemp()
    os.close(fd)
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        try:
            mp.spawn(
                _worker,
                args=(init_file, checkpoint_dir),
                nprocs=2,
                join=True,
            )
        finally:
            if os.path.exists(init_file):
                os.unlink(init_file)
