# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.tools.utils import get_local_device


class _FakeDeviceModule:
    def __init__(self, num_devices: int):
        self.num_devices = num_devices

    def device_count(self) -> int:
        return self.num_devices


def _patch_device(
    monkeypatch: pytest.MonkeyPatch, device_type: str, num_devices: int
) -> None:
    monkeypatch.setattr("torchtitan.tools.utils.get_device_type", lambda: device_type)
    monkeypatch.setattr(
        "torchtitan.tools.utils.get_device_module",
        lambda: _FakeDeviceModule(num_devices),
    )


def test_get_local_device_uses_local_rank_when_multiple_devices_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")
    _patch_device(monkeypatch, "cuda", 8)

    assert get_local_device() == torch.device("cuda:3")


def test_get_local_device_uses_zero_when_one_device_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")
    _patch_device(monkeypatch, "cuda", 1)

    assert get_local_device() == torch.device("cuda:0")


def test_get_local_device_rejects_out_of_range_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "8")
    _patch_device(monkeypatch, "cuda", 8)

    with pytest.raises(ValueError, match="outside the visible cuda device count"):
        get_local_device()


def test_get_local_device_requires_local_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    _patch_device(monkeypatch, "cuda", 8)

    with pytest.raises(ValueError, match="LOCAL_RANK must be set"):
        get_local_device()


@pytest.mark.parametrize("device_type", ["cpu", "tpu"])
def test_get_local_device_is_index_free_for_single_device_lanes(
    monkeypatch: pytest.MonkeyPatch, device_type: str
) -> None:
    # A process owns exactly one cpu/tpu device, so LOCAL_RANK is not a device
    # index there. tpu additionally reports the host chip count from
    # device_count() until the PJRT runtime is initialized, so the
    # visible-device heuristic must not run on that lane.
    monkeypatch.setenv("LOCAL_RANK", "3")
    _patch_device(monkeypatch, device_type, 4)

    assert get_local_device() == torch.device(device_type)
