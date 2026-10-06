# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torchtitan.tools.utils import (
    _patch_tpu_device_module,
    _TpuDeviceModule,
    GarbageCollection,
    get_cuda_flash_attention_impl,
    get_local_device,
)


class _FakeDeviceModule:
    def __init__(self, num_devices: int):
        self.num_devices = num_devices

    def device_count(self) -> int:
        return self.num_devices


def test_gc_debug_collects_once_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    gc_collect = Mock()
    monkeypatch.setattr("torchtitan.tools.utils.gc.collect", gc_collect)
    garbage_collection = GarbageCollection.__new__(GarbageCollection)
    garbage_collection.debug = True

    for step in (1, 2):
        assert garbage_collection.run(step)
        gc_collect.assert_called_once_with(2)
        gc_collect.reset_mock()


def test_get_local_device_uses_local_rank_when_multiple_devices_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr("torchtitan.tools.utils.device_type", "cuda")
    monkeypatch.setattr("torchtitan.tools.utils.device_module", _FakeDeviceModule(8))

    assert get_local_device() == torch.device("cuda:3")


def test_get_local_device_uses_zero_when_one_device_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr("torchtitan.tools.utils.device_type", "cuda")
    monkeypatch.setattr("torchtitan.tools.utils.device_module", _FakeDeviceModule(1))

    assert get_local_device() == torch.device("cuda:0")


def test_get_local_device_rejects_out_of_range_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "8")
    monkeypatch.setattr("torchtitan.tools.utils.device_type", "cuda")
    monkeypatch.setattr("torchtitan.tools.utils.device_module", _FakeDeviceModule(8))

    with pytest.raises(ValueError, match="outside the visible cuda device count"):
        get_local_device()


@pytest.mark.parametrize(
    ("capability", "expected_impl"),
    [
        ((8, 0), None),
        ((9, 0), "FA3"),
        ((9, 1), "FA3"),
        ((10, 0), "FA4"),
        ((10, 3), "FA4"),
        # SM 11.0+ falls through to the newest known impl (FA4).
        ((11, 0), "FA4"),
    ],
)
def test_get_cuda_flash_attention_impl(monkeypatch, capability, expected_impl):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.version, "hip", None)

    assert get_cuda_flash_attention_impl() == expected_impl


def test_get_cuda_flash_attention_impl_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert get_cuda_flash_attention_impl() is None


def test_get_cuda_flash_attention_impl_on_rocm(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "7.0")

    assert get_cuda_flash_attention_impl() is None


def _tpu_device_type_registered() -> bool:
    try:
        torch.device("tpu")
    except RuntimeError:
        return False
    return True


def _fake_tpu_module() -> ModuleType:
    module = ModuleType("torch.tpu")
    module.set_device = Mock(name="torch.tpu.set_device")
    module.current_device = Mock(name="torch.tpu.current_device", return_value=0)
    module.get_device_name = Mock(return_value="TPU v6e")
    module.is_triton_capable = Mock(return_value=False)
    return module


_ACCELERATOR_DEVICE_APIS = (
    "is_available",
    "device_count",
    "current_device_index",
    "set_device_index",
    "synchronize",
    "current_stream",
    "set_stream",
)


def _fake_accelerator(**overrides) -> SimpleNamespace:
    apis = {
        name: Mock(name=f"torch.accelerator.{name}")
        for name in _ACCELERATOR_DEVICE_APIS
    }
    apis["current_device_index"].return_value = 0
    apis["memory_stats"] = Mock(return_value={"allocated_bytes.all.current": 1})
    apis.update(overrides)
    return SimpleNamespace(**apis)


def test_tpu_device_module_without_torch_accelerator_uses_torch_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # e.g. a PyTorch build without torch.accelerator: keep torch.tpu as is.
    monkeypatch.delattr(torch, "accelerator", raising=False)
    tpu = _fake_tpu_module()

    device_module = _patch_tpu_device_module(tpu)

    assert device_module is tpu
    device_module.set_device(0)
    tpu.set_device.assert_called_once_with(0)
    assert device_module.memory_stats() == {}


def test_tpu_device_module_routes_internal_apis_to_torch_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _fake_accelerator()
    monkeypatch.setattr(torch, "accelerator", accelerator)
    tpu = _fake_tpu_module()

    device_module = _patch_tpu_device_module(tpu)

    assert isinstance(device_module, _TpuDeviceModule)
    assert device_module.current_device() == 0
    device_module.set_device(0)
    accelerator.set_device_index.assert_called_once_with(0)
    device_module.synchronize()
    accelerator.synchronize.assert_called_once_with(None)
    tpu.set_device.assert_not_called()
    tpu.current_device.assert_not_called()
    # Everything else is forwarded to torch.tpu.
    assert device_module.get_device_name() == "TPU v6e"
    assert device_module.memory_stats() == {"allocated_bytes.all.current": 1}


def test_tpu_device_module_without_set_device_index_uses_torch_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A torch.accelerator namespace without the device APIs (e.g. a stub).
    accelerator = _fake_accelerator()
    del accelerator.set_device_index
    monkeypatch.setattr(torch, "accelerator", accelerator)
    tpu = _fake_tpu_module()

    assert _patch_tpu_device_module(tpu) is tpu


@pytest.mark.skipif(
    not _tpu_device_type_registered(),
    reason="torch has no tpu device type registered",
)
def test_tpu_device_module_set_device_accepts_index_free_tpu_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _fake_accelerator()
    accelerator.current_device_index.return_value = 3
    monkeypatch.setattr(torch, "accelerator", accelerator)

    device_module = _patch_tpu_device_module(_fake_tpu_module())
    device_module.set_device(torch.device("tpu"))
    device_module.set_device("tpu")

    assert [c.args for c in accelerator.set_device_index.call_args_list] == [(3,), (3,)]
