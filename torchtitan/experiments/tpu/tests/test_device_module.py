"""Unit tests for the TPU device module returned by torchtitan.tools.utils."""

from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from absl.testing import absltest
import torch
from torchtitan.tools import utils


_ACCELERATOR_DEVICE_APIS = (
    "is_available",
    "device_count",
    "current_device_index",
    "set_device_index",
    "synchronize",
    "current_stream",
    "set_stream",
)


def _tpu_device_type_registered() -> bool:
  try:
    torch.device("tpu")
  except RuntimeError:
    return False
  return True


def _fake_tpu_module() -> ModuleType:
  module = ModuleType("torch.tpu")
  module.set_device = mock.Mock(name="torch.tpu.set_device")
  module.current_device = mock.Mock(
      name="torch.tpu.current_device", return_value=0
  )
  module.get_device_name = mock.Mock(return_value="TPU v6e")
  module.is_triton_capable = mock.Mock(return_value=False)
  return module


def _fake_accelerator(**overrides) -> SimpleNamespace:
  apis = {
      name: mock.Mock(name=f"torch.accelerator.{name}")
      for name in _ACCELERATOR_DEVICE_APIS
  }
  apis["current_device_index"].return_value = 0
  apis["memory_stats"] = mock.Mock(
      return_value={"allocated_bytes.all.current": 1}
  )
  apis.update(overrides)
  return SimpleNamespace(**apis)


class TpuDeviceModuleTest(absltest.TestCase):
  """Tests for _patch_tpu_device_module and _TpuDeviceModule."""

  def _set_accelerator(self, accelerator) -> None:
    """Replaces torch.accelerator for one test; None removes it."""
    missing = object()
    saved = getattr(torch, "accelerator", missing)

    def restore():
      if saved is missing:
        if hasattr(torch, "accelerator"):
          delattr(torch, "accelerator")
      else:
        torch.accelerator = saved

    self.addCleanup(restore)
    if accelerator is None:
      if hasattr(torch, "accelerator"):
        delattr(torch, "accelerator")
    else:
      torch.accelerator = accelerator

  def test_without_torch_accelerator_uses_torch_tpu(self):
    """A PyTorch build without torch.accelerator keeps torch.tpu as is."""
    self._set_accelerator(None)
    tpu = _fake_tpu_module()

    device_module = utils._patch_tpu_device_module(tpu)

    self.assertIs(device_module, tpu)
    device_module.set_device(0)
    tpu.set_device.assert_called_once_with(0)
    self.assertEqual(device_module.memory_stats(), {})

  def test_routes_internal_apis_to_torch_accelerator(self):
    """Internal torch.tpu device APIs go to torch.accelerator."""
    accelerator = _fake_accelerator()
    self._set_accelerator(accelerator)
    tpu = _fake_tpu_module()

    device_module = utils._patch_tpu_device_module(tpu)

    self.assertIsInstance(device_module, utils._TpuDeviceModule)
    self.assertEqual(device_module.current_device(), 0)
    device_module.set_device(0)
    accelerator.set_device_index.assert_called_once_with(0)
    device_module.synchronize()
    accelerator.synchronize.assert_called_once_with(None)
    tpu.set_device.assert_not_called()
    tpu.current_device.assert_not_called()
    # Everything else is forwarded to torch.tpu.
    self.assertEqual(device_module.get_device_name(), "TPU v6e")
    self.assertEqual(
        device_module.memory_stats(), {"allocated_bytes.all.current": 1}
    )

  def test_without_set_device_index_uses_torch_tpu(self):
    """A torch.accelerator without the device APIs (e.g. a stub) is unused."""
    accelerator = _fake_accelerator()
    del accelerator.set_device_index
    self._set_accelerator(accelerator)
    tpu = _fake_tpu_module()

    self.assertIs(utils._patch_tpu_device_module(tpu), tpu)

  @unittest.skipUnless(
      _tpu_device_type_registered(), "torch has no tpu device type registered"
  )
  def test_set_device_accepts_index_free_tpu_device(self):
    """An index-free "tpu" device resolves to the current device index."""
    accelerator = _fake_accelerator()
    accelerator.current_device_index.return_value = 3
    self._set_accelerator(accelerator)

    device_module = utils._patch_tpu_device_module(_fake_tpu_module())
    device_module.set_device(torch.device("tpu"))
    device_module.set_device("tpu")

    self.assertEqual(
        [c.args for c in accelerator.set_device_index.call_args_list],
        [(3,), (3,)],
    )


if __name__ == "__main__":
  absltest.main()
