# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import os
import psutil
import torch
from torch._utils import _get_available_device_type
from torch._utils import _get_device_module
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint import checkpoint_wrapper
from torchtitan.tools.logging import logger
from torchtitan.experiments.tpu import local_tpu_device

device_type = None
device_module = None
tpu_device_module = None


def get_device_type() -> str:
  """Determines and returns the available device type.

  Detection is side-effect free: torch._utils resolves TPU through
  torch.tpu.is_available(), which only scans PCI devices -- it does not
  initialize the PJRT runtime. Bootstrapping the TPU distributed environment
  is the caller's job (tpu_distributed_utils.maybe_init_distributed(), called
  from the TPU entry points before any device is used).

  The result is cached in the global `device_type` variable.

  Returns:
      The device type as a string ("cuda", "tpu", or "cpu").
  """
  global device_type
  if device_type is not None:
    return device_type

  # _get_available_device_type() ends at the privateuse1 backend, which
  # torch_tpu registers as "tpu" via PyTorch's backend autoload.
  device_type = _get_available_device_type() or "cpu"
  return device_type


def _get_cpu_module() -> torch.device:
  """Returns the torch.cpu module with additional functions."""
  global device_module
  # This is a hack to patch the torch.cpu module with additional functions
  # that are not available in the torch.cpu module. This is needed to avoid
  # crashing when using torch.cpu as a device type.
  if device_module is not None:
    return device_module

  logger.warning("No CUDA available, falling back to CPU for device info")
  logger.warning("Patching torch.cpu module to support missing functions")
  device_module = _get_device_module("cpu")

  # torch.cpu doesn't have get_device_name
  def get_device_name(device=None):
    return "cpu"

  setattr(device_module, "get_device_name", get_device_name)

  def current_device():
    # https://docs.pytorch.org/docs/2.8/generated/torch.cuda.current_device.html
    # https://docs.pytorch.org/docs/stable/generated/torch.cpu.current_device.html
    # Need to return index
    return dist.get_rank()

  setattr(device_module, "current_device", current_device)

  def get_device_properties(device=None):
    # properties = torch.cuda._CudaDeviceProperties()
    properties = SimpleNamespace(
        total_memory=psutil.virtual_memory().total
    )
    return properties
  setattr(device_module, "get_device_properties", get_device_properties)

  def reset_peak_memory_stats(device=None):
    pass
  setattr(device_module, "reset_peak_memory_stats", reset_peak_memory_stats)

  def empty_cache():
    pass
  setattr(device_module, "empty_cache", empty_cache)

  def memory_stats(device=None):
    return {}

  setattr(device_module, "memory_stats", memory_stats)
  return device_module


def _get_tpu_module() -> torch.device:
  """Returns the torch.tpu module with additional functions."""
  global tpu_device_module
  if tpu_device_module is not None:
    return tpu_device_module

  logger.warning("Using TPU device info")
  logger.warning("Patching torch.tpu module to support missing functions")
  _device_module = _get_device_module("tpu")
  _cached_device_name = None

  def get_device_name(device=None):
    nonlocal _cached_device_name
    if _cached_device_name is not None:
      return _cached_device_name
    _cached_device_name = "tpu"
    try:
      chip_type, _ = local_tpu_device.get_local_chips()
      if chip_type:
        _cached_device_name = f"tpu-{chip_type.value.name}"
    except RuntimeError as e:
      logger.warning(f"Error detecting local TPU chips: {e}")
    return _cached_device_name

  setattr(_device_module, "get_device_name", get_device_name)

  def set_device(device=None):
    # set_device is no-op for TPU
    pass
  setattr(_device_module, "set_device", set_device)


  def get_device_properties(device=None):
    # properties = torch.cuda._CudaDeviceProperties()
    properties = SimpleNamespace(
        # Figure out how to set this for TPU
        total_memory=16 * 1024 * 1024 * 1024  # 16GB for v5e
    )
    return properties
  setattr(_device_module, "get_device_properties", get_device_properties)

  def reset_peak_memory_stats(device=None):
    pass
  setattr(_device_module, "reset_peak_memory_stats", reset_peak_memory_stats)

  def empty_cache():
    pass
  setattr(_device_module, "empty_cache", empty_cache)

  def memory_stats(device=None):
    # torch_tpu exposes memory stats through the modern torch.accelerator namespace,
    # but torchtitan expects it on the legacy device module namespace (torch.tpu).
    # We mock it here by routing it to the native PyTorch accelerator API.
    return torch.accelerator.memory_stats()
  setattr(_device_module, "memory_stats", memory_stats)

  def is_triton_capable(*args, **kwargs):
    return False
  setattr(_device_module, "is_triton_capable", is_triton_capable)
  try:
    setattr(type(_device_module), "is_triton_capable", is_triton_capable)
  except TypeError:
    pass

  tpu_device_module = _device_module
  return _device_module

def _patch_tpu_device_module():
  def is_triton_capable(*args, **kwargs):
    return False

  try:
    from torch_tpu._internal.device._device_module import TpuDeviceModule
    setattr(TpuDeviceModule, "is_triton_capable", is_triton_capable)
  except Exception:
    pass

  try:
    from torch._dynamo.device_interface import get_interface_for_device
    tpu_interface = get_interface_for_device("tpu")
    setattr(tpu_interface, "is_triton_capable", is_triton_capable)
  except Exception:
    pass

_patch_tpu_device_module()


def get_device_module() -> torch.device:
  """Determines and returns the device module based on the available device type.
  """
  _device_type = get_device_type()
  if _device_type == "cuda":
    return _get_device_module(_device_type)  # returns torch.cuda
  elif _device_type == "tpu":
    return _get_tpu_module()
  elif _device_type == "cpu":
    return _get_cpu_module()
  else:
    return _get_device_module(_device_type)


def get_device() -> torch.device:
  """Determines and returns the device based on the available device type."""
  _device_type = get_device_type()
  if _device_type == "cuda" and torch.cuda.is_available():
    return torch.device(f"{_device_type}:{int(os.getenv('LOCAL_RANK', 0))}")
  elif _device_type == "tpu":
    import torchtitan.experiments.tpu.distributed_utils as tpu_dist_utils  # pylint: disable=g-import-not-at-top
    tpu_dist_utils.maybe_init_distributed()
    return torch.device("tpu")
  else:
    return torch.device(_device_type)


def ptd_checkpoint_wrapper_with_early_stop(
    module: torch.nn.Module,
    checkpoint_impl: checkpoint_wrapper.CheckpointImpl = checkpoint_wrapper.CheckpointImpl.NO_REENTRANT,
    checkpoint_fn=None,
    early_stop: bool = False,
    **checkpoint_fn_kwargs,
) -> torch.nn.Module:
    """Wraps a module with PyTorch Distributed's checkpoint_wrapper.

    This function is a wrapper around
    `torch.distributed.algorithms._checkpoint.checkpoint_wrapper`
    with an added `early_stop` parameter, although early stopping is not
    currently supported in g3.

    Args:
        module: The module to wrap.
        checkpoint_impl: The checkpoint implementation to use.
        checkpoint_fn: Optional function to use for checkpointing.
        early_stop: Whether to enable early stopping.
            Currently raises NotImplementedError if True.
        **checkpoint_fn_kwargs: Additional keyword arguments to pass to
            `checkpoint_fn`.

    Returns:
        The wrapped module.

    Raises:
        NotImplementedError: If `early_stop` is set to True.
    """
    if early_stop:
      raise NotImplementedError(
          "early_stop is not supported by ptd_checkpoint_wrapper."
      )

    checkpoint_fn_kwargs["early_stop"] = early_stop

    return checkpoint_wrapper.checkpoint_wrapper(
        module,
        checkpoint_impl=checkpoint_impl,
        checkpoint_fn=checkpoint_fn,
        **checkpoint_fn_kwargs,
    )
