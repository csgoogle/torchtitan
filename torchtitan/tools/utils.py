# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import gc
import os
import subprocess
import time
from collections.abc import Generator
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import psutil
import torch
import torch.distributed as dist
from torch._utils import _get_available_device_type, _get_device_module

from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import logger


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def has_cuda_capability(major: int, minor: int) -> bool:
    # torch.version.hip is None excludes ROCm (capability is a tuple on AMD too).
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability() >= (major, minor)
    )


def has_rocm_capability(major: int, minor: int) -> bool:
    is_rocm = torch.cuda.is_available() and torch.version.hip is not None
    return is_rocm and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )


def get_device_info() -> tuple[str, ModuleType]:
    device_type = _get_available_device_type() or "cuda"
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


# device_type, device_module = get_device_info()
_device_type: str | None = None
_cpu_device_module: ModuleType | None = None
_tpu_device_module: ModuleType | None = None


def get_device_type() -> str:
    """Returns the device type this process runs on, detected once and cached.

    Detection is side-effect free: torch._utils resolves TPU through
    torch.tpu.is_available(), a PCI scan that does not initialize the PJRT
    runtime. Which accelerator is visible is fixed by the environment before
    torch is imported (see experiments/tpu/device_env.py), so the answer cannot
    change during the process and is safe to cache.
    """
    global _device_type
    if _device_type is None:
        _device_type = _get_available_device_type() or "cpu"
    return _device_type


def _get_cpu_module() -> ModuleType:
    """Returns torch.cpu with the device-module functions it is missing."""
    global _cpu_device_module
    if _cpu_device_module is not None:
        return _cpu_device_module

    device_module = _get_device_module("cpu")

    # torch.cpu doesn't have get_device_name
    def get_device_name(device=None):
        return "cpu"

    setattr(device_module, "get_device_name", get_device_name)

    def current_device():
        # torch.cpu.current_device() returns the string "cpu"; callers here
        # need an index, e.g. torch.device(device_type, current_device()).
        return dist.get_rank()

    setattr(device_module, "current_device", current_device)

    def get_device_properties(device=None):
        return SimpleNamespace(total_memory=psutil.virtual_memory().total)

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
    _cpu_device_module = device_module
    return device_module


def _get_tpu_module() -> ModuleType:
    """Returns torch.tpu with the device-module functions it is missing.

    torch_tpu natively provides get_device_name, get_device_properties,
    set_device, current_device, device_count, synchronize and
    is_triton_capable, so only the memory-management trio is shimmed here.
    """
    global _tpu_device_module
    if _tpu_device_module is not None:
        return _tpu_device_module

    device_module = _get_device_module("tpu")

    def reset_peak_memory_stats(device=None):
        pass

    setattr(device_module, "reset_peak_memory_stats", reset_peak_memory_stats)

    def empty_cache():
        pass

    setattr(device_module, "empty_cache", empty_cache)

    def memory_stats(device=None):
        # torch_tpu exposes memory stats through the modern torch.accelerator
        # namespace, but torchtitan expects them on the legacy device-module
        # namespace (torch.tpu). Route them across.
        return torch.accelerator.memory_stats()

    setattr(device_module, "memory_stats", memory_stats)
    _tpu_device_module = device_module
    return device_module


def get_device_module() -> ModuleType:
    """Returns the device module for the detected device type."""
    device_type = get_device_type()
    if device_type == "tpu":
        return _get_tpu_module()
    if device_type == "cpu":
        return _get_cpu_module()
    return _get_device_module(device_type)  # e.g. torch.cuda


def get_local_device() -> torch.device:
    """Return this process's device under LOCAL_RANK or visible-device launch.

    Launchers normally expose multiple accelerators per process and set
    LOCAL_RANK to identify which local device the process should use. Some
    launchers instead mask each process down to a single accelerator with
    CUDA_VISIBLE_DEVICES or another backend-specific visible-device mask. When
    exactly one device is visible, target device index 0; otherwise preserve
    the existing LOCAL_RANK mapping.

    cpu and tpu are index-free: a process owns exactly one of them, so
    LOCAL_RANK is not a device index on those lanes. tpu cannot use the
    visible-device heuristic below either, because torch.tpu.device_count()
    reports the host's chip count -- not the process's single addressable chip
    -- until the PJRT runtime is initialized, which happens after callers of
    this function have already selected a device.
    """
    device_type = get_device_type()
    if device_type in ["cpu", "tpu"]:
        return torch.device(device_type)

    local_rank_str = os.environ.get("LOCAL_RANK")
    if local_rank_str is None:
        raise ValueError("LOCAL_RANK must be set before selecting a local device.")
    try:
        local_rank = int(local_rank_str)
    except ValueError as e:
        raise ValueError(
            "LOCAL_RANK environment variable must be a valid integer, "
            f"got: {local_rank_str}"
        ) from e
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative, got: {local_rank}")

    device_module = get_device_module()
    num_devices = None
    if hasattr(device_module, "device_count"):
        num_devices = device_module.device_count()
    device_index = 0 if num_devices == 1 else local_rank
    if num_devices is not None and num_devices > 1 and device_index >= num_devices:
        raise ValueError(
            f"LOCAL_RANK={local_rank} is outside the visible {device_type} "
            f"device count ({num_devices}). If each process is launched with a "
            f"single visible {device_type} device, set the visible-device mask "
            "per process so device_count() returns 1."
        )
    return torch.device(device_type, device_index)


# used to avoid stragglers in garbage collection
class GarbageCollection:
    def __init__(self, gc_freq: int = 1000, debug: bool = False):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        self.gc_freq = gc_freq
        self.debug = debug
        gc.disable()
        self.collect("Initial GC collection")
        if debug:
            from torch.utils.viz._cycles import warn_tensor_cycles

            if torch.distributed.get_rank() == 0:
                warn_tensor_cycles()

    @sl.log_trace_span("gc_collect")
    def run(self, step_count: int) -> bool:
        """Run a GC cycle if this step should collect. Returns True when a
        collection actually ran, False otherwise."""
        if self.debug:
            self.collect(
                "Force GC to perform collection to obtain debug information",
                generation=2,
            )
            gc.collect()
            sl.add_step_tag("gc")
            return True
        if step_count > 1 and step_count % self.gc_freq == 0:
            self.collect("Performing periodic GC collection")
            sl.add_step_tag("gc")
            return True
        return False

    @staticmethod
    def collect(reason: str, generation: int = 1):
        begin = time.monotonic()
        gc.collect(generation)
        logger.info("[GC] %s took %.2f seconds", reason, time.monotonic() - begin)


# hardcoded BF16 type peak flops for NVIDIA A100, H20, H100, H200, B200 GPU,
# AMD MI250, MI300X, MI325X, MI355X, Intel PVC, and AWS Trainium/Inferentia
def get_peak_flops(device_name: str) -> float:
    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [
            line
            for line in result.stdout.splitlines()
            if "NVIDIA" in line and "H100" in line
        ]
        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "A6000" in device_name:
        # data from https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/
        # quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20(1).pdf
        # NOTE: 309.7 TFLOPS is with sparsity; dense value is half.
        return 154.85e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    elif "H20" in device_name:
        # NVIDIA H20 is a region-specific GPU variant.
        # Since first-hand specifications do not seem to be readily available on
        # NVIDIA's official global website, we refer to technical reports from
        # Tom's Hardware. The peak BF16/FP16 Tensor performance is reported as
        # 148 TFLOPS.
        # Ref: https://www.tomshardware.com/news/
        # nvidias-latest-regulation-compliant-gpu-for-china-has-been-delayed-to-early-next-year
        return 148e12
    elif "GB200" in device_name or "GB300" in device_name:
        # Grace Blackwell Superchips (Grace CPU + Blackwell GPU)
        # BF16 dense per GPU: 2,500 TFLOPS (half of 5,000 TFLOPS with sparsity)
        # GB200 data from https://www.nvidia.com/en-us/data-center/dgx-gb200
        # GB300 data from https://www.nvidia.com/en-us/data-center/dgx-gb300
        return 2.5e15
    elif "B300" in device_name or "B200" in device_name:
        # data from https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
        # Checked after GB300 to avoid false match on "GB300"
        return 2.25e15
    elif "MI355X" in device_name:
        # MI355X data from https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
        return 2500e12
    elif "MI300X" in device_name or "MI325X" in device_name:
        # MI300X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
        # MI325X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
        return 1300e12
    elif "MI250X" in device_name:
        # data from https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html (per GCD)
        return 191.5e12
    elif "Data Center GPU Max 1550" in device_name:
        # Also known as Ponte Vecchio (PVC).
        # data from https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
        # Dot Product Accumulate Systolic (DPAS):
        # - Freq: 1300MHz
        # - #ops: 512
        # Full EU mode (i.e. 512 max compute units): 340.8 TFLOPS (BF16)
        # Standard EU mode (i.e. 448 max compute units): 298.2 TFLOPS (BF16)
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6
    elif "l40s" in device_name:
        # data from: "https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413"
        return 362e12
    elif "tpu" in device_name.lower():
        # torch_tpu reports "TPU v6e"; older torchtitan builds reported
        # "tpu-v6e" (and "tpu-7x" for v7). Match both, case-insensitively.
        tpu_name = device_name.lower().replace("-", " ")
        if "v4" in tpu_name:
            # data from https://docs.cloud.google.com/tpu/docs/v4
            return 275e12
        elif "v5e" in tpu_name:
            # data from https://docs.cloud.google.com/tpu/docs/v5e
            return 197e12
        elif "v5p" in tpu_name:
            # data from https://docs.cloud.google.com/tpu/docs/v5p
            # The quoted number is per chip. We use this number as we assume
            # megacore is enabled for v5p.
            return 459e12
        elif "v6e" in tpu_name:
            # data from https://docs.cloud.google.com/tpu/docs/v6e
            return 918e12
        elif "v7" in tpu_name or "7x" in tpu_name:
            # data from https://docs.cloud.google.com/tpu/docs/tpu7x
            # The quoted number is per chip. We halve it for per-core FLOPS.
            return 2307e12 / 2
        logger.warning(
            f"Unrecognized TPU device name: {device_name}, fallback to A100"
        )
    elif "neuron" in device_name:
        # AWS Trainium/Inferentia: query chip type via torch.neuron
        # TensorEngine BF16 TFLOPS per NeuronCore × default Logical NeuronCore (LNC) count per device
        neuron_device_name = device_module.get_device_properties().name
        if neuron_device_name in ("trn1", "trn1n", "inf2"):
            # NeuronCore-v2 TensorEngine: 90 BF16 TFLOPS/core, LNC=1
            # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v2.html
            return 90e12 * 1
        elif neuron_device_name in ("trn2", "trn2n", "trn2u", "trn3", "trn3u"):
            # NeuronCore-v3/NeuronCore-v4 TensorEngine: 79 BF16 TFLOPS/core, LNC=2
            # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html
            # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v4.html
            return 79e12 * 2
        else:
            logger.warning(
                f"Unknown neuron device: {neuron_device_name}, fallback to trn2/trn3"
            )
            return 79e12 * 2

    else:  # for other GPU types, assume A100
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12


@dataclass(frozen=True)
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"
    orange = "\033[38;2;180;60;0m"
    turquoise = "\033[38;2;54;234;195m"


@dataclass(frozen=True)
class NoColor:
    black = ""
    red = ""
    green = ""
    yellow = ""
    blue = ""
    magenta = ""
    cyan = ""
    white = ""
    reset = ""
    orange = ""
    turquoise = ""


assert set(NoColor.__dataclass_fields__.keys()) == set(
    Color.__dataclass_fields__.keys()
), "NoColor must have the same fields as Color."


def check_if_feature_in_pytorch(
    feature_name: str,
    pull_request: str,
    min_nightly_version: str | None = None,
) -> None:
    if "git" in torch.__version__:  # pytorch is built from source
        # notify users to check if the pull request is included in their pytorch
        logger.warning(
            "Detected that the pytorch is built from source. Please make sure the PR "
            f"({pull_request}) is included in pytorch for correct {feature_name}."
        )
    elif min_nightly_version is not None and torch.__version__ < min_nightly_version:
        logger.warning(
            f"Detected that the pytorch version {torch.__version__} is older than "
            f"{min_nightly_version}. Please upgrade a newer version to include the "
            f"change in ({pull_request}) for correct {feature_name}."
        )


@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (torch.dtype): The desired default dtype inside the context manager.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16


    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)
