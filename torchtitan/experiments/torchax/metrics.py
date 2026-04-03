# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
r"""copied from torchtitan/components/metrics.py with change for torchax.
https://github.com/pytorch/torchtitan/blob/fbafd44da2baef0afac58989f07d799c4251bdef/torchtitan/components/metrics.py#L326

- Removed all irrelevant class/methods but MetricsProcessor.
- Removed external logger and device_memory_monitor.
- Keep the same MFU fomula for a fair comparison.
- Added hardcoded TPU peak FLOPS.
"""

import time
from typing import Any

import torch
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.experiments.torchax import torchax_job_config
import torchtitan.distributed
from torchtitan.tools import utils
from torchtitan.tools.logging import logger


def get_peak_flops(accelerator: str, num_devices, tpu_megacore) -> int:
    """Gets the system-wide peak theoretical BF16 TFLOPS for calculating MFU.

    Args:
        accelerator: The type of accelerator device ('v4', 'v5e', 'H100', etc.).
        num_devices: The total count of addressable devices, as reported by
          jax.device_count().
        tpu_megacore: If True and accelerator is v4/v5p, ensure denominator is
          per-chip peak flops (2x per core).

    Returns:
        The total peak theoretical TFLOPS.
    """
    tflops_per_core = {
        # TPUs tflops. For TPUs with 2 cores per chip, we convert quoted peak
        # tflops per chip to per core.
        # See https://cloud.google.com/tpu/docs/v4
        "v4": 275 // 2,
        # See https://cloud.google.com/tpu/docs/v5p
        "v5p": 459 // 2,
        # See https://cloud.google.com/tpu/docs/v5e
        "v5e": 197,
        # See https://cloud.google.com/tpu/docs/v6e
        "v6e": 918,
        # See https://cloud.google.com/tpu/docs/tpu7x
        "v7x": 2307 // 2,
        # GPU tflops. We provide quoted dense peak tflops for SXM based GPUs.
        # See https://www.nvidia.com/en-us/data-center/a100/
        "a100": 312,
        # See https://www.nvidia.com/en-us/data-center/h100/
        "h100": 989,
        }

    if accelerator not in tflops_per_core:
        raise ValueError(f"Unsupported accelerator type: {accelerator}")

    flops_per_jax_device = tflops_per_core[accelerator]
    # If megacore is enabled for v4/v5p, we multiply by 2 as each JAX device
    # corresponds to one chip containing two cores.
    if tpu_megacore and accelerator in ("v4", "v5p"):
        flops_per_jax_device *= 2

    return int(flops_per_jax_device * num_devices * 1_000_000_000_000)


class TorchaxMetricsProcessor:
    """Metrics processor to processes the metrics and log metrics.

    The current MetricsProcessor log some metrics to STDOUT and some metrics to
    TensorBoard or WandB.

    Args:
        job_config (JobConfig): Job configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
        tag (Optional[str]): Tag to use for TensorBoard or WandB. Defaults to None.
    """

    parallel_dims: torchtitan.distributed.ParallelDims
    job_config: torchax_job_config.TorchaxJobConfig
    color: utils.NoColor | utils.Color

    gpu_peak_flops: int
    num_global_devices: int
    ntokens_since_last_log: int
    data_loading_times: list[float]
    time_last_log: float

    num_flops_per_token: int
    optimizers: OptimizersContainer | None
    lr_schedulers: LRSchedulersContainer | None
    model_parts: list[torch.nn.Module] | None

    def __init__(
        self,
        job_config: torchax_job_config.TorchaxJobConfig,
        parallel_dims: torchtitan.distributed.ParallelDims,
        accelerator: str,
        num_global_devices: int,
        tag: str | None = None,
    ):
        # self.logger = _build_metric_logger(job_config, parallel_dims, tag)
        self.parallel_dims = parallel_dims
        self.job_config = job_config
        # self.device_memory_monitor = build_device_memory_monitor()
        # used for colorful printing
        self.color = (
            utils.NoColor()
            if job_config.metrics.disable_color_printing
            else utils.Color()
        )

        self.gpu_peak_flops = get_peak_flops(
            accelerator,
            num_global_devices,
            job_config.torchax_config.tpu_megacore,
        )
        self.num_global_devices = num_global_devices
        self.ntokens_since_last_log = 0
        self.data_loading_times = []
        self.time_last_log = time.perf_counter()

        # These variables have to be set later as they depend on other components or model.
        self.num_flops_per_token = -1
        self.optimizers = None
        self.lr_schedulers = None
        self.model_parts = None

    def should_log(self, step: int) -> bool:
        return step == 1 or step % self.job_config.metrics.log_freq == 0

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ):
        assert self.num_flops_per_token > 0, "num_flops_per_token must be set"

        time_delta = time.perf_counter() - self.time_last_log

        # system-wide tokens per second, abbreviated as tps
        # Note: We set the non_data_parallel_size to 1 to get the system-wide
        # tps. See third_party/py/torchtitan/experiments/torchax/distributed.py
        tps = self.ntokens_since_last_log / (
            time_delta * self.parallel_dims.non_data_parallel_size
        )
        per_device_tps = tps / self.num_global_devices
        # model FLOPS utilization
        # For its definition and calculation, please refer to the PaLM paper:
        # https://arxiv.org/abs/2204.02311
        mfu = 100 * self.num_flops_per_token * tps / self.gpu_peak_flops
        tflops = self.num_flops_per_token * tps / 1e12

        actual_steps_since_last_log = max(1, len(self.data_loading_times))
        time_end_to_end = time_delta / actual_steps_since_last_log
        time_data_loading = sum(self.data_loading_times) / actual_steps_since_last_log
        time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta


        metrics = {
            "loss_metrics/global_avg_loss": global_avg_loss,
            "loss_metrics/global_max_loss": global_max_loss,
            "grad_norm": grad_norm,
            "throughput(tps)": tps,
            "throughput(per_device_tps)": per_device_tps,
            "tflops": tflops,
            "mfu(%)": mfu,
            "time_metrics/end_to_end(s)": time_end_to_end,
            "time_metrics/data_loading(s)": time_data_loading,
            "time_metrics/data_loading(%)": time_data_loading_pct,
        }

        if extra_metrics:
            metrics.update(extra_metrics)

        color = self.color
        logger.info(
            f"{color.red}step: {step:2}  "
            f"{color.green}loss: {global_avg_loss:7.4f}  "
            f"{color.orange}step_time: {time_end_to_end:7.4f}  "
            f"{color.blue}tps: {round(tps):,}  "
            f"per_device_tps: {round(per_device_tps):,}  "
            f"{color.cyan}tflops: {tflops:,.2f}  "
            f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
        )

        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
