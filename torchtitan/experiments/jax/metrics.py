"""Metrics tracking for JAX experiment."""

import time
from typing import Any

from torchtitan.experiments.torchax.metrics import get_peak_flops
from torchtitan.tools import utils
from torchtitan.tools.logging import logger


class JaxMetricsProcessor:
    def __init__(
        self,
        job_config,
        accelerator: str,
        num_global_devices: int,
        log_freq: int = 10,
    ):
        self.log_freq = log_freq
        self.color = (
            utils.NoColor()
            if job_config.metrics.disable_color_printing
            else utils.Color()
        )
        self.gpu_peak_flops = get_peak_flops(
            accelerator,
            num_global_devices,
            job_config.jax_config.tpu_megacore,
        )
        self.num_global_devices = num_global_devices
        self.ntokens_since_last_log = 0
        self.data_loading_times: list[float] = []
        self.time_last_log = time.perf_counter()
        self.num_flops_per_token: int = 0

    def should_log(self, step: int) -> bool:
        return step == 1 or step % self.log_freq == 0

    def log(
        self,
        step: int,
        loss: float,
        extra_metrics: dict[str, Any] | None = None,
    ):
        time_delta = time.perf_counter() - self.time_last_log
        tps = self.ntokens_since_last_log / time_delta
        per_device_tps = tps / self.num_global_devices
        mfu = (
            100 * self.num_flops_per_token * tps / self.gpu_peak_flops
            if self.num_flops_per_token > 0 else 0.0
        )
        tflops = self.num_flops_per_token * tps / 1e12

        actual_steps = max(1, len(self.data_loading_times))
        time_e2e = time_delta / actual_steps
        time_data = sum(self.data_loading_times) / actual_steps

        c = self.color
        logger.info(
            f'{c.red}step: {step:2}  '
            f'{c.green}loss: {loss:7.4f}  '
            f'{c.orange}step_time: {time_e2e:7.4f}s  '
            f'{c.blue}tps: {round(tps):,}  '
            f'per_device_tps: {round(per_device_tps):,}  '
            f'{c.cyan}tflops: {tflops:,.2f}  '
            f'{c.magenta}mfu: {mfu:.2f}%{c.reset}'
        )

        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
