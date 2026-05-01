"""Parallelization utilities for Llama models on TPU.

This module is used in distributed training tests configured with
`model_name="llama3_tpu".
"""

from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.experiments.tpu import workarounds
from torchtitan.models.llama3.infra import parallelize as dtensor_parallelize
from torchtitan.tools.logging import logger


def parallelize_llama(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: (
        torchtitan.config.JobConfig
        | torchtitan.experiments.tpu.tpu_job_config.TPUJobConfig
    ),
) -> nn.Module:
  """Parallelizes the Llama model based on the provided configuration and applies TPU-specific workarounds.

  Args:
    model: The nn.Module to be parallelized.
    parallel_dims: Contains the world mesh and other parallelization dimensions.
    job_config: The job configuration, potentially including TPU-specific
      settings.

  Returns:
    The parallelized nn.Module.
  """

  # TODO b/499075866: re-enable CPU offload for TPU torchtitan run.
  if job_config.training.enable_cpu_offload:
    logger.warning(
        "CPU offload is not supported for TPU torchtitan run, setting it to"
        " False."
    )
    job_config.training.enable_cpu_offload = False

  if (
      isinstance(job_config, tpu_job_config.TPUJobConfig)
      and job_config.splash_attention_kernel.use_splash_attention_kernel
  ):
    logger.info(
        "Applying Splash Attention patch with custom block sizes:"
        f" q={job_config.splash_attention_kernel.sa_block_q},"
        f" kv={job_config.splash_attention_kernel.sa_block_kv},"
        f" dkv={job_config.splash_attention_kernel.sa_block_dkv},"
        f" kv_compute={job_config.splash_attention_kernel.sa_block_kv_compute},"
        f" q_dkv={job_config.splash_attention_kernel.sa_block_q_dkv},"
        f" kv_dkv={job_config.splash_attention_kernel.sa_block_kv_dkv},"
        f" kv_dkv_compute={job_config.splash_attention_kernel.sa_block_kv_dkv_compute},"
        f" q_dq={job_config.splash_attention_kernel.sa_block_q_dq},"
        f" kv_dq={job_config.splash_attention_kernel.sa_block_kv_dq}"
    )
    workarounds.use_splash_attention_patch(
        model,
        block_q=job_config.splash_attention_kernel.sa_block_q,
        block_kv=job_config.splash_attention_kernel.sa_block_kv,
        block_dkv=job_config.splash_attention_kernel.sa_block_dkv,
        block_kv_compute=job_config.splash_attention_kernel.sa_block_kv_compute,
        block_q_dkv=job_config.splash_attention_kernel.sa_block_q_dkv,
        block_kv_dkv=job_config.splash_attention_kernel.sa_block_kv_dkv,
        block_kv_dkv_compute=job_config.splash_attention_kernel.sa_block_kv_dkv_compute,
        block_q_dq=job_config.splash_attention_kernel.sa_block_q_dq,
        block_kv_dq=job_config.splash_attention_kernel.sa_block_kv_dq,
        use_fused_bwd_kernel=job_config.splash_attention_kernel.sa_use_fused_bwd_kernel,
        q_layout=job_config.splash_attention_kernel.sa_q_layout,
        k_layout=job_config.splash_attention_kernel.sa_k_layout,
        v_layout=job_config.splash_attention_kernel.sa_v_layout,
    )
  if (
      isinstance(job_config, tpu_job_config.TPUJobConfig)
      and job_config.loss_kernel.use_loss_kernel
  ):
    workarounds.use_output_projection_patch(model)
  dtensor_parallelize.parallelize_llama(model, parallel_dims, job_config)
  return model
