"""Parallelization for OLMo 2 on TPU.

OLMo 2 is a plain ``Decoder`` with homogeneous dense layers, so it runs the
TPU llama3 path unchanged (AC -> per-block compile -> SimpleFSDP or FSDP2,
``tpu_compile_mode`` included); this module only rejects what that path does
not support for OLMo 2 yet.
"""

from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.tpu.llama3.infra.parallelize import parallelize_llama


def _validate_supported(
    parallel_dims: torchtitan.distributed.ParallelDims,
    parallelism: torchtitan.config.ParallelismConfig,
) -> None:
  # The q/k norms span all heads (FlatHeadsRMSNorm); a TP shard holds only
  # some of them, so its local statistic would be wrong.
  if parallel_dims.tp_enabled:
    raise NotImplementedError(
        "Tensor parallelism is not supported for OLMo 2 on TPU: its q/k norms"
        " span all heads."
    )
  if parallel_dims.cp_enabled:
    raise NotImplementedError(
        "Context parallelism is not supported for OLMo 2 on TPU yet."
    )
  if parallel_dims.pp_enabled:
    raise NotImplementedError(
        "Pipeline parallelism is not supported for OLMo 2 on TPU yet."
    )
  if parallelism.spmd_backend != "partial_dtensor":
    raise NotImplementedError(
        "OLMo 2 on TPU supports only parallelism.spmd_backend=partial_dtensor."
    )


def parallelize_olmo2(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    parallelism: torchtitan.config.ParallelismConfig,
    **kwargs,
) -> nn.Module:
  """Validates the parallelism, then applies the TPU llama3 parallelization.

  Args:
    model: The ``Olmo2Model`` to parallelize (may be on meta device).
    parallel_dims: Parallel dimensions for this run.
    parallelism: Parallelism config.
    **kwargs: Passed to ``parallelize_llama`` (training, compile_config,
      ac_config, dump_folder, skip_dp, tpu_compile_mode).

  Returns:
    The parallelized model.
  """
  _validate_supported(parallel_dims, parallelism)
  return parallelize_llama(
      model, parallel_dims=parallel_dims, parallelism=parallelism, **kwargs
  )
