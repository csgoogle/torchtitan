"""Parallelization utilities for Qwen3 models on TPU."""

from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.models.qwen3 import parallelize as dtensor_parallelize


from torchtitan.experiments.graph_trainer.simple_fsdp import (
    MixedPrecisionPolicy as SimpleFSDPMixedPrecisionPolicy,
    data_parallel as simple_fsdp_data_parallel,
)
from torchtitan.tools.logging import logger

def parallelize_qwen3(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config: torchtitan.config.ActivationCheckpointConfig,
    dump_folder: str,
    skip_dp: bool = False,
) -> nn.Module:
  """Parallelizes the Qwen3 model, optionally applying Simple FSDP.

  This wrapper delegates to the upstream Qwen3 parallelize function. If 
  `--parallelism.use_simple_fsdp` is set, it skips upstream Data Parallelism
  and instead applies the TPU-specific Simple FSDP implementation.
  """
  if parallelism.use_simple_fsdp:
      # Apply all other parallelisms but skip data parallelism
      dtensor_parallelize.parallelize_qwen3(
          model,  # pyrefly: ignore[bad-argument-type]
          ac_config=ac_config,
          compile_config=compile_config,
          dump_folder=dump_folder,
          parallel_dims=parallel_dims,
          parallelism=parallelism,
          skip_dp=True,
          training=training,
      )

      dp_mesh_names = (
          ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
      )
      dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

      dp_mode = "hybrid_shard" if parallel_dims.dp_replicate_enabled else "fully_shard"
      mp_policy = SimpleFSDPMixedPrecisionPolicy(
          param_dtype=torchtitan.config.TORCH_DTYPE_MAP[training.mixed_precision_param],
          reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[training.mixed_precision_reduce],
      )
      model = simple_fsdp_data_parallel(
          model,
          dp_mesh,
          mode=dp_mode,
          mp_policy=mp_policy,
      )
      logger.info(f"Applied SimpleFSDP ({dp_mode}) to the model")
  else:
      dtensor_parallelize.parallelize_qwen3(
          model,  # pyrefly: ignore[bad-argument-type]
          ac_config=ac_config,
          compile_config=compile_config,
          dump_folder=dump_folder,
          parallel_dims=parallel_dims,
          parallelism=parallelism,
          skip_dp=skip_dp,
          training=training,
      )
  return model
