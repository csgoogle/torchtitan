"""Muse Spark on the TPU GraphTrainer path.

Mirrors ``graph_trainer/muse_glimmer.py``; there is no upstream
``experiments.graph_trainer.muse_spark``, so the GraphTrainer model class and
setup live here:

* The model config comes from ``experiments.tpu.muse_spark.model_registry``, so
  every layer keeps the window-aware SDPA inner attention that the Splash patch
  turns into causal / sliding-window kernels, and the recipe's TPU MoE
  converters (``TpuGroupedExperts``) are kept field for field.
* Before SimpleFSDP, expert banks, router gates and vocabulary-sized weights
  whose dim 0 does not divide the FSDP degree switch to dim-1 sharding
  (``configure_muse_spark_simple_fsdp_shard_dims``).

Only SimpleFSDP / HSDP is supported, as on the regular Muse Spark TPU path.
"""

import dataclasses
from dataclasses import dataclass

import torch
from torch import nn
from torchtitan.config import ParallelismConfig
from torchtitan.config import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import annotate_module_fqns
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.simple_fsdp import disable_active_parametrization
from torchtitan.experiments.tpu import muse_spark as tpu_muse_spark
from torchtitan.experiments.tpu.muse_spark.infra.parallelize import configure_muse_spark_simple_fsdp_shard_dims
from torchtitan.models.muse_spark.model import MuseSparkModel
from torchtitan.protocols.model_spec import ModelSpec


class GraphTrainerMuseSparkModel(MuseSparkModel):

  @dataclass(kw_only=True, slots=True)
  class Config(MuseSparkModel.Config):
    pass

  def init_states(
      self,
      *,
      buffer_device: torch.device | None = None,
  ) -> None:
    with disable_active_parametrization():
      super().init_states(buffer_device=buffer_device)


def parallelize_muse_spark(
    model: GraphTrainerMuseSparkModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    dump_folder: str,
    **kwargs,
) -> nn.Module:
  """Applies the TPU uneven-shard FSDP policy, SimpleFSDP and graph capture."""
  del kwargs  # ac_config: GraphTrainer rematerializes the joint graph instead.
  # The ported model carries no TP/EP sharding plans (see
  # muse_spark/infra/parallelize.py).
  for name, enabled in (
      ("Tensor", parallel_dims.tp_enabled),
      ("Expert", parallel_dims.ep_enabled),
      ("Context", parallel_dims.cp_enabled),
      ("Pipeline", parallel_dims.pp_enabled),
  ):
    if enabled:
      raise NotImplementedError(
          f"{name} parallelism is not supported for GraphTrainer Muse Spark"
          " on TPU."
      )
  annotate_module_fqns(model)
  configure_muse_spark_simple_fsdp_shard_dims(model, parallel_dims)
  model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)
  return apply_compile(
      model,
      compile_config=compile_config,
      parallelism=parallelism,
      parallel_dims=parallel_dims,
      dump_folder=dump_folder,
  )


def model_registry(flavor: str) -> ModelSpec:
  """Returns the GraphTrainer ``ModelSpec`` for a TPU Muse Spark flavor.

  ``to_graph_trainer_config`` only takes the model class, ``parallelize_fn``
  and ``pipelining_fn`` from this spec; the field values come from the recipe's
  own model config.
  """
  model_spec = tpu_muse_spark.model_registry(flavor)
  model_config = GraphTrainerMuseSparkModel.Config(**{
      field.name: getattr(model_spec.model, field.name)
      for field in dataclasses.fields(model_spec.model)
  })
  return dataclasses.replace(
      model_spec,
      model=model_config,
      parallelize_fn=parallelize_muse_spark,
      pipelining_fn=None,
  )
