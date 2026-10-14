"""OLMo 2 on the TPU GraphTrainer path (``--tpu_config.use_graph_trainer``).

Upstream ``experiments/graph_trainer`` has no OLMo 2 model, so this module
supplies the three pieces its models have, the same way the TPU ``gpt_oss``
wrapper does: a model class whose ``init_states`` runs with SimpleFSDP's
parametrization disabled, a ``parallelize_fn`` that annotates module FQNs,
applies SimpleFSDP and hands the model to GraphTrainer's compile, and a
``model_registry`` built on ``experiments.tpu.examples.olmo2``.

Only SimpleFSDP / HSDP is supported, as on the eager path: the full-width q/k
norms rule out TP, and CP / PP are not wired up for OLMo 2.
"""

import dataclasses
from dataclasses import dataclass

import torch
from torch import nn
from torchtitan.config import CompileConfig
from torchtitan.config import ParallelismConfig
from torchtitan.config import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import annotate_module_fqns
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.simple_fsdp import disable_active_parametrization
from torchtitan.experiments.tpu.examples import olmo2 as tpu_olmo2
from torchtitan.experiments.tpu.examples.olmo2.infra.parallelize import validate_olmo2_parallelism
from torchtitan.protocols.model_spec import ModelSpec


class GraphTrainerOlmo2Model(tpu_olmo2.Olmo2Model):
  """``Olmo2Model`` initialized outside SimpleFSDP's parametrization."""

  @dataclass(kw_only=True, slots=True)
  class Config(tpu_olmo2.Olmo2Model.Config):
    pass

  def init_states(
      self,
      *,
      buffer_device: torch.device | None = None,
  ) -> None:
    with disable_active_parametrization():
      super().init_states(buffer_device=buffer_device)


def parallelize_olmo2(
    model: GraphTrainerOlmo2Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    dump_folder: str,
    **kwargs,
) -> nn.Module:
  """Annotates FQNs, applies SimpleFSDP, then GraphTrainer's compile.

  Args:
    model: The ``GraphTrainerOlmo2Model`` (may be on meta device).
    parallel_dims: Parallel dimensions for this run.
    training: Training config (mixed-precision dtypes).
    parallelism: Parallelism config.
    compile_config: GraphTrainer compile config.
    dump_folder: Dump folder for compile artifacts.
    **kwargs: Ignored (``ac_config``: GraphTrainer's memory policy replaces it).

  Returns:
    The parallelized, compiled model.
  """
  del kwargs  # Unused.
  validate_olmo2_parallelism(parallel_dims, parallelism)
  annotate_module_fqns(model)
  model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)
  return apply_compile(
      model,
      compile_config=compile_config,
      parallelism=parallelism,
      parallel_dims=parallel_dims,
      dump_folder=dump_folder,
  )


def model_registry(flavor: str) -> ModelSpec:
  """Returns the GraphTrainer ``ModelSpec`` for an OLMo 2 flavor.

  ``to_graph_trainer_config`` takes only the model class, ``parallelize_fn``
  and ``pipelining_fn`` from this spec; the field values come from the recipe's
  own model config.

  Args:
    flavor: A key of ``olmo2_configs``.

  Returns:
    A spec with a ``GraphTrainerOlmo2Model.Config`` model.
  """
  model_spec = tpu_olmo2.model_registry(flavor)
  model_config = GraphTrainerOlmo2Model.Config(**{
      field.name: getattr(model_spec.model, field.name)
      for field in dataclasses.fields(model_spec.model)
  })
  return dataclasses.replace(
      model_spec,
      model=model_config,
      parallelize_fn=parallelize_olmo2,
      pipelining_fn=None,
  )
