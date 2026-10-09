"""Gemma 4 on the TPU GraphTrainer path.

Gemma 4 is not an upstream ``torchtitan/models`` model, so unlike the qwen3 and
llama3 wrappers there is no ``experiments.graph_trainer.gemma4`` to reuse. This
module supplies the two pieces ``to_graph_trainer_config`` needs from a
GraphTrainer model registry:

* ``GraphTrainerGemma4Model``: ``Gemma4Model`` whose ``init_states`` runs with
  SimpleFSDP's active parametrization disabled, so parameters are initialized
  on their local shards (same shape as ``GraphTrainerQwen3Model``).
  ``Config`` adds no fields, so the recipe's own model config is re-wrapped
  field for field and keeps the flavor's hybrid sliding/global layers,
  ``SlidingWindowScaledDotProductAttention`` on sliding layers, the MaxText
  style ``get_nparams_and_flops``, and ``enable_weight_tying``.
* ``parallelize_gemma4``: the SimpleFSDP + aot_fx_trace setup. It applies the
  same uneven-dimension FSDP policy as the eager TPU path
  (``configure_gemma4_simple_fsdp_shard_dims``: the 262144-token vocabulary
  and the 128-expert banks switch to a divisible hidden axis when the FSDP
  degree does not divide dim 0), then upstream's ``apply_simple_fsdp`` and
  ``apply_compile``.

Only SimpleFSDP / HSDP is supported here.

Scan (``--tpu_config.use_scan_layers``) is not supported: the sliding and
global layers have different ``head_dim`` / ``n_kv_heads`` and different QKV
projections (``QKVLinear`` vs ``QKVLinearKeqV``), so
``check_layers_are_stackable`` refuses them. Graph split is the large-size
path for Gemma 4.
"""

import dataclasses
from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    apply_simple_fsdp,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    disable_active_parametrization,
)
from torchtitan.experiments.tpu.examples import gemma4 as tpu_gemma4
from torchtitan.experiments.tpu.examples.gemma4.infra.parallelize import (
    configure_gemma4_simple_fsdp_shard_dims,
)
from torchtitan.experiments.tpu.examples.gemma4.model import Gemma4Model
from torchtitan.models.common.moe import configure_moe_aux_loss_reduction
from torchtitan.protocols.model_spec import ModelSpec


class GraphTrainerGemma4Model(Gemma4Model):
  """``Gemma4Model`` that initializes its states on SimpleFSDP local shards."""

  @dataclass(kw_only=True, slots=True)
  class Config(Gemma4Model.Config):
    pass

  def init_states(
      self,
      *,
      buffer_device: torch.device | None = None,
  ) -> None:
    with disable_active_parametrization():
      super().init_states(buffer_device=buffer_device)


def parallelize_gemma4(
    model: GraphTrainerGemma4Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> nn.Module:
  """Applies SimpleFSDP (TPU uneven-dim policy) and graph capture to Gemma 4.

  ``ac_config`` is accepted for the ``parallelize_fn`` protocol but unused: the
  graph trainer rematerializes its joint graph with
  ``tpu_config.graph_trainer_memory_policy`` instead of eager AC wrappers.
  """
  del ac_config
  unsupported = [
      name
      for name, enabled in (
          ("tensor", parallel_dims.tp_enabled),
          ("context", parallel_dims.cp_enabled),
          ("pipeline", parallel_dims.pp_enabled),
          ("expert", parallel_dims.ep_enabled),
      )
      if enabled
  ]
  if unsupported:
    raise NotImplementedError(
        "TPU GraphTrainer Gemma 4 supports SimpleFSDP/HSDP only; "
        f"{', '.join(unsupported)} parallelism is not supported."
    )
  # SimpleFSDP wraps tok_embeddings with a read-only `weight` property, which
  # is incompatible with Decoder.init_states re-tying tok_embeddings.weight
  # (same check as the eager TPU path in examples/gemma4/infra/parallelize.py).
  if getattr(model, "enable_weight_tying", False):
    raise ValueError(
        "SimpleFSDP requires an untied Gemma4 model; build it with "
        "model_registry(..., enable_weight_tying=False)."
    )

  annotate_module_fqns(model)

  if parallelism.spmd_backend == "spmd_types":
    model.parallelize(parallel_dims)

  configure_moe_aux_loss_reduction(model, parallel_dims)

  # Apply simple_fsdp unconditionally. The `fsdp` mesh always exists with a
  # real backend (see ParallelDims._mesh_exist), even at degree 1, so that
  # MixedPrecisionPolicy's param_dtype cast still applies in single-chip runs.
  configure_gemma4_simple_fsdp_shard_dims(model, parallel_dims)
  model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)

  return apply_compile(
      model,
      compile_config=compile_config,
      parallelism=parallelism,
      parallel_dims=parallel_dims,
      dump_folder=dump_folder,
  )


def model_registry(flavor: str) -> ModelSpec:
  """Returns the GraphTrainer ``ModelSpec`` for a TPU Gemma 4 flavor.

  ``to_graph_trainer_config`` only takes the model class, ``parallelize_fn``
  and ``pipelining_fn`` from this spec; the field values (weight tying,
  converters, sharding) come from the recipe's own model config.

  Args:
    flavor: a flavor of ``experiments.tpu.examples.gemma4.model_registry``
      (``testmodel``, ``testmodel_moe``, ``debugmodel``, ``debugmodel_moe``,
      ``31B``, ``26B-A4B``).

  Returns:
    A spec with a ``GraphTrainerGemma4Model.Config`` model, the graph-trainer
    ``parallelize_gemma4``, and no ``pipelining_fn``.
  """
  model_spec = tpu_gemma4.model_registry(flavor, enable_weight_tying=False)
  # Same field-for-field re-wrap as upstream graph_trainer.qwen3's
  # model_registry: only the class changes (GraphTrainerGemma4Model.Config adds
  # no fields), so the flavor's layer configs are kept as built.
  model_config = GraphTrainerGemma4Model.Config(**{
      field.name: getattr(model_spec.model, field.name)
      for field in dataclasses.fields(model_spec.model)
  })
  return dataclasses.replace(
      model_spec,
      model=model_config,
      parallelize_fn=parallelize_gemma4,
      pipelining_fn=None,
  )


__all__ = [
    "GraphTrainerGemma4Model",
    "model_registry",
    "parallelize_gemma4",
]
