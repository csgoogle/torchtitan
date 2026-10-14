# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn

from torchtitan.config import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.qwen3 import (
    model_registry as graph_trainer_model_registry,
)
from torchtitan.experiments.graph_trainer.qwen3.parallelize import (
    parallelize_qwen3 as graph_trainer_parallelize_qwen3,
)
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.tpu.mfu_flops import (
    get_maxtext_decoder_nparams_and_flops,
)
from torchtitan.experiments.tpu.qwen3.infra.parallelize import (
    configure_qwen_simple_fsdp_shard_dims,
)
from torchtitan.experiments.tpu.qwen3.model import MaxTextQwen3Config
from torchtitan.protocols.model_spec import ModelSpec


class TpuGraphTrainerQwen3Model(GraphTrainerQwen3Model):
    """GraphTrainer Qwen3 for TPU; reports upstream torchtitan FLOPs."""

    @dataclass(kw_only=True, slots=True)
    class Config(GraphTrainerQwen3Model.Config):
        pass


class MaxTextTpuGraphTrainerQwen3Model(TpuGraphTrainerQwen3Model):
    """GraphTrainer Qwen3 for TPU; reports MaxText FLOPs.

    Selected when the recipe's model config is a ``MaxTextQwen3Config``, so a
    recipe reports the same FLOPs and MFU on GraphTrainer as on the regular
    TPU trainer.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TpuGraphTrainerQwen3Model.Config):
        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_maxtext_decoder_nparams_and_flops(self, model, seq_len)


def parallelize_qwen3(
    model: GraphTrainerQwen3Model,
    *,
    training: TrainingConfig,
    parallel_dims: ParallelDims,
    **kwargs,
) -> nn.Module:
    """Apply the TPU uneven-FSDP policy before setup."""
    configure_qwen_simple_fsdp_shard_dims(model, parallel_dims)
    model = graph_trainer_parallelize_qwen3(
        model,
        training=training,
        parallel_dims=parallel_dims,
        **kwargs,
    )
    # Creates process groups, so every rank runs it here, right after sharding.
    for holder in model.modules():
        if hasattr(holder, "build_split_meshes"):
            holder.build_split_meshes()
    return model


def model_registry(flavor: str, *, maxtext_flops: bool = False) -> ModelSpec:
    """GraphTrainer Qwen3 model spec for TPU.

    ``maxtext_flops`` selects MaxText FLOP accounting; the default reports
    upstream torchtitan FLOPs. Use ``model_registry_for`` to follow a recipe's
    own choice.
    """
    model_spec = graph_trainer_model_registry(flavor)
    config_cls = (
        MaxTextTpuGraphTrainerQwen3Model.Config
        if maxtext_flops
        else TpuGraphTrainerQwen3Model.Config
    )
    model_config = config_cls(
        **{
            field.name: getattr(model_spec.model, field.name)
            for field in dataclasses.fields(model_spec.model)
        }
    )
    return dataclasses.replace(
        model_spec,
        model=model_config,
        parallelize_fn=parallelize_qwen3,
    )


def model_registry_for(model_config: object) -> Callable[[str], ModelSpec]:
    """The GraphTrainer registry that keeps ``model_config``'s FLOP convention.

    GraphTrainer rebuilds the recipe's model config in its own class, which
    would drop the convention a ``MaxTextQwen3Config`` carries. Passing this
    registry to the conversion keeps MaxText FLOPs for such recipes and
    torchtitan FLOPs for a plain ``Qwen3Model.Config``.
    """
    return functools.partial(
        model_registry,
        maxtext_flops=isinstance(model_config, MaxTextQwen3Config),
    )
