# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass

import torch.nn as nn

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.experiments.graph_trainer.qwen3 import (
    model_registry as graph_trainer_model_registry,
)
from torchtitan.experiments.graph_trainer.qwen3.parallelize import (
    parallelize_qwen3 as graph_trainer_parallelize_qwen3,
)
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.tpu.grouped_experts import (
    configure_expert_dweight_output_dtype,
)
from torchtitan.experiments.tpu.mfu_flops import (
    get_maxtext_decoder_nparams_and_flops,
)
from torchtitan.protocols.model_spec import ModelSpec


class TpuGraphTrainerQwen3Model(GraphTrainerQwen3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(GraphTrainerQwen3Model.Config):
        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_maxtext_decoder_nparams_and_flops(self, model, seq_len)


def parallelize_qwen3(
    model: GraphTrainerQwen3Model,
    *,
    training: TrainingConfig,
    **kwargs,
) -> nn.Module:
    """Match unpaired expert dWeight stores to FSDP before setup."""
    configure_expert_dweight_output_dtype(
        model, TORCH_DTYPE_MAP[training.mixed_precision_reduce]
    )
    return graph_trainer_parallelize_qwen3(model, training=training, **kwargs)


def model_registry(flavor: str) -> ModelSpec:
    model_spec = graph_trainer_model_registry(flavor)
    model_config = TpuGraphTrainerQwen3Model.Config(
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
