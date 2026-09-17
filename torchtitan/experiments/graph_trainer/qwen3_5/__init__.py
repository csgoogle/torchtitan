# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.experiments.graph_trainer.graph_pp.pipeline import graph_pipeline_llm
from torchtitan.experiments.tpu.qwen3_5 import qwen3_5_configs as tpu_qwen3_5_configs
from torchtitan.models.qwen3_5 import qwen3_5_configs
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from ..common_utils import build_decoder_config_for_backend
from .model import GraphTrainerQwen35Model
from .parallelize import parallelize_qwen3_5


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    moe_comm_backend: str | None = None,
) -> ModelSpec:
    kwargs = {}
    if moe_comm_backend is not None:
        kwargs["moe_comm_backend"] = moe_comm_backend
    config_builder = (
        tpu_qwen3_5_configs[flavor]
        if flavor in tpu_qwen3_5_configs
        else qwen3_5_configs[flavor]
    )
    base = build_decoder_config_for_backend(
        config_builder, attn_backend, **kwargs
    )
    config = GraphTrainerQwen35Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    return ModelSpec(
        name="graph_trainer/qwen3_5",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=graph_pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen35StateDictAdapter,
    )
