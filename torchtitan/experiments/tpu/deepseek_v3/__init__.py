# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan.components.loss as loss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer, default_adamw
from torchtitan.components.validate import Validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.hf_datasets.text_datasets import DATASETS
import torchtitan.models.common as common_models
import torchtitan.models.deepseek_v3 as deepseek_v3_models
from torchtitan.models.deepseek_v3.model import DeepSeekV3Model
from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

__all__ = [
    "parallelize_deepseekv3",
    "deepseekv3_configs",
]


# TODO(tpu): swap in the splash converter -- "sdpa" no longer exists upstream
# (only flex / flex_flash / varlen); TPU attention will be selected through a
# ModelConfigConverter passed as ``converters=`` to ``model_registry``.
def _testmodel(
    attn_backend: str = "flex",
) -> deepseek_v3_models.DeepSeekV3Model.Config:
  dim = 128
  head_dim = 16
  n_layers = 3
  vocab_size = 2048
  # pytype: disable=wrong-keyword-args
  layers = deepseek_v3_models._build_dsv3_layers(
      n_layers=n_layers,
      n_dense_layers=n_layers,
      dim=dim,
      n_heads=8,
      q_lora_rank=0,
      kv_lora_rank=512,
      qk_nope_head_dim=128,
      qk_rope_head_dim=head_dim,
      v_head_dim=128,
      mscale=0.70,
      dense_hidden_dim=256,
      moe_hidden_dim=128,
      num_experts=8,
      num_shared_experts=2,
      router_top_k=2,
      router_score_func="softmax",
      attn_backend=attn_backend,
      moe_comm_backend="standard",
      non_blocking_capacity_factor=None,
      rope=common_models.ComplexRoPE.Config(
          dim=head_dim,
          max_context_length=128,
          theta=10000.0,
          scaling="yarn",
          rope_factor=40.0,
          beta_fast=32.0,
          beta_slow=1.0,
          original_seq_len=128,
      ),
  )
  # pytype: enable=wrong-keyword-args
  return deepseek_v3_models.DeepSeekV3Model.Config(
      vocab_size=vocab_size,
      dim=dim,
      tok_embeddings=common_models.Embedding.Config(
          num_embeddings=vocab_size,
          embedding_dim=dim,
          param_init=deepseek_v3_models._EMBEDDING_INIT,
      ),
      norm=common_models.RMSNorm.Config(
          normalized_shape=dim, param_init=deepseek_v3_models._NORM_INIT
      ),
      lm_head=common_models.Linear.Config(
          in_features=dim,
          out_features=vocab_size,
          param_init=deepseek_v3_models._output_linear_init(dim),
      ),
      layers=layers,
  )


def _testmodel_moe(
    attn_backend: str = "flex",
) -> deepseek_v3_models.DeepSeekV3Model.Config:
  dim = 128
  head_dim = 16
  n_layers = 3
  vocab_size = 2048
  # pytype: disable=wrong-keyword-args
  layers = deepseek_v3_models._build_dsv3_layers(
      n_layers=n_layers,
      n_dense_layers=1,
      dim=dim,
      n_heads=8,
      q_lora_rank=0,
      kv_lora_rank=512,
      qk_nope_head_dim=128,
      qk_rope_head_dim=head_dim,
      v_head_dim=128,
      mscale=0.70,
      dense_hidden_dim=256,
      moe_hidden_dim=128,
      num_experts=8,
      num_shared_experts=2,
      router_top_k=2,
      router_score_func="softmax",
      attn_backend=attn_backend,
      moe_comm_backend="standard",
      non_blocking_capacity_factor=None,
      rope=common_models.ComplexRoPE.Config(
          dim=head_dim,
          max_context_length=128,
          theta=10000.0,
          scaling="yarn",
          rope_factor=40.0,
          beta_fast=32.0,
          beta_slow=1.0,
          original_seq_len=128,
      ),
  )
  # pytype: enable=wrong-keyword-args
  return deepseek_v3_models.DeepSeekV3Model.Config(
      vocab_size=vocab_size,
      dim=dim,
      tok_embeddings=common_models.Embedding.Config(
          num_embeddings=vocab_size,
          embedding_dim=dim,
          param_init=deepseek_v3_models._EMBEDDING_INIT,
      ),
      norm=common_models.RMSNorm.Config(
          normalized_shape=dim, param_init=deepseek_v3_models._NORM_INIT
      ),
      lm_head=common_models.Linear.Config(
          in_features=dim,
          out_features=vocab_size,
          param_init=deepseek_v3_models._output_linear_init(dim),
      ),
      layers=layers,
  )


def _native(flavor: str, attn_backend: str = "flex") -> deepseek_v3_models.DeepSeekV3Model.Config:
  """Upstream flavor at its maximum context length (upstream now stores
  ``(builder, max_context_length)`` tuples and requires ``seq_len``)."""
  get_config, max_context_len = deepseek_v3_models.deepseekv3_configs[flavor]
  return get_config(
      attn_backend=attn_backend,
      moe_comm_backend="standard",
      seq_len=max_context_len,
  )


deepseekv3_configs = {
    "testmodel": _testmodel(),
    "testmodel_moe": _testmodel_moe(),
    "debugmodel": _native("debugmodel"),
    "debugmodel_flex_attn": _native("debugmodel"),
    "16B": _native("16B"),
    "236B": _native("236B"),
    "671B": _native("671B"),
}


# pytype: disable=wrong-arg-types
register_train_spec(
    name="deepseek_v3",
    train_spec=TrainSpec(
        model_cls=DeepSeekV3Model,
        model_args=deepseekv3_configs,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        loss_config=loss.ChunkedLossWrapper.Config(),
        optimizer_config=default_adamw(lr=8e-4),
        lr_scheduler_config=LRSchedulersContainer.Config(warmup_steps=2),
        dataloader_config=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            shuffle=False,
        ),
        validator_config=Validator.Config(enable=True),
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    ),
)
# pytype: enable=wrong-arg-types
