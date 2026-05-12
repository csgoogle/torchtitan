# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
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


def _testmodel(
    attn_backend: str = "sdpa",
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
      rope=common_models.RoPE.Config(
          dim=head_dim,
          max_seq_len=128,
          theta=10000.0,
          backend="complex",
          scaling="yarn",
          rope_factor=40.0,
          beta_fast=32.0,
          beta_slow=1.0,
          original_seq_len=128,
      ),
      layers=layers,
  )


def _testmodel_moe(
    attn_backend: str = "sdpa",
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
      rope=common_models.RoPE.Config(
          dim=head_dim,
          max_seq_len=128,
          theta=10000.0,
          backend="complex",
          scaling="yarn",
          rope_factor=40.0,
          beta_fast=32.0,
          beta_slow=1.0,
          original_seq_len=128,
      ),
      layers=layers,
  )


deepseekv3_configs = {
    "testmodel": _testmodel(),
    "testmodel_moe": _testmodel_moe(),
    "debugmodel": deepseek_v3_models.deepseekv3_configs["debugmodel"](
        attn_backend="sdpa", moe_comm_backend="standard"
    ),
    "debugmodel_flex_attn": deepseek_v3_models.deepseekv3_configs["debugmodel"](
        attn_backend="flex", moe_comm_backend="standard"
    ),
    "16B": deepseek_v3_models.deepseekv3_configs["16B"](
        attn_backend="flex", moe_comm_backend="standard"
    ),
    "236B": deepseek_v3_models.deepseekv3_configs["236B"](
        attn_backend="flex", moe_comm_backend="standard"
    ),
    "671B": deepseek_v3_models.deepseekv3_configs["671B"](
        attn_backend="flex", moe_comm_backend="standard"
    ),
}


# pytype: disable=wrong-arg-types
register_train_spec(
    name="deepseek_v3",
    train_spec=TrainSpec(
        model_cls=DeepSeekV3Model,
        model_args=deepseekv3_configs,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        loss_config=ChunkedCELoss.Config(),
        optimizer_config=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler_config=LRSchedulersContainer.Config(warmup_steps=2),
        dataloader_config=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        validator_config=Validator.Config(enable=True),
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    ),
)
# pytype: enable=wrong-arg-types
