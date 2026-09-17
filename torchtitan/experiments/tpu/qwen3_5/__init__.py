# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU overlay for Qwen3.5.

Mirrors `torchtitan.experiments.tpu.qwen3`: the model itself comes from
`torchtitan.models.qwen3_5`, and this package only supplies the flavors,
trainer configs and parallelization wrapper that TPU needs.

Two deviations from the upstream flavors are baked in here rather than left to
callers, because neither is optional on TPU:

  * `fla_backend="tpu_chunked"`. The `fla_chunked` / `fla_fused_recurrent`
    defaults call into `flash-linear-attention`, a Triton/CUDA package that is
    neither vendored in internal nor runnable on TPU, and the `torch_native`
    fallback steps the recurrence one token at a time in Python, which is
    intractable to trace at training sequence lengths. `tpu_chunked` is the
    same chunkwise-parallel algorithm written in plain torch ops; see
    `torchtitan.experiments.tpu.gated_delta`.
  * `vision_encoder=None`. Qwen3.5 is a VLM upstream, but the TPU training
    configs here feed it a text corpus, so the ViT tower would be ~27 layers of
    parameters that never see a gradient.
"""

import dataclasses
from typing import Any

import torch
from torchtitan.experiments.graph_trainer import simple_fsdp
from torchtitan.experiments.tpu.qwen3_5.infra import parallelize as tpu_parallelize
from torchtitan.models import common as common_models
import torchtitan.models.qwen3_5 as qwen3_5_models
from torchtitan.models.qwen3_5 import model as qwen35_model
from torchtitan.models.qwen3_5 import rope as qwen35_rope
from torchtitan.models.qwen3_5 import state_dict_adapter as qwen35_adapter
from torchtitan.protocols import model_spec as model_spec_lib

parallelize_qwen3_5 = tpu_parallelize.parallelize_qwen3_5

__all__ = [
    "TpuQwen35Model",
    "parallelize_qwen3_5",
    "qwen3_5_configs",
    "model_registry",
    "qwen35_moe_35b",
    "qwen35_moe_35b_proxy",
    "qwen35_moe_35b_fsdp_only_full_ac_benchmark",
]

# Depth of the real Qwen3.5-35B-A3B. Layer slices are fractions of this.
QWEN35_35B_A3B_NUM_LAYERS = 40

# Every `full_attention_interval`-th layer is full attention; the rest are
# GatedDeltaNet. Keeping a layer count that is a multiple of this preserves the
# real model's 25/75 attention mix in a truncated stack.
_FULL_ATTENTION_INTERVAL = 4


class TpuQwen35Model(qwen35_model.Qwen35Model):
  """Qwen35Model subclass that disables SimpleFSDP parametrization during init_states."""

  @dataclasses.dataclass(kw_only=True, slots=True)
  class Config(qwen35_model.Qwen35Model.Config):
    pass

  def init_states(
      self,
      *,
      buffer_device: torch.device | None = None,
  ) -> None:
    with simple_fsdp.disable_active_parametrization():
      super().init_states(buffer_device=buffer_device)


def _tpu_35b_a3b(
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
    *,
    n_layers: int = QWEN35_35B_A3B_NUM_LAYERS,
) -> TpuQwen35Model.Config:
  """Text-only Qwen3.5-35B-A3B, optionally truncated in depth.

  Hyperparameters are copied verbatim from `models.qwen3_5._35b_a3b`: dim 2048,
  head_dim 256 with rotary_dim 64, 16 query heads over 2 KV heads, 256 routed
  experts of hidden dim 512 with top-k 8 plus one shared expert, and a 248320
  token vocabulary. Only `n_layers`, the vision tower and the linear-attention
  backend differ.

  Args:
    attn_backend: Backend for the full-attention layers. TPU uses `sdpa`.
    moe_comm_backend: MoE communication backend.
    n_layers: Layers to build, out of the real model's 40. Multiples of 4 keep
      the full-attention / GatedDeltaNet ratio intact.

  Returns:
    A text-only `TpuQwen35Model.Config`.

  Raises:
    ValueError: If `n_layers` is outside `(0, 40]`.
  """
  if not 0 < n_layers <= QWEN35_35B_A3B_NUM_LAYERS:
    raise ValueError(
        f"n_layers must be in (0, {QWEN35_35B_A3B_NUM_LAYERS}], got {n_layers}"
    )

  dim = 2048
  head_dim = 256
  rotary_dim = 64
  vocab_size = 248320
  return TpuQwen35Model.Config(
      vocab_size=vocab_size,
      dim=dim,
      # pyrefly: ignore [bad-argument-type]
      norm=qwen3_5_models._offset_norm(dim),  # pylint: disable=protected-access
      tok_embeddings=common_models.Embedding.Config(
          num_embeddings=vocab_size,
          embedding_dim=dim,
          param_init=qwen3_5_models._EMBEDDING_INIT,  # pylint: disable=protected-access
      ),
      lm_head=common_models.Linear.Config(
          in_features=dim,
          out_features=vocab_size,
          param_init=qwen3_5_models._output_linear_init(dim),  # pylint: disable=protected-access
      ),
      # pytype: disable=wrong-keyword-args
      layers=qwen3_5_models._build_qwen35_moe_layers(  # pylint: disable=protected-access
          rope=qwen35_rope.MRoPE.Config(
              dim=rotary_dim,
              max_seq_len=262144,
              theta=10_000_000.0,
              mrope_section=[11, 11, 10],
          ),
          attn_backend=attn_backend,
          n_layers=n_layers,
          dim=dim,
          n_heads=16,
          n_kv_heads=2,
          head_dim=head_dim,
          rotary_dim=rotary_dim,
          moe_hidden_dim=512,
          num_experts=256,
          top_k=8,
          shared_expert_hidden_dim=512,
          n_key_heads=16,
          n_value_heads=32,
          key_head_dim=128,
          value_head_dim=128,
          full_attention_interval=_FULL_ATTENTION_INTERVAL,
          fla_backend="tpu_chunked",
          moe_comm_backend=moe_comm_backend,
      ),
      # pytype: enable=wrong-keyword-args
      vision_encoder=None,
  )


def _tpu_35b_a3b_4layer(
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
) -> TpuQwen35Model.Config:
  """Qwen3.5-35B-A3B at 4 of 40 layers, for single-host TPU slices."""
  return _tpu_35b_a3b(attn_backend, moe_comm_backend, n_layers=4)


qwen3_5_configs = {
    "35B-A3B": _tpu_35b_a3b,
    "35B-A3B-4layer": _tpu_35b_a3b_4layer,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    moe_comm_backend: str | None = None,
    converters: list[Any] | None = None,
) -> model_spec_lib.ModelSpec:
  """Builds a TPU `ModelSpec` for a Qwen3.5 flavor.

  Flavors defined here are text-only and use the pure-torch linear-attention
  path; anything else falls through to the upstream registry so the full set of
  Qwen3.5 flavors stays reachable.

  Args:
    flavor: Flavor name, from `qwen3_5_configs` or the upstream registry.
    attn_backend: Backend for full-attention layers.
    moe_comm_backend: MoE communication backend.
    converters: Model config converters, applied in order.

  Returns:
    A `ModelSpec` whose `parallelize_fn` is the TPU wrapper.
  """
  if flavor not in qwen3_5_configs:
    model_spec = qwen3_5_models.model_registry(
        flavor,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        converters=converters,
    )
    model_spec.parallelize_fn = parallelize_qwen3_5
    return model_spec

  kwargs = dict(attn_backend=attn_backend)
  if moe_comm_backend is not None:
    kwargs["moe_comm_backend"] = moe_comm_backend
  config = qwen3_5_configs[flavor](**kwargs)
  if converters is not None:
    qwen3_5_models.validate_converter_order(converters)
    for c in converters:
      config = c.build().convert(config)

  return model_spec_lib.ModelSpec(
      name="qwen3_5_tpu",
      flavor=flavor,
      model=config,
      parallelize_fn=parallelize_qwen3_5,
      pipelining_fn=None,
      post_optimizer_build_fn=(qwen3_5_models.register_moe_load_balancing_hook),
      state_dict_adapter=qwen35_adapter.Qwen35StateDictAdapter,
  )


from .config_registry import *  # pylint: disable=wildcard-import,g-import-not-at-top
