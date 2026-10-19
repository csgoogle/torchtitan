"""Qwen3.5 (text-only) model configurations and registry for TPU.

Wraps upstream ``torchtitan.models.qwen3_5`` for TPU execution by:
1. Installing the ``attn_gym`` stub before importing
   ``torchtitan.models.qwen3_5``.
2. Stripping the multimodal ``vision_encoder`` for text-only pretraining.
3. Replacing ``InnerGatedDeltaNet`` and ``GatedDeltaKernel`` with
   ``TpuInnerGatedDeltaNet`` and ``TpuGatedDeltaKernel`` (chunked gated delta
   rule in PyTorch/XLA or Tokamax's Mosaic kernels, per-sequence causal
   ``conv1d``).
4. Replacing ``Qwen35Attention`` with ``TpuQwen35Attention`` and
   ``PackedScaledDotProductInnerAttention`` so ``positions`` is forwarded to
   the inner attention for document-masked SDPA and Splash / Tokamax attention.
5. Turning off the auxiliary-loss-free expert-bias balancer
   (``load_balance_coeff=None``): HF checkpoints carry no expert biases.
"""

import copy
import dataclasses
from functools import partial
import math

import torch
import torch.nn as nn
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.experiments.tpu.qwen3_5 import gated_delta
from torchtitan.experiments.tpu.qwen3_5.infra.parallelize import parallelize_qwen3_5
from torchtitan.models.common import Embedding, Linear
from torchtitan.models.qwen3_5 import (
    GatedDeltaNet,
    MRoPE,
    OffsetRMSNorm,
    Qwen35Model,
    Qwen35StateDictAdapter,
    Qwen35TransformerBlock,
    RMSNormGated,
    _EMBEDDING_INIT,
    _build_qwen35_layers,
    _build_qwen35_moe_layers,
    _offset_norm,
    _output_linear_init,
    qwen3_5_configs as upstream_qwen3_5_configs,
)
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

__all__ = [
    "DEFAULT_MAX_CONTEXT_LENGTH",
    "GatedDeltaNet",
    "MRoPE",
    "OffsetRMSNorm",
    "Qwen35Model",
    "Qwen35StateDictAdapter",
    "Qwen35TransformerBlock",
    "RMSNormGated",
    "TESTMODEL_MAX_CONTEXT_LENGTH",
    "model_registry",
    "qwen3_5_configs",
]

TESTMODEL_MAX_CONTEXT_LENGTH = 128

_CONV1D_INIT = {
    "weight": partial(nn.init.kaiming_uniform_, a=math.sqrt(5)),
}


def _tpu_a_log_init(param: nn.Parameter) -> None:
  """DTensor-safe initialization for GatedDeltaNet A_log."""
  with torch.no_grad():
    nn.init.uniform_(param, 0.01, 16.0)
    param.log_()


def _to_tpu_text_only_config(
    config: Qwen35Model.Config,
    *,
    seq_len: int | None = None,
) -> gated_delta.TpuQwen35Model.Config:
  """Converts an upstream ``Qwen35Model.Config`` to a TPU text-only config."""
  tpu_config = copy.deepcopy(config)
  tpu_config.vision_encoder = None

  for layer in tpu_config.layers:
    if layer.attention is not None:
      attn = layer.attention
      rope = (
          dataclasses.replace(attn.rope, max_context_length=seq_len)
          if seq_len is not None
          else attn.rope
      )
      layer.attention = gated_delta.TpuQwen35Attention.Config(
          n_heads=attn.n_heads,
          n_kv_heads=attn.n_kv_heads,
          head_dim=attn.head_dim,
          rotary_dim=attn.rotary_dim,
          rope=rope,
          wq=attn.wq,
          wk=attn.wk,
          wv=attn.wv,
          wo=attn.wo,
          q_norm=attn.q_norm,
          k_norm=attn.k_norm,
          inner_attention=gated_delta.PackedScaledDotProductInnerAttention.Config(),
      )
    elif layer.delta_net is not None:
      layer.delta_net.inner_gated_delta_net = (
          gated_delta.TpuInnerGatedDeltaNet.Config(
              kernel=gated_delta.TpuGatedDeltaKernel.Config()
          )
      )
      # Under SimpleFSDP, Conv1d.reset_parameters() accesses self.weight via
      # the SimpleFSDP @property, writing to a temporary tensor and leaving
      # self._parameters["weight"] uninitialized after model.to_empty(). Setting
      # param_init routes initialization through Module._init_self_parameters
      # -> self.named_parameters(recurse=False), which initializes the real
      # parameter in-place.
      layer.delta_net.conv_q.param_init = _CONV1D_INIT
      layer.delta_net.conv_k.param_init = _CONV1D_INIT
      layer.delta_net.conv_v.param_init = _CONV1D_INIT
      layer.delta_net.param_init = {
          "A_log": _tpu_a_log_init,
          "dt_bias": nn.init.ones_,
      }
    if layer.moe is not None:
      # No `expert_bias` buffer: HF checkpoints (and the state-dict adapter)
      # have none.
      layer.moe.load_balance_coeff = None

  return gated_delta.TpuQwen35Model.Config(**{
      field.name: getattr(tpu_config, field.name)
      for field in dataclasses.fields(tpu_config)
      if field.init
  })


_testmodel = _to_tpu_text_only_config(
    Qwen35Model.Config(
        vocab_size=2048,
        dim=128,
        enable_weight_tying=True,
        norm=_offset_norm(128),  # pyrefly: ignore[bad-argument-type]
        tok_embeddings=Embedding.Config(
            num_embeddings=2048,
            embedding_dim=128,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=128,
            out_features=2048,
            param_init=_output_linear_init(128),
        ),
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=8,
                max_context_length=TESTMODEL_MAX_CONTEXT_LENGTH,
                theta=10_000_000.0,
                mrope_section=[2, 1, 1],
            ),
            attn_backend="sdpa",
            n_layers=4,
            dim=128,
            n_heads=4,
            n_kv_heads=2,
            head_dim=32,
            rotary_dim=8,
            hidden_dim=256,
            n_key_heads=2,
            n_value_heads=4,
            key_head_dim=16,
            value_head_dim=16,
        ),
    )
)

_testmodel_moe = _to_tpu_text_only_config(
    Qwen35Model.Config(
        vocab_size=2048,
        dim=128,
        enable_weight_tying=False,
        norm=_offset_norm(128),  # pyrefly: ignore[bad-argument-type]
        tok_embeddings=Embedding.Config(
            num_embeddings=2048,
            embedding_dim=128,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=128,
            out_features=2048,
            param_init=_output_linear_init(128),
        ),
        layers=_build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=8,
                max_context_length=TESTMODEL_MAX_CONTEXT_LENGTH,
                theta=10_000_000.0,
                mrope_section=[2, 1, 1],
            ),
            attn_backend="sdpa",
            n_layers=4,
            dim=128,
            n_heads=4,
            n_kv_heads=2,
            head_dim=32,
            rotary_dim=8,
            moe_hidden_dim=64,
            num_experts=8,
            top_k=2,
            shared_expert_hidden_dim=64,
            n_key_heads=2,
            n_value_heads=4,
            key_head_dim=16,
            value_head_dim=16,
        ),
    )
)

DEFAULT_MAX_CONTEXT_LENGTH: dict[str, int] = {
    "testmodel": TESTMODEL_MAX_CONTEXT_LENGTH,
    "testmodel_moe": TESTMODEL_MAX_CONTEXT_LENGTH,
    **{
        name: max_seq_len
        for name, (_, max_seq_len) in upstream_qwen3_5_configs.items()
    },
    "35B-A3B-proxy": upstream_qwen3_5_configs["35B-A3B"][1],
}

qwen3_5_configs: dict[str, Qwen35Model.Config] = {
    "testmodel": _testmodel,
    "testmodel_moe": _testmodel_moe,
    **{
        name: _to_tpu_text_only_config(
            build_fn("sdpa", seq_len=max_seq_len), seq_len=max_seq_len
        )
        for name, (build_fn, max_seq_len) in upstream_qwen3_5_configs.items()
    },
}

# 4-layer proxy of 35B-A3B (1 full 3:1 GatedDeltaNet:Attention cycle with
# identical hidden dim, head dim, and 256 routed experts) for fast single-host
# TPU v7x testing.
_proxy_35b = copy.deepcopy(qwen3_5_configs["35B-A3B"])
_proxy_35b.layers = _proxy_35b.layers[:4]
qwen3_5_configs["35B-A3B-proxy"] = _proxy_35b


def _split_optimizer_groups_by_mesh(optimizers_container) -> None:
  """Splits every optimizer param group into one group per DeviceMesh.

  With Expert Parallelism the routed experts are DTensors on the
  `(efsdp, ep)` mesh and all other parameters on the `fsdp` mesh; torch_tpu's
  fused and foreach AdamW take one mesh per param group.

  Args:
    optimizers_container: The `OptimizersContainer` to update in place.
  """

  def mesh_key(param: torch.Tensor) -> int | None:
    spec = getattr(param, "_spec", None)  # A DTensor's `DTensorSpec`.
    return None if spec is None else id(spec.mesh)

  for opt in optimizers_container.optimizers:
    new_groups = []
    for group in opt.param_groups:
      names = group.get("param_names")
      options = {
          k: v for k, v in group.items() if k not in ("params", "param_names")
      }
      buckets: dict[int | None, dict[str, list]] = {}
      for i, param in enumerate(group["params"]):
        bucket = buckets.setdefault(
            mesh_key(param), {"params": [], "param_names": []}
        )
        bucket["params"].append(param)
        if names is not None:
          bucket["param_names"].append(names[i])
      for bucket in buckets.values():
        if names is None:
          del bucket["param_names"]
        new_groups.append({**bucket, **options})
    if len(new_groups) > len(opt.param_groups):
      opt.param_groups = []
      for group in new_groups:
        opt.add_param_group(group)


def post_optimizer_build_qwen3_5(
    optimizers,
    model_parts: list[nn.Module],
    parallel_dims,
) -> None:
  """Splits multi-mesh optimizer param groups, then registers the MoE hook."""
  _split_optimizer_groups_by_mesh(optimizers)
  register_moe_load_balancing_hook(optimizers, model_parts, parallel_dims)


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    moe_comm_backend: str | None = None,
    seq_len: int | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
  """Returns a TPU ``ModelSpec`` for the requested Qwen3.5 text-only flavor."""
  if flavor not in qwen3_5_configs:
    raise ValueError(
        f"Unsupported Qwen3.5 flavor {flavor!r}; expected one of "
        f"{sorted(qwen3_5_configs)}"
    )
  if attn_backend != "sdpa":
    raise ValueError(
        "Only attn_backend='sdpa' is supported for Qwen3.5 on TPU, got "
        f"{attn_backend!r}"
    )

  default_max_context = DEFAULT_MAX_CONTEXT_LENGTH[flavor]
  if (
      flavor in ("testmodel", "testmodel_moe")
      and seq_len is not None
      and seq_len > default_max_context
  ):
    raise ValueError(
        f"Requested seq_len={seq_len} exceeds max context length "
        f"{default_max_context} for flavor={flavor!r}."
    )
  effective_seq_len = seq_len if seq_len is not None else default_max_context
  config = _to_tpu_text_only_config(
      qwen3_5_configs[flavor], seq_len=effective_seq_len
  )

  if moe_comm_backend is not None and flavor in upstream_qwen3_5_configs:
    build_fn, _ = upstream_qwen3_5_configs[flavor]
    config = _to_tpu_text_only_config(
        build_fn(
            "sdpa",
            seq_len=effective_seq_len,
            moe_comm_backend=moe_comm_backend,
        ),
        seq_len=effective_seq_len,
    )

  if converters:
    validate_converter_order(converters)
    for c in converters:
      config = c.build().convert(config)

  return ModelSpec(
      name="qwen3_5",
      flavor=flavor,
      model=config,
      max_context_length=effective_seq_len,
      parallelize_fn=parallelize_qwen3_5,
      pipelining_fn=None,
      post_optimizer_build_fn=post_optimizer_build_qwen3_5,
      state_dict_adapter=Qwen35StateDictAdapter,
  )
