"""OLMo 2 on TPU, ported from HF transformers (``Olmo2ForCausalLM``).

Flavors are ``(builder, max_context_length)`` pairs, the upstream qwen3 /
llama3 registry form. ``1B`` is ``allenai/OLMo-2-0425-1B``; its constants come
from that checkpoint's ``config.json`` (transformers 4.50.3 export):
hidden_size 2048, num_hidden_layers 16, num_attention_heads 16,
num_key_value_heads 16 (head_dim 128), intermediate_size 8192,
vocab_size 100352, rope_theta 500000, rms_norm_eps 1e-6,
max_position_embeddings 4096, tie_word_embeddings false, hidden_act silu,
attention_bias false.

Both trainers reuse the llama3 parallelization: ``model_registry`` uses the TPU
llama3 ``parallelize_llama`` (AC -> per-block compile -> SimpleFSDP or FSDP2)
and ``graph_trainer_model_registry`` upstream GraphTrainer's (annotate FQNs ->
SimpleFSDP -> GraphTrainer compile). Unsupported parallelisms are rejected in
``Olmo2Model.Config.update_from_config``, before either runs.

Every layer attends with ``SlidingWindowScaledDotProductInnerAttention``
(window None): causal SDPA restricted to the query's packed document, the
same masking Splash applies from ``segment_ids``. Stock
``ScaledDotProductInnerAttention`` ignores ``positions`` and would attend
across documents.
"""

from collections.abc import Callable
import dataclasses
from functools import partial

from torch import nn
from torchtitan.experiments.graph_trainer.llama3.parallelize import parallelize_llama as graph_trainer_parallelize_llama
from torchtitan.experiments.tpu.llama3.infra.parallelize import parallelize_llama
from torchtitan.experiments.tpu.muse_glimmer.attention import SlidingWindowScaledDotProductInnerAttention
from torchtitan.models.common import CosSinRoPE
from torchtitan.models.common import Embedding
from torchtitan.models.common import Linear
from torchtitan.models.common.config_utils import make_ffn_config
from torchtitan.models.common.config_utils import make_gqa_config
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import FlatHeadsRMSNorm
from .model import Olmo2Attention
from .model import Olmo2Model
from .model import Olmo2TransformerBlock
from .state_dict_adapter import Olmo2StateDictAdapter

__all__ = [
    "FlatHeadsRMSNorm",
    "Olmo2Attention",
    "Olmo2Model",
    "Olmo2StateDictAdapter",
    "Olmo2TransformerBlock",
    "build_olmo2_config",
    "graph_trainer_model_registry",
    "model_registry",
    "olmo2_configs",
]

# HF Olmo2PreTrainedModel initializes every Linear and Embedding with
# normal(0, initializer_range=0.02) and RMSNorm weights with ones.
_INIT_STD = 0.02
_LINEAR_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.normal_, std=_INIT_STD)
}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=_INIT_STD)}
_NORM_INIT = {"weight": nn.init.ones_}
_EPS = 1e-6
_ROPE_THETA = 500000.0


def _norm(dim: int) -> RMSNorm.Config:
  return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _flat_heads_norm(width: int) -> FlatHeadsRMSNorm.Config:
  return FlatHeadsRMSNorm.Config(
      normalized_shape=width, eps=_EPS, param_init=_NORM_INIT
  )


def build_olmo2_config(
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    vocab_size: int,
    seq_len: int,
) -> Olmo2Model.Config:
  """Builds an OLMo 2 model config from HF-config-level hyperparameters."""
  layers = []
  for _ in range(n_layers):
    gqa = make_gqa_config(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        wqkv_param_init=_LINEAR_INIT,
        wo_param_init=_LINEAR_INIT,
        inner_attention=SlidingWindowScaledDotProductInnerAttention.Config(),
        rope=CosSinRoPE.Config(
            dim=head_dim, max_context_length=seq_len, theta=_ROPE_THETA
        ),
    )
    attention = Olmo2Attention.Config(
        **{
            field.name: getattr(gqa, field.name)
            for field in dataclasses.fields(gqa)
            if field.init and field.name != "qk_norm"
        },
        q_norm=_flat_heads_norm(n_heads * head_dim),
        k_norm=_flat_heads_norm(n_kv_heads * head_dim),
    )
    layers.append(
        Olmo2TransformerBlock.Config(
            attention=attention,
            attention_norm=_norm(dim),
            ffn_norm=_norm(dim),
            feed_forward=make_ffn_config(
                dim=dim,
                hidden_dim=hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_LINEAR_INIT,
            ),
        )
    )
  return Olmo2Model.Config(
      dim=dim,
      vocab_size=vocab_size,
      tok_embeddings=Embedding.Config(
          num_embeddings=vocab_size,
          embedding_dim=dim,
          param_init=_EMBEDDING_INIT,
      ),
      norm=_norm(dim),
      lm_head=Linear.Config(
          in_features=dim, out_features=vocab_size, param_init=_LINEAR_INIT
      ),
      layers=layers,
  )


def _debugmodel(*, seq_len: int) -> Olmo2Model.Config:
  # head_dim 128, as in the 1B, so the later Splash rung runs on this flavor.
  return build_olmo2_config(
      dim=512,
      n_layers=8,
      n_heads=4,
      n_kv_heads=4,
      head_dim=128,
      hidden_dim=2048,
      vocab_size=2048,
      seq_len=seq_len,
  )


def _1b(*, seq_len: int) -> Olmo2Model.Config:
  return build_olmo2_config(
      dim=2048,
      n_layers=16,
      n_heads=16,
      n_kv_heads=16,
      head_dim=128,
      hidden_dim=8192,
      vocab_size=100352,
      seq_len=seq_len,
  )


olmo2_configs: dict[str, tuple[Callable[..., Olmo2Model.Config], int]] = {
    "debugmodel": (_debugmodel, 2048),
    "1B": (_1b, 4096),
}


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
  """Returns the TPU ``ModelSpec`` for an OLMo 2 flavor.

  Args:
    flavor: A key of ``olmo2_configs``.
    seq_len: Context length; defaults to the flavor's maximum.
    converters: Optional model-config converters.

  Returns:
    The model spec, using the TPU llama3 ``parallelize_llama`` and
    ``Olmo2StateDictAdapter``.
  """
  build_config, max_context_length = olmo2_configs[flavor]
  context_length = seq_len or max_context_length
  if context_length > max_context_length:
    raise ValueError(
        f"Requested seq_len {context_length} exceeds max context length "
        f"{max_context_length} for flavor {flavor}"
    )
  config = build_config(seq_len=context_length)
  if converters is not None:
    validate_converter_order(converters)
    for converter in converters:
      config = converter.build().convert(config)
  return ModelSpec(
      name="olmo2",
      flavor=flavor,
      model=config,
      max_context_length=context_length,
      parallelize_fn=parallelize_llama,
      # PP is not supported on TPU yet; update_from_config rejects it.
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=Olmo2StateDictAdapter,
  )


def graph_trainer_model_registry(flavor: str) -> ModelSpec:
  """The GraphTrainer spec (``--tpu_config.use_graph_trainer``) for a flavor.

  ``train.py`` routes ``olmo2`` here. ``to_graph_trainer_config`` takes only the
  model class, ``parallelize_fn`` and ``pipelining_fn`` from it, so the model
  config stays the recipe's own and only upstream GraphTrainer's llama3
  parallelization is swapped in.

  Args:
    flavor: A key of ``olmo2_configs``.

  Returns:
    ``model_registry(flavor)`` with GraphTrainer's ``parallelize_llama``.
  """
  return dataclasses.replace(
      model_registry(flavor),
      parallelize_fn=graph_trainer_parallelize_llama,
      pipelining_fn=None,
  )


from .config_registry import *  # pylint: disable=wildcard-import,g-import-not-at-top,g-bad-import-order
