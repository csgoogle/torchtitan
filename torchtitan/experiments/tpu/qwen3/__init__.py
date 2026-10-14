import torchtitan.models.common as common_models
import torchtitan.models.qwen3 as qwen3_models

from .infra.parallelize import parallelize_qwen3

from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

__all__ = [
    "parallelize_qwen3",
    "qwen3_configs",
    "model_registry",
    "TESTMODEL_MAX_CONTEXT_LENGTH",
]

# Context lengths of the flavors this module defines itself, named so the
# `ModelSpec.max_context_length` that upstream d771abb0 made required can reuse
# the exact value each flavor's RoPE config is built with.
# `_testmodel_moe` matches upstream's `("testmodel", 128)` context length;
# `_8b_tpu` matches upstream's `("8B", 40960)` entry.
TESTMODEL_MAX_CONTEXT_LENGTH = 128
_8B_MAX_CONTEXT_LENGTH = 40960


def _testmodel_moe(
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
) -> Qwen3Model.Config:
    dim = 128
    head_dim = 16
    n_layers = 3
    vocab_size = 2048
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=qwen3_models._qwen3_norm(dim),
        tok_embeddings=common_models.Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=qwen3_models._EMBEDDING_INIT
        ),
        lm_head=common_models.Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=qwen3_models._LINEAR_INIT,
        ),

        # pytype: disable=wrong-keyword-args
        layers=qwen3_models._build_qwen3_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=8,
            n_kv_heads=8,
            head_dim=head_dim,
            moe_hidden_dim=128,
            num_experts=8,
            top_k=2,
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
            rope=common_models.CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=TESTMODEL_MAX_CONTEXT_LENGTH,
                theta=1000000.0,
            ),
        ),
        # pytype: enable=wrong-keyword-args
    )


def _8b_tpu(attn_backend: str = "sdpa") -> Qwen3Model.Config:
    """TPU-specific Qwen3 8B model configuration.

    Differs from the baseline models/qwen3/_8b config by setting fuse_qkv=False
    to avoid running attention projection GEMMs as a single fused block on TPU.
    """
    dim = 4096
    head_dim = 128
    n_layers = 36
    vocab_size = 151936
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=qwen3_models._qwen3_norm(dim),
        tok_embeddings=common_models.Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=qwen3_models._EMBEDDING_INIT,
        ),
        lm_head=common_models.Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=qwen3_models._LINEAR_INIT,
        ),
        # pytype: disable=wrong-keyword-args
        layers=qwen3_models._build_qwen3_layers(
            fuse_qkv=False,
            n_layers=n_layers,
            dim=dim,
            n_heads=32,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=12288,
            attn_backend=attn_backend,
            rope=common_models.CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=_8B_MAX_CONTEXT_LENGTH,
                theta=1000000.0,
            ),
        ),
        # pytype: enable=wrong-keyword-args
    )


def _upstream_flavor(flavor: str, attn_backend: str):
  """Builds an upstream qwen3 flavor at that flavor's full context length.

  Upstream d771abb0 changed `qwen3_models.qwen3_configs[flavor]` from a
  `(attn_backend) -> Config` callable into a `(builder, max_context_length)`
  pair, and made `seq_len` a required keyword-only argument of the builder.
  Passing the flavor's own `max_context_length` reproduces what the old
  callable produced.
  """
  build_config, max_context_length = qwen3_models.qwen3_configs[flavor]
  return build_config(attn_backend, seq_len=max_context_length)


# Model configs by flavor.
qwen3_configs = {
    "testmodel": _upstream_flavor("testmodel", "sdpa"),
    "testmodel_moe": _testmodel_moe("sdpa"),
    "debugmodel": _upstream_flavor("debugmodel", "sdpa"),
    "debugmodel_moe": _upstream_flavor("debugmodel_moe", "sdpa"),
    "0.6B": _upstream_flavor("0.6B", "sdpa"),
    "1.7B": _upstream_flavor("1.7B", "sdpa"),
    "4B": _upstream_flavor("4B", "sdpa"),
    "8B": _8b_tpu("sdpa"),
    "32B": _upstream_flavor("32B", "sdpa"),
    "30B-A3B": _upstream_flavor("30B-A3B", "sdpa"),
}


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
    converters: list = None,
) -> ModelSpec:
  if flavor == "8B" and attn_backend == "sdpa":
    config = _8b_tpu(attn_backend="sdpa")
    # Since we return a custom ModelSpec for 8B, apply converters here.
    if converters is not None:
      qwen3_models.validate_converter_order(converters)
      for c in converters:
        config = c.build().convert(config)
    # Upstream d771abb0 made ModelSpec.max_context_length required.
    return ModelSpec(
        name="qwen3",
        flavor=flavor,
        model=config,
        max_context_length=_8B_MAX_CONTEXT_LENGTH,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=qwen3_models.pipeline_llm,
        post_optimizer_build_fn=qwen3_models.register_moe_load_balancing_hook,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
  return qwen3_models.model_registry(
      flavor,
      attn_backend=attn_backend,
      moe_comm_backend=moe_comm_backend,
      converters=converters,
  )

from .config_registry import *
