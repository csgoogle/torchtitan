"""Qwen3 on the TorchTPU lane.

Thin wrapper over ``torchtitan.models.qwen3``: the same model configs, with
every flex inner attention swapped for the TPU splash kernel
(``SplashAttentionConverter``), MaxText FLOP accounting
(``MaxTextQwen3Config``) and the TPU parallelizer installed.
"""

from torchtitan.experiments.tpu.attention import SplashAttentionConverter
import torchtitan.models.common as common_models
import torchtitan.models.qwen3 as qwen3_models
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .infra.parallelize import parallelize_qwen3
from .model import MaxTextQwen3Config

__all__ = [
    "MaxTextQwen3Config",
    "parallelize_qwen3",
    "qwen3_configs",
    "qwen3_debugmodel",
    "qwen3_debugmodel_moe",
    "model_registry",
    "tpu_converters",
]


def tpu_converters(
    converters: list[ModelConfigConverter.Config] | None = None,
) -> list[ModelConfigConverter.Config]:
  """The splash converter first, then whatever the recipe asked for."""
  return [SplashAttentionConverter.Config(), *(converters or [])]


def _with_splash(config: Qwen3Model.Config) -> MaxTextQwen3Config:
  config = SplashAttentionConverter.Config().build().convert(config)
  return MaxTextQwen3Config.from_qwen3(config)


def _testmodel(seq_len: int = 128) -> MaxTextQwen3Config:
  dim = 128
  head_dim = 16
  n_layers = 3
  vocab_size = 2048
  return _with_splash(
      MaxTextQwen3Config(
          vocab_size=vocab_size,
          dim=dim,
          norm=qwen3_models._qwen3_norm(dim),
          enable_weight_tying=True,
          tok_embeddings=common_models.Embedding.Config(
              num_embeddings=vocab_size,
              embedding_dim=dim,
              param_init=qwen3_models._EMBEDDING_SKIP_INIT,
          ),
          lm_head=common_models.Linear.Config(
              in_features=dim,
              out_features=vocab_size,
              param_init=qwen3_models._LINEAR_INIT,
          ),
          # pytype: disable=wrong-keyword-args
          layers=qwen3_models._build_qwen3_layers(
              n_layers=n_layers,
              dim=dim,
              n_heads=8,
              n_kv_heads=8,
              head_dim=head_dim,
              hidden_dim=256,
              attn_backend="flex",
              rope=common_models.CosSinRoPE.Config(
                  dim=head_dim,
                  max_context_length=seq_len,
                  theta=1000000.0,
              ),
          ),
          # pytype: enable=wrong-keyword-args
      )
  )


def _testmodel_moe(
    seq_len: int = 128,
    moe_comm_backend: str = "standard",
) -> MaxTextQwen3Config:
  dim = 128
  head_dim = 16
  n_layers = 3
  vocab_size = 2048
  return _with_splash(
      MaxTextQwen3Config(
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
          layers=qwen3_models._build_qwen3_moe_layers(
              n_layers=n_layers,
              dim=dim,
              n_heads=8,
              n_kv_heads=8,
              head_dim=head_dim,
              moe_hidden_dim=128,
              num_experts=8,
              top_k=2,
              attn_backend="flex",
              moe_comm_backend=moe_comm_backend,
              rope=common_models.CosSinRoPE.Config(
                  dim=head_dim,
                  max_context_length=seq_len,
                  theta=1000000.0,
              ),
          ),
          # pytype: enable=wrong-keyword-args
      )
  )


_8B_MAX_CONTEXT_LENGTH = 40960


def _8b_tpu(attn_backend: str, *, seq_len: int) -> Qwen3Model.Config:
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
              max_context_length=seq_len,
              theta=1000000.0,
          ),
      ),
      # pytype: enable=wrong-keyword-args
  )


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
  """Upstream's qwen3 registry with splash attention, MaxText FLOPs and the TPU
  parallelizer.

  ``attn_backend`` names the upstream inner attention the converter replaces;
  it never survives to the built model.
  """
  converters = tpu_converters(converters)
  if flavor == "8B":
    context_len = seq_len or _8B_MAX_CONTEXT_LENGTH
    if context_len > _8B_MAX_CONTEXT_LENGTH:
      raise ValueError(
          f"Requested seq_len {context_len} exceeds max context length "
          f"{_8B_MAX_CONTEXT_LENGTH} for flavor {flavor}"
      )
    config = _8b_tpu(attn_backend, seq_len=context_len)
    qwen3_models.validate_converter_order(converters)
    for c in converters:
      config = c.build().convert(config)
    return ModelSpec(
        name="qwen3",
        flavor=flavor,
        model=MaxTextQwen3Config.from_qwen3(config),
        max_context_length=context_len,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=qwen3_models.pipeline_llm,
        post_optimizer_build_fn=qwen3_models.register_moe_load_balancing_hook,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
  # Reuse the upstream model specification and change only its FLOP accounting.
  model_spec = qwen3_models.model_registry(
      flavor,
      seq_len=seq_len,
      attn_backend=attn_backend,
      moe_comm_backend=moe_comm_backend,
      converters=converters,
  )
  assert isinstance(model_spec.model, Qwen3Model.Config)
  model_spec.model = MaxTextQwen3Config.from_qwen3(model_spec.model)
  model_spec.parallelize_fn = parallelize_qwen3
  return model_spec


# Ready-built model configs for the unit tests (``testmodel*`` are tiny; the
# named flavors are upstream's at their default context length).
qwen3_configs = {
    "testmodel": _testmodel(),
    "testmodel_moe": _testmodel_moe(),
    "debugmodel": model_registry("debugmodel", seq_len=2048).model,
    "debugmodel_moe": model_registry("debugmodel_moe", seq_len=2048).model,
    "0.6B": model_registry("0.6B").model,
    "1.7B": model_registry("1.7B").model,
    "4B": model_registry("4B").model,
    "8B": model_registry("8B").model,
    "32B": model_registry("32B").model,
    "30B-A3B": model_registry("30B-A3B").model,
}


def qwen3_debugmodel():
  return qwen3_configs["debugmodel"]


def qwen3_debugmodel_moe():
  return qwen3_configs["debugmodel_moe"]
