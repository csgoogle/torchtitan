"""Llama3 on the TorchTPU lane.

Thin wrapper over ``torchtitan.models.llama3``: the same model configs, with
every flex inner attention swapped for the TPU splash kernel
(``SplashAttentionConverter``) and the TPU parallelizer installed.
"""

from torchtitan.experiments.tpu.attention import SplashAttentionConverter
import torchtitan.models.common as common_models
from torchtitan.models.common.config_utils import TpGemmBackend
import torchtitan.models.llama3 as llama3_models
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .infra.parallelize import parallelize_llama

__all__ = [
    "parallelize_llama",
    "llama3_configs",
    "model_registry",
    "tpu_converters",
]


def tpu_converters(
    converters: list[ModelConfigConverter.Config] | None = None,
) -> list[ModelConfigConverter.Config]:
  """The splash converter first, then whatever the recipe asked for."""
  return [SplashAttentionConverter.Config(), *(converters or [])]


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    tp_gemm_backend: TpGemmBackend = "default",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
  """Upstream's llama3 registry with splash attention and the TPU parallelizer.

  ``attn_backend`` names the upstream inner attention the converter replaces;
  it never survives to the built model.
  """
  model_spec = llama3_models.model_registry(
      flavor,
      seq_len=seq_len,
      attn_backend=attn_backend,
      tp_gemm_backend=tp_gemm_backend,
      converters=tpu_converters(converters),
  )
  model_spec.parallelize_fn = parallelize_llama
  return model_spec


def _testmodel(seq_len: int = 512) -> llama3_models.Llama3Model.Config:
  dim = 64
  n_heads = 8
  n_layers = 1
  config = llama3_models.Llama3Model.Config(
      dim=dim,
      vocab_size=128,
      tok_embeddings=common_models.Embedding.Config(
          num_embeddings=128,
          embedding_dim=dim,
          param_init=llama3_models._EMBEDDING_INIT,
      ),
      norm=common_models.RMSNorm.Config(
          normalized_shape=dim,
          param_init=llama3_models._NORM_INIT,
      ),
      lm_head=common_models.Linear.Config(
          in_features=dim,
          out_features=128,
          param_init=llama3_models._LINEAR_INIT,
      ),
      layers=llama3_models._build_llama3_layers(
          n_layers=n_layers,
          dim=dim,
          n_heads=n_heads,
          n_kv_heads=n_heads,
          hidden_dim=4 * dim,
          attn_backend="flex",
          rope=common_models.ComplexRoPE.Config(
              dim=dim // n_heads,
              max_context_length=seq_len,
          ),
      ),
  )
  return SplashAttentionConverter.Config().build().convert(config)


# Ready-built model configs for the unit tests (``testmodel`` is tiny; the
# named flavors are upstream's at their default context length).
llama3_configs = {
    "testmodel": _testmodel(),
    "debugmodel": model_registry("debugmodel", seq_len=2048).model,
    "1B": model_registry("1B").model,
    "8B": model_registry("8B").model,
    "70B": model_registry("70B").model,
    "405B": model_registry("405B").model,
}
