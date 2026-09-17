import torchtitan.models.common as common_models
import torchtitan.models.llama3 as llama3_models

from .infra.parallelize import parallelize_llama

__all__ = [
    "parallelize_llama",
    "llama3_configs",
]


def _testmodel(attn_backend: str = "flex") -> llama3_models.Llama3Model.Config:
  dim = 64
  n_heads = 8
  n_layers = 1
  return llama3_models.Llama3Model.Config(
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
          attn_backend=attn_backend,
          rope=common_models.ComplexRoPE.Config(
              dim=dim // n_heads,
              max_seq_len=512,
          ),
      ),
  )


llama3_configs = {
    "testmodel": _testmodel("sdpa"),
    "debugmodel": llama3_models.llama3_configs["debugmodel"]("sdpa"),
    "debugmodel_flex_attn": llama3_models.llama3_configs["debugmodel"]("flex"),
    "1B": llama3_models.llama3_configs["1B"]("sdpa"),
    "8B": llama3_models.llama3_configs["8B"]("sdpa"),
    "70B": llama3_models.llama3_configs["70B"]("sdpa"),
    "405B": llama3_models.llama3_configs["405B"]("sdpa"),
}
