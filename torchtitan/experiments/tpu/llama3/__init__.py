from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.components.validate import Validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec
import torchtitan.models.common as common_models
import torchtitan.models.llama3 as llama3_models

from .infra.parallelize import parallelize_llama
from .infra.pipeline import pipeline_llama

__all__ = [
    "parallelize_llama",
    "llama3_configs",
]


def _testmodel(attn_backend: str = "sdpa") -> llama3_models.Llama3Model.Config:
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
      rope=common_models.RoPE.Config(
          dim=dim // n_heads,
          max_seq_len=512,
          backend="complex",
      ),
      layers=llama3_models._build_llama3_layers(
          n_layers=n_layers,
          dim=dim,
          n_heads=n_heads,
          n_kv_heads=n_heads,
          hidden_dim=4 * dim,
          attn_backend=attn_backend,
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

# pytype: disable=wrong-arg-types
register_train_spec(
    name="llama3_tpu",
    train_spec=TrainSpec(
        model_cls=llama3_models.Llama3Model,
        model_args=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        loss_config=ChunkedCELoss.Config(),
        optimizer_config=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler_config=LRSchedulersContainer.Config(warmup_steps=2),
        dataloader_config=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        validator_config=Validator.Config(enable=True),
        state_dict_adapter=llama3_models.Llama3StateDictAdapter,
    )
)
# pytype: enable=wrong-arg-types
