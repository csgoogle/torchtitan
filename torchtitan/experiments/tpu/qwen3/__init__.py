import torchtitan.components.loss as loss
from torchtitan.components.optimizer import OptimizersContainer, default_adamw
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.components.validate import Validator
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec
import torchtitan.models.common as common_models
import torchtitan.models.qwen3 as qwen3_models

from .infra.parallelize import parallelize_qwen3

from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .model import MaxTextQwen3Config

__all__ = [
    "MaxTextQwen3Config",
    "parallelize_qwen3",
    "qwen3_configs",
    "qwen3_debugmodel",
    "qwen3_debugmodel_moe",
    "model_registry",
]

def _testmodel(attn_backend: str = "sdpa") -> MaxTextQwen3Config:
    dim = 128
    head_dim = 16
    n_layers = 3
    vocab_size = 2048
    return MaxTextQwen3Config(
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
            attn_backend=attn_backend,
            rope=common_models.CosSinRoPE.Config(
                dim=head_dim,
                max_seq_len=128,
                theta=1000000.0,
            ),
        ),
        # pytype: enable=wrong-keyword-args
    )


def _testmodel_moe(
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
) -> MaxTextQwen3Config:
    dim = 128
    head_dim = 16
    n_layers = 3
    vocab_size = 2048
    return MaxTextQwen3Config(
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
                max_seq_len=128,
                theta=1000000.0,
            ),
        ),
        # pytype: enable=wrong-keyword-args
    )


def _8b_tpu(attn_backend: str = "sdpa") -> MaxTextQwen3Config:
    """TPU-specific Qwen3 8B model configuration.

    Differs from the baseline models/qwen3/_8b config by setting fuse_qkv=False
    to avoid running attention projection GEMMs as a single fused block on TPU.
    """
    dim = 4096
    head_dim = 128
    n_layers = 36
    vocab_size = 151936
    return MaxTextQwen3Config(
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
                max_seq_len=40960,
                theta=1000000.0,
            ),
        ),
        # pytype: enable=wrong-keyword-args
    )


qwen3_configs = {
    "testmodel": _testmodel("sdpa"),
    "testmodel_moe": _testmodel_moe("sdpa"),
    "debugmodel": MaxTextQwen3Config.from_qwen3(
        qwen3_models.qwen3_configs["debugmodel"]("sdpa")
    ),
    "debugmodel_moe": MaxTextQwen3Config.from_qwen3(
        qwen3_models.qwen3_configs["debugmodel_moe"]("sdpa")
    ),
    "0.6B": MaxTextQwen3Config.from_qwen3(
        qwen3_models.qwen3_configs["0.6B"]("sdpa")
    ),
    "1.7B": MaxTextQwen3Config.from_qwen3(
        qwen3_models.qwen3_configs["1.7B"]("sdpa")
    ),
    "4B": MaxTextQwen3Config.from_qwen3(
        qwen3_models.qwen3_configs["4B"]("sdpa")
    ),
    "8B": _8b_tpu("sdpa"),
    "32B": MaxTextQwen3Config.from_qwen3(
        qwen3_models.qwen3_configs["32B"]("sdpa")
    ),
    "30B-A3B": MaxTextQwen3Config.from_qwen3(
        qwen3_models.qwen3_configs["30B-A3B"]("sdpa")
    ),
}

def qwen3_debugmodel():
    return qwen3_configs["debugmodel"]

def qwen3_debugmodel_moe():
    return qwen3_configs["debugmodel_moe"]

# pytype: disable=wrong-arg-types
register_train_spec(
    name="qwen3_tpu",
    train_spec=TrainSpec(
        model_cls=Qwen3Model,
        model_args=qwen3_configs,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        loss_config=loss.ChunkedLossWrapper.Config(),
        optimizer_config=default_adamw(lr=8e-4, implementation="foreach"),
        lr_scheduler_config=LRSchedulersContainer.Config(warmup_steps=2),
        dataloader_config=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        validator_config=Validator.Config(enable=True),
        state_dict_adapter=Qwen3StateDictAdapter,
    )
)
# pytype: enable=wrong-arg-types


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
    config = MaxTextQwen3Config.from_qwen3(config)
    return ModelSpec(
        name="qwen3",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=qwen3_models.pipeline_llm,
        post_optimizer_build_fn=qwen3_models.register_moe_load_balancing_hook,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
  # Reuse the upstream model specification and change only its FLOP accounting.
  model_spec = qwen3_models.model_registry(
      flavor,
      attn_backend=attn_backend,
      moe_comm_backend=moe_comm_backend,
      converters=converters,
  )
  assert isinstance(model_spec.model, Qwen3Model.Config)
  model_spec.model = MaxTextQwen3Config.from_qwen3(model_spec.model)
  return model_spec

from .config_registry import *
