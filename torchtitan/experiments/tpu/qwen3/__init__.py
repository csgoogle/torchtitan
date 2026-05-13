from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.components.validate import Validator
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec
import torchtitan.models.common as common_models
import torchtitan.models.qwen3 as qwen3_models

from .infra.parallelize import parallelize_qwen3

from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter

__all__ = [
    "parallelize_qwen3",
    "qwen3_configs",
]

def _testmodel(attn_backend: str = "sdpa") -> qwen3_models.Qwen3Model.Config:
    dim = 128
    head_dim = 16
    n_layers = 3
    vocab_size = 2048
    return qwen3_models.Qwen3Model.Config(
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
            param_init=qwen3_models._output_linear_init(dim),
        ),
        rope=common_models.RoPE.Config(
            dim=head_dim,
            max_seq_len=128,
            theta=1000000.0,
            backend="cos_sin",
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
        ),
        # pytype: enable=wrong-keyword-args
    )


def _testmodel_moe(
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
) -> qwen3_models.Qwen3Model.Config:
    dim = 128
    head_dim = 16
    n_layers = 3
    vocab_size = 2048
    return qwen3_models.Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=qwen3_models._qwen3_norm(dim),
        tok_embeddings=common_models.Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=qwen3_models._EMBEDDING_INIT
        ),
        lm_head=common_models.Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=qwen3_models._output_linear_init(dim),
        ),
        rope=common_models.RoPE.Config(
            dim=head_dim,
            max_seq_len=128,
            theta=1000000.0,
            backend="cos_sin",
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
        ),
        # pytype: enable=wrong-keyword-args
    )


qwen3_configs = {
    "testmodel": _testmodel("sdpa"),
    "testmodel_moe": _testmodel_moe("sdpa"),
    "debugmodel": qwen3_models.qwen3_configs["debugmodel"]("sdpa"),
    "debugmodel_moe": qwen3_models.qwen3_configs["debugmodel_moe"]("sdpa"),
    "0.6B": qwen3_models.qwen3_configs["0.6B"]("sdpa"),
    "1.7B": qwen3_models.qwen3_configs["1.7B"]("sdpa"),
    "4B": qwen3_models.qwen3_configs["4B"]("sdpa"),
    "32B": qwen3_models.qwen3_configs["32B"]("sdpa"),
}

# pytype: disable=wrong-arg-types
register_train_spec(
    name="qwen3_tpu",
    train_spec=TrainSpec(
        model_cls=Qwen3Model,
        model_args=qwen3_configs,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        loss_config=ChunkedCELoss.Config(),
        optimizer_config=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler_config=LRSchedulersContainer.Config(warmup_steps=2),
        dataloader_config=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        validator_config=Validator.Config(enable=True),
        state_dict_adapter=Qwen3StateDictAdapter,
    )
)
# pytype: enable=wrong-arg-types
