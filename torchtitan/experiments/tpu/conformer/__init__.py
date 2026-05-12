"""Registration of Conformer model for TorchTitan."""

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.protocols import train_spec
from . import model
from .infra.parallelize import parallelize_conformer

conformer_args = {
    "test": model.ConformerModelArgs(
        vocab_size=64,
        hidden_dim=512,
        num_layers=17,
        num_heads=8,
        kernel_size=31,
    )
}

train_spec.register_train_spec(
    name="conformer_tpu",
    train_spec=train_spec.TrainSpec(
        model_cls=model.Conformer,
        model_args=conformer_args,
        parallelize_fn=parallelize_conformer,
        pipelining_fn=None,
        loss_config=CrossEntropyLoss.Config(),
        optimizer_config=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler_config=LRSchedulersContainer.Config(warmup_steps=2),
        dataloader_config=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        tokenizer_config=HuggingFaceTokenizer.Config(),
        validator_config=Validator.Config(enable=True),
        state_dict_adapter=None,
    ),
)
