"""Registration of Conformer model for TorchTitan."""

from torchtitan.components import lr_scheduler
from torchtitan.components import optimizer
from torchtitan.components import tokenizer
from torchtitan.components import validate
from torchtitan.experiments.tpu import loss
from torchtitan.hf_datasets import text_datasets
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
        build_optimizers_fn=optimizer.build_optimizers,
        build_lr_schedulers_fn=lr_scheduler.build_lr_schedulers,
        build_dataloader_fn=text_datasets.build_text_dataloader,
        build_tokenizer_fn=tokenizer.build_hf_tokenizer,
        build_loss_fn=loss.build_cross_entropy_loss,
        build_validator_fn=validate.build_validator,
        state_dict_adapter=None,
    ),
)
