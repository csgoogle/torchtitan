import os
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.experiments.tpu.conformer import model
from torchtitan.experiments.tpu.conformer.infra.parallelize import parallelize_conformer

conformer_args = {
    "test": model.ConformerModelArgs(
        vocab_size=64,
        hidden_dim=512,
        num_layers=17,
        num_heads=8,
        kernel_size=31,
    )
}


def _conformer_model_spec(flavor: str) -> ModelSpec:
  model_config = conformer_args[flavor]
  return ModelSpec(
      name="conformer_tpu",
      flavor=flavor,
      model=model_config,
      parallelize_fn=parallelize_conformer,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def conformer_test() -> TPUTrainerConfig:
  """Conformer test model configuration for TPU."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="tests/assets/tokenizer",
      dump_folder=os.environ.get("TEST_TMPDIR", ".") + "/outputs",
      model_spec=_conformer_model_spec("test"),
      optimizer=OptimizersContainer.Config(lr=8e-4),
      lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
      dataloader=HuggingFaceTextDataLoader.Config(
          dataset="c4_test",
          dataset_path="tests/assets/c4_test",
      ),
      tokenizer=HuggingFaceTokenizer.Config(),
      validator=Validator.Config(enable=True),
      training=TPUTrainerConfig().training.__class__(
          local_batch_size=768,
          seq_len=512,
          steps=10,
          dtype="bfloat16",
      ),
      parallelism=TPUTrainerConfig().parallelism.__class__(
          data_parallel_shard_degree=-1,
          data_parallel_replicate_degree=1,
      ),
      activation_checkpoint=TPUTrainerConfig().activation_checkpoint.__class__(
          mode="full",
      ),
      compile=TPUTrainerConfig().compile.__class__(
          enable=False,
          backend="tpu",
          components=["model"],
      ),
      tpu_config=TPUTrainerConfig().tpu_config.__class__(
          use_simple_fsdp=True,
          compile_mode="layer",
      ),
      splash_attention_kernel=TPUTrainerConfig().splash_attention_kernel.__class__(
          use_splash_attention_kernel=True,
      ),
      conformer=TPUTrainerConfig().conformer.__class__(
          use_ctc_loss=True,
      ),
  )
