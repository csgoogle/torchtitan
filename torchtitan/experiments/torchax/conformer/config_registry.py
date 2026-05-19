import os
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.experiments.torchax.torchax_job_config import TorchaxJobConfig
from torchtitan.experiments.torchax import conformer as torchax_conformer


def _model_spec(flavor: str) -> ModelSpec:
  return ModelSpec(
      name="conformer",
      flavor=flavor,
      model=torchax_conformer.args[flavor],
      parallelize_fn=None,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def conformer() -> TorchaxJobConfig:
  """Conformer benchmark model on torchax."""
  cfg = TorchaxJobConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="tests/assets/tokenizer",
      dump_folder=os.environ.get("TEST_TMPDIR", ".") + "/outputs",
      model_spec=_model_spec("test"),
      optimizer=OptimizersContainer.Config(lr=8e-4),
      lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
      dataloader=HuggingFaceTextDataLoader.Config(
          dataset="c4_test",
          dataset_path="tests/assets/c4_test",
      ),
      tokenizer=HuggingFaceTokenizer.Config(),
      validator=Validator.Config(enable=True),
      training=TorchaxJobConfig().training.__class__(
          local_batch_size=16,
          seq_len=512,
          steps=100,
          dtype="bfloat16",
      ),
      parallelism=TorchaxJobConfig().parallelism.__class__(
          data_parallel_shard_degree=-1,
          data_parallel_replicate_degree=1,
      ),
      activation_checkpoint=TorchaxJobConfig().activation_checkpoint.__class__(
          mode="full",
      ),
      compile=TorchaxJobConfig().compile.__class__(
          enable=False,
          backend="tpu",
          components=["model"],
      ),
      splash_attention_kernel=TorchaxJobConfig().splash_attention_kernel.__class__(
          use_splash_attention_kernel=True,
      ),
      conformer=TorchaxJobConfig().conformer.__class__(
          use_ctc_loss=True,
      ),
  )
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = False
  return cfg
