import os
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer, default_adamw
from torchtitan.components.validate import Validator
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
import torchtitan.experiments.tpu.conformer as tpu_conformer
from torchtitan.experiments.tpu.conformer.infra.parallelize import parallelize_conformer
from torchtitan.distributed.activation_checkpoint import FullAC


def _conformer_model_spec(flavor: str) -> ModelSpec:
  model_config = tpu_conformer.conformer_args[flavor]
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
      optimizer=default_adamw(lr=8e-4),
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
          dtype="float32",
          mixed_precision_param="bfloat16",
      ),
      parallelism=TPUTrainerConfig().parallelism.__class__(
          data_parallel_shard_degree=-1,
          data_parallel_replicate_degree=1,
      ),
      activation_checkpoint=FullAC.Config(),
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


def conformer_debugmodel() -> TPUTrainerConfig:
  """Conformer debug model configuration for TPU."""
  config = conformer_test()
  config.model_spec = _conformer_model_spec("debugmodel")
  config.training = config.training.__class__(
      local_batch_size=4,
      seq_len=128,
      steps=3,
      dtype="float32",
      mixed_precision_param="bfloat16",
  )
  return config



# ---------------------------------------------------------------------------
# Recipe configs
#
# One config per recipe variant under torchtitan/experiments/tpu/recipes,
# carrying everything the recipe used to pass on the launcher command line.
# The Conformer recipe needs no libtpu init flags.
# ---------------------------------------------------------------------------


def _conformer_test_v6e_8() -> TPUTrainerConfig:
  """Shared base for the Conformer v6e-8 recipes.

  Data parallelism is replicated (DDP) rather than sharded: the model is small
  enough that sharding its parameters costs more in collectives than it saves.
  """
  config = conformer_test()
  config.parallelism.data_parallel_shard_degree = 1
  config.parallelism.data_parallel_replicate_degree = -1
  config.training.local_batch_size = 512
  config.training.steps = 20
  config.metrics.log_freq = 1
  # Synthetic audio-shaped batches; the Conformer trainer builds these itself
  # rather than going through the text dataloader.
  config.dataloader.dataset = "random"
  return config


def conformer_test_v6e_8_compile() -> TPUTrainerConfig:
  """Conformer on v6e-8, compiled (recipes/trillium/conformer/v6e-8)."""
  config = _conformer_test_v6e_8()
  config.compile.enable = True
  return config


def conformer_test_v6e_8_eager() -> TPUTrainerConfig:
  """Conformer on v6e-8, eager (recipes/trillium/conformer/v6e-8)."""
  config = _conformer_test_v6e_8()
  config.tpu_config.eager_mode = "DEFER_AND_FUSE"
  return config
