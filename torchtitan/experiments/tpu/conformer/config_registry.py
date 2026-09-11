import os
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer, default_adamw
from torchtitan.components.validate import Validator
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.config.configs import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.tpu.redacted.random_dataset import RandomTokensDatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.experiments.tpu.tpu_job_config import (
    ConformerConfig,
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
)
import torchtitan.experiments.tpu.conformer as tpu_conformer
from torchtitan.experiments.tpu.conformer.infra.parallelize import parallelize_conformer
from torchtitan.distributed.activation_checkpoint import FullAC


def _conformer_model_spec(flavor: str, *, max_context_length: int) -> ModelSpec:
  model_config = tpu_conformer.conformer_args[flavor]
  return ModelSpec(
      name="conformer_tpu",
      flavor=flavor,
      model=model_config,
      max_context_length=max_context_length,
      parallelize_fn=parallelize_conformer,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def _set_batch_shape(
    config: TPUTrainerConfig, *, flavor: str, local_batch_size: int, seq_len: int
) -> None:
  """Re-derive the token-count fields that replaced local_batch_size/seq_len."""
  config.model_spec = _conformer_model_spec(flavor, max_context_length=seq_len)
  config.training.max_context_length = seq_len
  config.training.num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len


def conformer_test() -> TPUTrainerConfig:
  """Conformer test model configuration for TPU."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="tests/assets/tokenizer",
      dump_folder=os.environ.get("TEST_TMPDIR", ".") + "/outputs",
      model_spec=_conformer_model_spec("test", max_context_length=512),
      optimizer=default_adamw(lr=8e-4),
      lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
      # DATASETS["c4_test"] reads the local tests/assets/c4_test asset.
      dataloader=GrainDataLoader.Config(
          dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
      ),
      tokenizer=HuggingFaceTokenizer.Config(),
      validator=Validator.Config(enable=True),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=768 * 512,
          max_context_length=512,
          steps=10,
          dtype="float32",
          mixed_precision_param="bfloat16",
      ),
      parallelism=ParallelismConfig(
          data_parallel_shard_degree=-1,
          data_parallel_replicate_degree=1,
      ),
      activation_checkpoint=FullAC.Config(),
      compile=CompileConfig(
          enable=False,
          backend="tpu",
          components=["model"],
      ),
      tpu_config=TPUConfig(
          use_simple_fsdp=True,
          compile_mode="layer",
      ),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
      ),
      conformer=ConformerConfig(
          use_ctc_loss=True,
      ),
  )


def conformer_test_random() -> TPUTrainerConfig:
  """``conformer_test`` on synthetic tokens (was ``--dataloader.dataset=random``,
  which the Grain ``dataset`` field no longer exposes on the CLI)."""
  config = conformer_test()
  config.dataloader = GrainDataLoader.Config(
      dataset=RandomTokensDatasetConfig(
          vocab_size=tpu_conformer.conformer_args["test"].vocab_size
      ),
  )
  return config


def conformer_debugmodel() -> TPUTrainerConfig:
  """Conformer debug model configuration for TPU."""
  config = conformer_test()
  _set_batch_shape(config, flavor="debugmodel", local_batch_size=4, seq_len=128)
  config.training.steps = 3
  return config

