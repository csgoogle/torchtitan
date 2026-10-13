"""TPU training recipes for OLMo 2."""

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import CompileConfig
from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

__all__ = [
    "olmo2_debugmodel",
    "olmo2_1b",
]

_DEBUGMODEL_CONTEXT_LENGTH = 2048
_1B_CONTEXT_LENGTH = 4096


def olmo2_debugmodel() -> TPUTrainerConfig:
  """OLMo 2 debug model (8 layers, dim 512, head_dim 128) with SimpleFSDP.

  Eager by default; ``--compile.enable --compile.backend=tpu`` runs the same
  recipe with per-block compile.
  """
  # Imported here to avoid a circular import: the package __init__ imports this
  # module at the end.
  from torchtitan.experiments.tpu.examples.olmo2 import model_registry  # pylint: disable=g-import-not-at-top

  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_registry(
          "debugmodel", seq_len=_DEBUGMODEL_CONTEXT_LENGTH
      ),
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * _DEBUGMODEL_CONTEXT_LENGTH,
          max_context_length=_DEBUGMODEL_CONTEXT_LENGTH,
          steps=10,
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(use_simple_fsdp=True),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(),
      validator=Validator.Config(
          freq=5,
          steps=10,
      ),
  )


def olmo2_1b() -> TPUTrainerConfig:
  """``allenai/OLMo-2-0425-1B``: per-block compile, SimpleFSDP, bf16 params.

  Plain SDPA and the full-logits loss; the kernel rungs (Splash, fused CE) are
  not enabled yet. Pass the checkpoint's tokenizer and safetensors with
  ``--hf_assets_path`` (and ``--checkpoint.initial_load_in_hf`` to start from
  the pretrained weights).
  """
  from torchtitan.experiments.tpu.examples.olmo2 import model_registry  # pylint: disable=g-import-not-at-top

  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./assets/hf/OLMo-2-0425-1B",
      model_spec=model_registry("1B", seq_len=_1B_CONTEXT_LENGTH),
      optimizer=default_adamw(lr=4e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
      training=TrainingConfig(
          max_norm=1.0,
          num_tokens_per_microbatch_per_dp_rank=1 * _1B_CONTEXT_LENGTH,
          max_context_length=_1B_CONTEXT_LENGTH,
          steps=10,
          mixed_precision_param="bfloat16",
          mixed_precision_reduce="float32",
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          data_parallel_shard_degree=-1,
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(),
      compile=CompileConfig(
          enable=True,
          backend="tpu",
          components=["model"],
      ),
      validator=Validator.Config(enable=False),
  )
