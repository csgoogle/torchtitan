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
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

__all__ = [
    "olmo2_debugmodel",
    "olmo2_debugmodel_fused_ce",
    "olmo2_1b",
    "olmo2_1b_fused_ce",
]

_DEBUGMODEL_CONTEXT_LENGTH = 2048
_1B_CONTEXT_LENGTH = 4096

# Overrides are identified by the import path of the decorated factory.
_FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"
)


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Select the fused loss and its pre-construction LM-head override together.

  The override swaps the plain ``Linear`` lm_head for ``FusedLossLinearWrapper``
  (same weight and checkpoint key), which hands ``(hidden, weight.T)`` to the
  fused kernel, so the ``[tokens, vocab]`` logits are never materialized. OLMo 2
  has no logit soft-cap or multiplier, so the loss keeps its defaults.
  """
  config.loss = PallasCrossEntropyLoss.Config()
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)


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


def olmo2_debugmodel_fused_ce() -> TPUTrainerConfig:
  """``olmo2_debugmodel`` with the fused Pallas linear + cross entropy."""
  config = olmo2_debugmodel()
  _enable_pallas_cross_entropy(config)
  return config


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


def olmo2_1b_fused_ce() -> TPUTrainerConfig:
  """``olmo2_1b`` with the fused Pallas linear + cross entropy."""
  config = olmo2_1b()
  _enable_pallas_cross_entropy(config)
  return config
