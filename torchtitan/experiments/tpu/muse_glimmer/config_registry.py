"""TPU training recipes for Muse Glimmer (text-only)."""

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

__all__ = [
    "muse_glimmer_debugmodel",
    "muse_glimmer_debugmodel_fused_ce",
]

_DEBUGMODEL_CONTEXT_LENGTH = 2048

# Overrides are identified by the import path of the decorated factory.
_FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss."
    "fused_loss_soft_capped_lm_head"
)


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Select the fused loss and its pre-construction LM-head override together.

  Same idiom as the qwen3 registry, with two Muse specifics: the override
  targets ``SoftCappedLinear.Config`` (Muse's lm_head type), and the head's
  soft-cap / output multiplier are copied into the loss config, because the
  wrapper only hands ``(hidden, weight.T)`` to the loss and the transform is
  applied inside the fused kernel.
  """
  lm_head = config.model_spec.model.lm_head
  config.loss = PallasCrossEntropyLoss.Config(
      soft_cap=lm_head.output_soft_cap_temp,
      output_multiplier=lm_head.output_multiplier,
  )
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)


def muse_glimmer_debugmodel() -> TPUTrainerConfig:
  """Muse Glimmer debug model (8 layers, dim 256) on TPU with SimpleFSDP.

  Unlike upstream ``muse_glimmer_debugmodel`` this uses plain
  ``CrossEntropyLoss``: ``TPUTrainer`` rebuilds ``loss_fn`` from the config via
  ``loss.build_cross_entropy_loss``, which drops ``ChunkedLossWrapper``. The
  soft-cap lives in ``SoftCappedLinear`` (the ``lm_head``), so it is still
  applied on the full-logits path.

  The legacy Pallas fused loss (``--loss_kernel.use_loss_kernel``) is not
  supported: it bypasses ``SoftCappedLinear.forward`` and would drop the output
  multiplier and soft-cap, so ``workarounds.use_output_projection_patch``
  rejects it. Use ``muse_glimmer_debugmodel_fused_ce`` instead, which applies
  the soft-cap inside the fused kernel.
  """
  # Imported here to avoid a circular import: the package __init__ imports this
  # module at the end.
  from torchtitan.experiments.tpu.muse_glimmer import model_registry  # pylint: disable=g-import-not-at-top

  model_spec = model_registry("debugmodel", seq_len=_DEBUGMODEL_CONTEXT_LENGTH)
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
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
  )


def muse_glimmer_debugmodel_fused_ce() -> TPUTrainerConfig:
  """``muse_glimmer_debugmodel`` with the fused Pallas cross entropy.

  The ``SoftCappedLinear`` lm_head is replaced by ``FusedLossLinearWrapper`` and
  ``PallasCrossEntropyLoss`` re-applies the head's ``cap * tanh(z * mul / cap)``
  to each logits tile in-kernel (cap 20, multiplier 1.0 for the debug flavor),
  so the full fp32 logits are never materialized.
  """
  config = muse_glimmer_debugmodel()
  _enable_pallas_cross_entropy(config)
  return config
