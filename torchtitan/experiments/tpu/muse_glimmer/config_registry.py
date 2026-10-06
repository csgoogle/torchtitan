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
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

__all__ = ["muse_glimmer_debugmodel"]

_DEBUGMODEL_CONTEXT_LENGTH = 2048


def muse_glimmer_debugmodel() -> TPUTrainerConfig:
  """Muse Glimmer debug model (8 layers, dim 256) on TPU with SimpleFSDP.

  Unlike upstream ``muse_glimmer_debugmodel`` this uses plain
  ``CrossEntropyLoss``: ``TPUTrainer`` rebuilds ``loss_fn`` from the config via
  ``loss.build_cross_entropy_loss``, which drops ``ChunkedLossWrapper``. The
  soft-cap lives in ``SoftCappedLinear`` (the ``lm_head``), so it is still
  applied on the full-logits path.

  The Pallas fused loss (``--loss_kernel.use_loss_kernel``) is not supported:
  it bypasses ``SoftCappedLinear.forward`` and would drop the output multiplier
  and soft-cap, so ``workarounds.use_output_projection_patch`` rejects it.
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
