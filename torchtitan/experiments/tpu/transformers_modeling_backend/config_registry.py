"""TPU configs for the HuggingFace transformers modeling backend.

These are the GPU registry's configs with three things changed: the config
type (so the TPU entry point accepts it), the parallelize function (SimpleFSDP
instead of FSDP2), and `attn_implementation` (sdpa rather than flex, which has
no TPU path). Model, inputs, loss and trainer are the backend's and the TPU
lane's existing ones.

    python -m torchtitan.experiments.tpu.train \
        --module=torchtitan.experiments.tpu.transformers_modeling_backend \
        --config=hf_debugmodel
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config.configs import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.transformers_modeling_backend import model_registry
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader

from .job_config import TPUTransformersBackendConfig
from .parallelize import parallelize_hf_transformers_tpu

# Small enough to compile and run a step on a single v6e chip.
_DEFAULT_HF_MODEL = "Qwen/Qwen3-0.6B"


def _tpu_hf_config(
    flavor: str,
    hf_model: str,
    *,
    local_batch_size: int = 1,
    seq_len: int = 2048,
    steps: int = 10,
    compile: bool = False,
) -> TPUTransformersBackendConfig:
  """Build a TPU job config around one of the backend's model flavors."""
  model_spec = model_registry(flavor)
  model_spec.parallelize_fn = parallelize_hf_transformers_tpu
  # flex attention has no TPU lowering; sdpa is what the TPU lane's own models
  # use, and it is what the splash-attention patch would later replace.
  model_spec.model.attn_implementation = "sdpa"

  return TPUTransformersBackendConfig(
      hf_model=hf_model,
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
          local_batch_size=local_batch_size,
          seq_len=seq_len,
          steps=steps,
      ),
      # SimpleFSDP is the data-parallel implementation on this lane; see
      # parallelize.py for why it is not FSDP2.
      parallelism=ParallelismConfig(use_simple_fsdp=True),
      # "layer" compiles each decoder block on its own. Compiling is what makes
      # the SimpleFSDP choice pay off: its all-gather sits on the .weight
      # access path, so the graph break lands before the layer body and the
      # body stays one compile region.
      tpu_config=TPUConfig(compile_mode="layer"),
      compile=CompileConfig(
          enable=compile,
          backend="tpu",
          components=["model"],
      ),
      dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
      metrics=MetricsProcessor.Config(log_freq=1),
      checkpoint=CheckpointManager.Config(
          interval=10,
          last_save_model_only=False,
      ),
      activation_checkpoint=SelectiveAC.Config(),
  )


def hf_debugmodel() -> TPUTransformersBackendConfig:
  """Two-layer, narrow model -- the smoke test for the TPU path."""
  return _tpu_hf_config("debugmodel", _DEFAULT_HF_MODEL, seq_len=512)


def hf_full() -> TPUTransformersBackendConfig:
  """The published model at its real width and depth."""
  return _tpu_hf_config("full", _DEFAULT_HF_MODEL)


def hf_debugmodel_compile() -> TPUTransformersBackendConfig:
  """`hf_debugmodel` with per-layer torch.compile on the TPU backend."""
  return _tpu_hf_config("debugmodel", _DEFAULT_HF_MODEL, seq_len=512, compile=True)


def hf_full_compile() -> TPUTransformersBackendConfig:
  """`hf_full` with per-layer torch.compile on the TPU backend."""
  return _tpu_hf_config("full", _DEFAULT_HF_MODEL, compile=True)
