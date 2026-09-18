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

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config.configs import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from . import attention as _tpu_attention  # noqa: F401  (registers splash_tpu
                                            #  before the backend builds its
                                            #  flavor Configs at import time)
from torchtitan.experiments.transformers_modeling_backend import model_registry
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.hf_datasets.text_datasets import DATASETS

from .job_config import TPUTransformersBackendConfig
from .parallelize import parallelize_hf_transformers_tpu

# Small enough to compile and run a step on a single v6e chip.
_DEFAULT_HF_MODEL = "Qwen/Qwen3-0.6B"


def _training_config(*, local_batch_size: int, seq_len: int, steps: int) -> TrainingConfig:
  """Express the (local_batch_size, seq_len) recipe in the token budget."""
  training = TrainingConfig(steps=steps)
  training.num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len
  training.max_context_length = seq_len
  return training


def _set_attn_implementation(model_config, attn: str) -> None:
  """Point HF at an attention implementation this lane can actually run.

  The shared backend configures flex attention unconditionally, which has no
  TPU lowering, and its _configure_hf_attention takes no argument. Set the
  same three fields it sets: HF selects on the underscore field,
  update_from_config re-applies _titan_injected_model_args over the loaded HF
  config, and the public name is kept in step so the two do not disagree.
  splash_tpu is registered with AttentionInterface by this package's attention
  module at import.
  """
  model_config._titan_injected_model_args["attn_implementation"] = attn
  model_config.attn_implementation = attn
  model_config._attn_implementation = attn


def _c4_test_loader() -> GrainDataLoader.Config:
  """This base replaced the HF text dataloader with Grain (upstream #4088)."""
  return GrainDataLoader.Config(
      dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
      shuffle=False,
  )


def _tpu_hf_config(
    flavor: str,
    hf_model: str,
    *,
    local_batch_size: int = 1,
    seq_len: int = 2048,
    steps: int = 10,
    compile: bool = False,
    attn: str = "sdpa",
    fused_loss: bool = False,
) -> TPUTransformersBackendConfig:
  """Build a TPU job config around one of the backend's model flavors."""
  model_spec = model_registry(flavor, seq_len=seq_len)
  model_spec.parallelize_fn = parallelize_hf_transformers_tpu
  # flex attention has no TPU lowering. "sdpa" is the portable default;
  # "splash_tpu" routes attention through the TPU splash kernel via HF's
  # AttentionInterface (see attention.py), which works for any architecture
  # rather than only those whose modules the lane's patch can pattern-match.
  # Go through _configure_hf_attention rather than setting the attribute:
  # it also updates _titan_injected_model_args, which update_from_config
  # re-applies afterwards and would otherwise overwrite a bare assignment.
  _set_attn_implementation(model_spec.model, attn)
  # Route the vocab projection through the lane's fused Pallas CE: the model
  # hands back (hidden, lm_head.weight.T) and the kernel fuses projection,
  # softmax and cross entropy, so the B*S*vocab logits tensor is never built.
  model_spec.model._fused_loss = fused_loss

  return TPUTransformersBackendConfig(
      hf_model=hf_model,
      loss=(PallasCrossEntropyLoss.Config() if fused_loss
            else CrossEntropyLoss.Config()),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      # This base states the recipe as a token budget rather than
      # (local_batch_size, seq_len) -- see _set_batch below.
      training=_training_config(
          local_batch_size=local_batch_size, seq_len=seq_len, steps=steps
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
      dataloader=_c4_test_loader(),
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


def hf_full_splash() -> TPUTransformersBackendConfig:
  """`hf_full` with attention served by the TPU splash-attention kernel."""
  return _tpu_hf_config("full", _DEFAULT_HF_MODEL, attn="splash_tpu")


def hf_full_splash_compile() -> TPUTransformersBackendConfig:
  """Splash attention plus per-layer torch.compile."""
  return _tpu_hf_config("full", _DEFAULT_HF_MODEL, compile=True, attn="splash_tpu")


def hf_full_fused() -> TPUTransformersBackendConfig:
  """`hf_full` with the vocab projection fused into the Pallas CE kernel."""
  return _tpu_hf_config("full", _DEFAULT_HF_MODEL, fused_loss=True)


def hf_full_fused_compile() -> TPUTransformersBackendConfig:
  """Fused CE plus per-layer torch.compile."""
  return _tpu_hf_config("full", _DEFAULT_HF_MODEL, compile=True, fused_loss=True)


def hf_full_max() -> TPUTransformersBackendConfig:
  """Everything on: splash attention, fused CE, per-layer compile."""
  return _tpu_hf_config("full", _DEFAULT_HF_MODEL, compile=True,
                        attn="splash_tpu", fused_loss=True)
