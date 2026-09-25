"""TPU configs for the HuggingFace transformers modeling backend.

These are the GPU registry's configs with a few things changed: the config
type (so the TPU entry point accepts it), the model (`TPUHFTransformerModel`,
see model.py), the parallelize function (SimpleFSDP instead of FSDP2), and
`attn_implementation` (sdpa or splash_tpu rather than flex, which has no TPU
path). Inputs, trainer and the rest are the backend's and the TPU lane's
existing ones.

The splash kernel and the fused CE are the lane's usual switches, so any
config can toggle them from the command line:

    --splash_attention_kernel.use_splash_attention_kernel / .no_...
    --loss_kernel.use_loss_kernel / .no_...

    python -m torchtitan.experiments.tpu.train \
        --module=torchtitan.experiments.tpu.transformers_modeling_backend \
        --config=hf_debugmodel
"""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
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
from torchtitan.experiments.transformers_modeling_backend import (
    model_registry,
    TitanModelConfig,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    LossKernelConfig,
    SplashAttentionKernelConfig,
    TPUConfig,
)
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.hf_datasets.text_datasets import DATASETS

from .job_config import TPUTransformersBackendConfig
from .model import TPUHFTransformerModel
from .parallelize import parallelize_hf_transformers_tpu

# Small enough to compile and run a step on a single v6e chip.
_DEFAULT_HF_MODEL = "Qwen/Qwen3-0.6B"


def _training_config(*, local_batch_size: int, seq_len: int, steps: int) -> TrainingConfig:
  """Express the (local_batch_size, seq_len) recipe in the token budget."""
  training = TrainingConfig(steps=steps)
  training.num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len
  training.max_context_length = seq_len
  return training


# The backend's "debugmodel" and "full" flavors.
_FLAVORS = {
    "debugmodel": lambda: TitanModelConfig(
        dim=256, n_layers=2, n_heads=16, n_kv_heads=16
    ),
    "full": TitanModelConfig,
}


def _splash_config(enabled: bool) -> SplashAttentionKernelConfig:
  """Splash switch and block sizes for the splash_tpu attention implementation.

  The library default is 512 for every block. Measured on Llama-3.2-1B at
  batch 8, seq 2048, 4x v6e: 512 gives 17.08% MFU, 1024 gives 18.12%, 2048
  gives 17.78% and 256 gives 11.88% -- an interior optimum, +6.0% over the
  default, against a repeat-measured run-to-run spread of 0.05pp.

  use_splash_attention_kernel selects splash_tpu; the model Config reads it
  (see model.py for why it then clears it).
  """
  block = 1024
  return SplashAttentionKernelConfig(
      use_splash_attention_kernel=enabled,
      sa_block_q=block,
      sa_block_kv=block,
      sa_block_dkv=block,
      sa_block_kv_compute=block,
      sa_block_q_dkv=block,
      sa_block_kv_dkv=block,
      sa_block_kv_dkv_compute=block,
      sa_block_q_dq=block,
      sa_block_kv_dq=block,
  )


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
    splash: bool = False,
    fused_loss: bool = False,
) -> TPUTransformersBackendConfig:
  """Build a TPU job config around one of the backend's model flavors."""
  model_spec = model_registry(flavor, seq_len=seq_len)
  model_spec.model = TPUHFTransformerModel.Config(
      model_config=_FLAVORS[flavor]()
  )
  model_spec.parallelize_fn = parallelize_hf_transformers_tpu

  return TPUTransformersBackendConfig(
      hf_model=hf_model,
      # loss_kernel.use_loss_kernel swaps this for the fused Pallas CE: the
      # model then hands back (hidden, lm_head.weight.T) and the kernel fuses
      # projection, softmax and cross entropy, so the B*S*vocab logits tensor
      # is never built.
      loss=CrossEntropyLoss.Config(),
      loss_kernel=LossKernelConfig(use_loss_kernel=fused_loss),
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
      # parallelize.py for why it is not FSDP2. The backend only accepts the
      # spmd_types SPMD backend, and this base defaults to partial_dtensor.
      parallelism=ParallelismConfig(
          use_simple_fsdp=True, spmd_backend="spmd_types"
      ),
      # "layer" compiles each decoder block on its own. Compiling is what makes
      # the SimpleFSDP choice pay off: its all-gather sits on the .weight
      # access path, so the graph break lands before the layer body and the
      # body stays one compile region.
      tpu_config=TPUConfig(compile_mode="layer"),
      # flex attention has no TPU lowering, so attention is sdpa, or the
      # splash kernel through HF's AttentionInterface (see attention.py).
      splash_attention_kernel=_splash_config(splash),
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
  return _tpu_hf_config("full", _DEFAULT_HF_MODEL, splash=True)


def hf_full_splash_compile() -> TPUTransformersBackendConfig:
  """Splash attention plus per-layer torch.compile."""
  return _tpu_hf_config("full", _DEFAULT_HF_MODEL, compile=True, splash=True)


def hf_full_fused() -> TPUTransformersBackendConfig:
  """`hf_full` with the vocab projection fused into the Pallas CE kernel."""
  return _tpu_hf_config("full", _DEFAULT_HF_MODEL, fused_loss=True)


def hf_full_fused_compile() -> TPUTransformersBackendConfig:
  """Fused CE plus per-layer torch.compile."""
  return _tpu_hf_config("full", _DEFAULT_HF_MODEL, compile=True, fused_loss=True)


def hf_full_max() -> TPUTransformersBackendConfig:
  """Everything on: splash attention, fused CE, per-layer compile."""
  return _tpu_hf_config("full", _DEFAULT_HF_MODEL, compile=True,
                        splash=True, fused_loss=True)
