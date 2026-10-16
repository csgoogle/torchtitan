"""TPU training recipes for Muse Spark Open (text-only)."""

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import CompileConfig
from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExpertsConverter
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.token_dispatcher import TpuTokenDispatcherConverter
from torchtitan.experiments.tpu.tpu_job_config import LossKernelConfig
from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

__all__ = [
    "muse_spark_debugmodel",
    "muse_spark_6b",
    "muse_spark_220b_4x4x4",
]

_DEBUGMODEL_CONTEXT_LENGTH = 2048
_6B_CONTEXT_LENGTH = 8192
_220B_CONTEXT_LENGTH = 8192

# Dense tpu7x-4x4x4 launch flags, as in the Muse Glimmer 30B 4x4x4 recipes.
_TPU7X_4X4X4_FLAGS = tpu_flags.merge_flags(
    tpu_flags.TPU7X_THROUGHPUT_P_STATE,
    {"xla_tpu_scoped_vmem_limit_kib": 65472},
    tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
    tpu_flags.HIERARCHICAL_REDUCE_SCATTER,
)

# Overrides are identified by the import path of the decorated factory.
_FUSED_LOSS_OVERRIDE = (
    "torchtitan.experiments.tpu.overrides.fused_loss."
    "fused_loss_soft_capped_lm_head"
)


def _tpu_moe_converters():
  return [TpuTokenDispatcherConverter.Config(), TpuGroupedExpertsConverter.Config()]


def _use_pallas_grouped_mm(
    config: TPUTrainerConfig, *, tile_m: int, tile_k: int = 0, tile_n: int = 0
) -> None:
  """Runs the routed experts with the Pallas segment matmul.

  Native ``torch._grouped_mm`` dWeight is incorrect at large M on TPU
  (https://github.com/google-pytorch/torch_tpu/issues/2668), which is why the
  Qwen3 MoE recipes use Pallas as well.
  """
  for _, experts, _, _ in config.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.use_pallas_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = tile_m
    experts.pallas_grouped_mm_tile_k = tile_k
    experts.pallas_grouped_mm_tile_n = tile_n


def _enable_pallas_cross_entropy(config: TPUTrainerConfig) -> None:
  """Fuses the soft-capped LM head into a Pallas cross entropy.

  Same as the Muse Glimmer recipes: the lm_head is a ``SoftCappedLinear``, and
  its soft cap and output multiplier are applied per logits tile inside the
  kernel, so the [tokens, 202048] FP32 logits are never materialized.
  """
  lm_head = config.model_spec.model.lm_head
  config.loss = PallasCrossEntropyLoss.Config(
      soft_cap=lm_head.output_soft_cap_temp,
      output_multiplier=lm_head.output_multiplier,
  )
  if _FUSED_LOSS_OVERRIDE not in config.override.imports:
    config.override.imports.append(_FUSED_LOSS_OVERRIDE)


def muse_spark_debugmodel() -> TPUTrainerConfig:
  """Muse Spark debug model (8 layers, dim 256, 8 experts top-2) with SimpleFSDP.

  Same flavor as upstream ``debugmodel``: layers 3 and 7 are global (no RoPE,
  position temperature), the rest use a 128-token sliding window.
  """
  # Imported here to avoid a circular import: the package __init__ imports this
  # module at the end.
  from torchtitan.experiments.tpu.muse_spark import model_registry  # pylint: disable=g-import-not-at-top

  model_spec = model_registry(
      "debugmodel",
      seq_len=_DEBUGMODEL_CONTEXT_LENGTH,
      converters=_tpu_moe_converters(),
  )
  config = TPUTrainerConfig(
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
      checkpoint=CheckpointManager.Config(enable=False),
      activation_checkpoint=SelectiveAC.Config(),
      validator=Validator.Config(enable=False),
  )
  _use_pallas_grouped_mm(config, tile_m=512)
  return config


def muse_spark_6b() -> TPUTrainerConfig:
  """Scaled-down Muse Spark (``6B-A1.4B``) on one v6e-8 host.

  816B-A42B architecture at dim 2048, 24 layers (6 global), 64 experts top-8,
  202048 vocab, 8192-token context (2048-token window). SimpleFSDP over all
  chips, one sequence per chip, per-layer compile, Splash attention and the
  fused soft-capped cross entropy.
  """
  from torchtitan.experiments.tpu.muse_spark import model_registry  # pylint: disable=g-import-not-at-top

  model_spec = model_registry(
      "6B-A1.4B",
      seq_len=_6B_CONTEXT_LENGTH,
      converters=_tpu_moe_converters(),
  )
  config = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      # Test tokenizer, like the other TPU recipes.
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=model_spec,
      optimizer=default_adamw(lr=3e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(warmup_steps=10),
      training=TrainingConfig(
          max_norm=1.0,
          num_tokens_per_microbatch_per_dp_rank=_6B_CONTEXT_LENGTH,
          max_context_length=_6B_CONTEXT_LENGTH,
          steps=20,
          mixed_precision_reduce="float32",
      ),
      dataloader=c4_test_dataloader(),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          use_simple_fsdp=True,
          enable_tpu_topo_mesh=True,
          data_parallel_shard_degree=-1,
      ),
      checkpoint=CheckpointManager.Config(enable=False),
      activation_checkpoint=SelectiveAC.Config(),
      xla_flags=tpu_flags.merge_flags(xla_enable_hlo_sharding_v3=False),
      compile=CompileConfig(enable=True, backend="tpu", components=["model"]),
      splash_attention_kernel=SplashAttentionKernelConfig(
          use_splash_attention_kernel=True,
          sa_block_q=1024,
          sa_block_kv=1024,
          sa_block_dkv=1024,
          sa_block_kv_compute=1024,
          sa_block_q_dkv=1024,
          sa_block_kv_dkv=1024,
          sa_block_kv_dkv_compute=1024,
          sa_block_q_dq=1024,
          sa_block_kv_dq=1024,
      ),
      loss_kernel=LossKernelConfig(use_loss_kernel=False),
      tpu_config=TPUConfig(
          eager_mode="DEFER_AND_FUSE",
          use_graph_split=True,
          compile_mode="layer",
      ),
      validator=Validator.Config(enable=False),
  )
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags, {"xla_tpu_scoped_vmem_limit_kib": 65472}
  )
  _use_pallas_grouped_mm(config, tile_m=1024)
  # Full FP32 logits are 6.17 GiB per chip at 8192 tokens and run out of HBM.
  _enable_pallas_cross_entropy(config)
  return config


def muse_spark_220b_4x4x4() -> TPUTrainerConfig:
  """``220B-A44B`` on a tpu7x-4x4x4 slice (64 chips, 128 devices).

  Production width/depth and per-token compute with 64 experts. SimpleFSDP
  Zero-3 over all 128 devices (experts, router gates and vocabulary shard on
  their hidden dim), one 8192-token sequence per device, full AC, Splash and
  the fused soft-capped cross entropy. FP32 weights and AdamW state are
  ~27.5 GiB per device. A first-run configuration: not yet tuned.
  """
  from torchtitan.experiments.tpu.muse_spark import model_registry  # pylint: disable=g-import-not-at-top

  config = muse_spark_6b()
  config.model_spec = model_registry(
      "220B-A44B",
      seq_len=_220B_CONTEXT_LENGTH,
      converters=_tpu_moe_converters(),
  )
  config.training.num_tokens_per_microbatch_per_dp_rank = _220B_CONTEXT_LENGTH
  config.training.max_context_length = _220B_CONTEXT_LENGTH
  config.training.steps = 15
  config.parallelism.fsdp_reshard_after_forward = "always"
  config.activation_checkpoint = FullAC.Config()
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags, _TPU7X_4X4X4_FLAGS
  )
  # Untiled (tile_k = tile_n = 0) 4096 x 4096 expert weights need 104 MiB of
  # VMEM in moe_gmm_fwd against v7x's 64 MiB; 2048 x 2048 weight tiles fit
  # (fwd + bwd checked at 65536 routed rows under a 65472 KiB scoped limit).
  _use_pallas_grouped_mm(config, tile_m=1024, tile_k=2048, tile_n=2048)
  _enable_pallas_cross_entropy(config)
  return config
