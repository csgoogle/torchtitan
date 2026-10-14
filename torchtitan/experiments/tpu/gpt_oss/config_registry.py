"""Unoptimized GPT-OSS SDPA recipes for correctness bring-up."""

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig, TPUTrainerConfig


def gpt_oss_debugmodel() -> TPUTrainerConfig:
    from . import model_registry

    return TPUTrainerConfig(
        model_spec=model_registry("debugmodel", seq_len=256),
        hf_assets_path="./tests/assets/tokenizer",
        loss=CrossEntropyLoss.Config(),
        optimizer=default_adamw(lr=8e-4, implementation="foreach"),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            max_context_length=256,
            num_tokens_per_microbatch_per_dp_rank=256,
            steps=10,
            mixed_precision_reduce="float32",
        ),
        dataloader=c4_test_dataloader(),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(use_simple_fsdp=True),
        checkpoint=CheckpointManager.Config(enable=False),
        compile=CompileConfig(enable=True, backend="tpu", components=["model"]),
        tpu_config=TPUConfig(bld_block_layout=False),
    )


def gpt_oss_debugmodel_graphtrainer() -> TPUTrainerConfig:
    """Opt-in joint forward/backward tracing; keep the baseline recipe intact."""
    config = gpt_oss_debugmodel()
    config.tpu_config.use_graph_trainer = True
    config.tpu_config.compile_mode = "aot_fx_trace"
    return config


def gpt_oss_debugmodel_splash() -> TPUTrainerConfig:
    """Opt-in GraphTrainer + sink-aware Tokamax; default recipes stay on SDPA."""
    config = gpt_oss_debugmodel_graphtrainer()
    config.override.imports.append(
        "torchtitan.experiments.tpu.gpt_oss.splash_attention.use_sink_splash"
    )
    return config


def gpt_oss_debugmodel_fused_ce() -> TPUTrainerConfig:
    """Opt-in Splash + fused LM-head cross entropy on the debug architecture."""
    from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss

    config = gpt_oss_debugmodel_splash()
    config.override.imports.append(
        "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head"
    )
    # Cover the default 256-token microbatch and 256-wide hidden state exactly.
    config.loss = PallasCrossEntropyLoss.Config(
        b_block_size=256, h_block_size=256, v_block_size=2048
    )
    return config


def gpt_oss_20b() -> TPUTrainerConfig:
    """20B architecture on SDPA; not a benchmarked production recipe."""
    from . import model_registry

    config = gpt_oss_debugmodel()
    config.model_spec = model_registry("20b", seq_len=256)
    config.optimizer = default_adamw(lr=3e-4, implementation="foreach")
    return config


def gpt_oss_120b() -> TPUTrainerConfig:
    """120B architecture on SDPA; not a benchmarked production recipe."""
    from . import model_registry

    config = gpt_oss_20b()
    config.model_spec = model_registry("120b", seq_len=256)
    return config
