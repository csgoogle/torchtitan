# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the GSM8K example (``--module gsm8k``).

See README.md for the requirements behind these settings and the measured results.
"""

from renderers import Qwen3RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.examples.gsm8k import GSM8KRollouter
from torchtitan.experiments.rl.losses import DAPOLoss
from torchtitan.experiments.rl.models.cast_linear import LMHeadCastConverter
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RenderersLibraryConfig
from torchtitan.models.qwen3 import model_registry


def rl_grpo_qwen3_0_6b_gsm8k_tpu() -> Controller.Config:
    """GRPO (DAPO clip-higher) on GSM8K for Qwen3-0.6B on a TPU slice.

    Requirements, in order of importance (README.md has the evidence):

    1. fp32 master weights. Keep ``TrainingConfig.dtype`` at its float32
       default; bf16 (round-to-nearest) masters stalled learning at this lr.
    2. An exact sampler: the Gumbel noise must come from an fp32 uniform. With
       a bf16 uniform every run collapsed. Here ``LMHeadCastConverter`` makes
       the generator's logits fp32 (the trainer and generator share this
       ``model_spec``), so the vLLM TPU backend's stock ``rand_like(logits)``
       is already fp32. Keep the cast unless the backend draws ``u`` in fp32.
    3. pi_old = the generator's logprobs with the PPO clip; no importance
       reweighting is needed once the sampler is exact.
    """
    seq_len = 2048
    return Controller.Config(
        model_spec=model_registry(
            "0.6B",
            # torch_tpu has no varlen attention kernel; SDPA needs the packed-
            # document mask built by Decoder.get_attention_masks.
            attn_backend="sdpa",
            # fp32 lm_head logits on the trainer and on the generator. On the
            # generator this also keeps the sampler's Gumbel uniform in fp32.
            converters=[LMHeadCastConverter.Config()],
        ),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        async_loop=AsyncLoopConfig(
            num_training_steps=696,
            # 128 prompts x 16 samples = 2048 completions per optimizer step.
            num_prompts_per_train_step=128,
            num_samples_per_prompt=16,
            # The measured runs were synchronous: every rollout used the
            # latest weights (on-policy, strict FIFO).
            target_offpolicy_steps=0,
            window_fraction=None,
            validation=ValidationConfig(num_samples=512),
        ),
        # The measured trainer ran eagerly on TPU.
        compile=CompileConfig(enable=False),
        rollouter=GSM8KRollouter.Config(),
        renderer=RenderersLibraryConfig(
            renderers_config=Qwen3RendererConfig(enable_thinking=False)
        ),
        metrics=MetricsProcessor.Config(),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(
                lr=2.5e-6, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
            ),
            # Linear warmup over 35 steps, then cosine decay to 0.1x over a
            # 1170-step horizon. The horizon is longer than the run on purpose:
            # it is the schedule the measured runs used, cut off at step 696.
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=35,
                decay_type="cosine",
                min_lr_factor=0.1,
                total_steps=1170,
            ),
            # fp32 master weights (the default dtype). Do NOT set
            # dtype="bfloat16": see TrainingConfig.dtype.
            # Rows of 2048 tokens, 2 rows per microbatch.
            training=TrainingConfig(
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
                max_norm=1.0,
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=-1,
                tensor_parallel_degree=1,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=100,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=1,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            cudagraph=VLLMCudagraphConfig(enable=False),
            sampling=SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=1792),
        ),
    )
