# GSM8K GRPO for Qwen3-0.6B on TPU

`config_registry.py:rl_grpo_qwen3_0_6b_gsm8k_tpu` trains Qwen3-0.6B on the GSM8K train split with GRPO and DAPO's clip-higher bounds. It reports held-out pass@1 on a fixed 512-problem subset of the GSM8K test split.

```
python3 -m torchtitan.experiments.rl.train \
    --module gsm8k --config rl_grpo_qwen3_0_6b_gsm8k_tpu \
    --hf_assets_path=<path to Qwen3-0.6B>
```

> **Status: draft, not yet runnable end to end on TPU.** The results below were measured with a standalone colocated trainer that uses the same hyperparameters (see [How the numbers were measured](#how-the-numbers-were-measured)). This config has not been run end to end under the torchtitan RL controller, and as written it cannot run on TPU. Known blockers (details in [google-pytorch/torchtitan#538](https://github.com/google-pytorch/torchtitan/pull/538)):
>
> - `PolicyTrainer` and `VLLMGenerator` accept only varlen or flex attention (`actors/trainer.py`, `actors/generator.py`), but this config uses SDPA.
> - SDPA needs the packed-document mask (`Decoder.get_attention_masks`), which is not in this tree yet.
> - `models/attention.py` registers a CUDA attention backend in vLLM's `CUSTOM` slot. The generator requests `CUSTOM` for every model that does not use flex attention, and the vLLM TPU plugin registers its own backend in the same slot.
> - `train.py` uses only the controller's own host and partitions devices only through `CUDA_VISIBLE_DEVICES`.
>
> It does **not** depend on the fp32-uniform fix in the vLLM TPU backend. The trainer and the generator share one `model_spec`, and that `model_spec` includes `LMHeadCastConverter`. So the generator's `compute_logits` returns fp32 logits, and the backend's stock `u = torch.rand_like(logits)` is already fp32. This config therefore runs the measured "fp32 logits and arithmetic, fp32 `u`" sampler. The 85.25 headline comes from a different arm, "bf16 logits, fp32 `u`". See requirement 2 and the last table.

## Requirements, in order of importance

### 1. fp32 master weights

Keep `TrainingConfig.dtype` at its default, `"float32"`. The forward and backward still run in bf16 through FSDP mixed precision (`mixed_precision_param="bfloat16"`). Only the optimizer's copy of the weights has to stay fp32.

With `dtype="bfloat16"`, the weights are stored in bf16 and updated with round-to-nearest. At the peak lr of 2.5e-6, an AdamW step is below half a bf16 ulp for every weight with |w| >= 2^-10, so most updates round back to the old value. The run does not diverge. It stops learning.

| master weights (348 steps, lr 2.5e-6) | runs | held-out pass@1, steps 304-344 | stalled (< 79.6) |
|---|---|---|---|
| fp32 (default) | 5 | **84.24** (83.50-84.60) | 0/5 |
| bf16, round-to-nearest (weights, grads and Adam moments, or the weights alone) | 10 | **70.43** (69.56-71.26) | **10/10** |
| fp32 weights, bf16 Adam moments | 3 | 84.37 (83.40-85.06) | 0/3 |
| bf16 weights with stochastic rounding | 3 | 83.54 (82.26-84.31) | 0/3 |

- The table shows the mean, with the range in parentheses.
- Same-build effect of bf16 masters: **-13.94 pp** [95% CI -14.59, -13.29] (7 runs vs 5).
- By step 348, 1.85% of the rank-0 weight elements had changed their bf16 value with bf16 masters, against 36.0% with fp32 masters.
- The weights-only arm reproduces the stall. bf16 moment storage does not, so only the weight update needs fp32 accumulation.
- Stochastic rounding of bf16 weights avoids the stall. Its level relative to fp32 is not established (3 runs).
- Raising the lr does not substitute cleanly. fp32 masters at 4x lr drop to 74.50 (6 runs). bf16 masters at 6x and 8x lr reach 83.38 and 83.85 (4 runs each), below fp32 at 1x on point estimates.

### 2. An exact sampler: draw the Gumbel uniform in fp32

The vLLM TPU backend samples with the Gumbel-max trick. The uniform `u` used to be drawn in the logits dtype (bf16). Clamped to `1 - eps(bf16)`, the Gumbel noise is capped at 4.85, against 15.94 for an fp32 `u`. Tokens more than a few nats below the top logit are then under-sampled, or never drawn. The trainer's logprobs do not see this bias.

| sampler (lr 2.5e-6 unless noted) | runs | collapsed |
|---|---|---|
| bf16 logits, bf16 `u` (stock) | 35 (all lr and pi_old settings) | **35/35** |
| bf16 logits, fp32 `u` | 21 (13 at 348 steps, 8 at 696) | **0/21** |
| bf16-head logits cast to fp32, fp32 arithmetic, `u` on the bf16 grid | 4 | 4/4 |
| fp32 logits and arithmetic, fp32 `u` (fp32 lm_head, or bf16-head logits cast to fp32) | 78 | 0/78 |

- At 1x lr, stock-sampler runs learn normally at first. They then collapse at steps 63-69 (15 runs), and held-out pass@1 falls to a mean minimum of 17.2.
- The precision of `u` is what matters. Sampler arithmetic precision is neither necessary nor sufficient. A true fp32 lm_head with a bf16-grid `u` was never run.
- Pooled over every fp32-`u` run in the study: 0/147 collapsed.

**What this config runs.** `LMHeadCastConverter` makes the generator's logits fp32. With the stock backend, the sampler arithmetic and `u` are then fp32 as well. That is the last row of the table above: 0/78 runs collapsed, and 84.91 held-out pass@1 at 696 steps (11 runs, 0/11 collapsed). Within that pool, an fp32 lm_head and a cast of bf16-head logits score the same (difference 0.04 [-0.52, +0.59], 11 vs 12 runs).

The 21/21 clean runs and the 85.44 / 85.25 values at 696 steps come from the second row instead, which keeps bf16 logits and draws only `u` in fp32. At 696 steps, the first 8 of those runs (85.44) and the 11 fp32-logit runs (84.91) differ by +0.53 [-0.02, +1.08]. That comparison is descriptive, because the two arms ran on different builds.

If the cast is removed from the generator, its logits go back to bf16, and so does the stock `u` (the first row) unless the backend draws `u` in fp32.

### 3. pi_old = generator logprobs, with the PPO clip

With an exact sampler, no importance reweighting is needed:

| pi_old (348 steps) | runs | held-out pass@1, steps 304-344 | grad-norm > 1 |
|---|---|---|---|
| generator logprobs + PPO clip (this config) | 23 | 83.94 ± 0.63 | 0/41 runs (pooled) |
| ratio == 1 (trainer logprobs, detached) | 13 | 83.84 ± 0.39 | 17/17 runs (pooled) |
| truncated importance sampling, min(pi/q, 2) | 9 | 83.79 ± 0.76 (-0.15 [-0.77, +0.47]) | 0/9 runs |

On the stock bf16 sampler, truncated importance sampling rescued 0 of 7 runs.

## Hyperparameters

| | value |
|---|---|
| model | Qwen3-0.6B, chat template, thinking disabled |
| data | `openai/gsm8k` "main", revision `740312a`, train split (7,473 problems) |
| prompt | question + `"\n\nPlease reason step by step, and put your final answer within \boxed{}."` |
| reward | 1 if the last `\boxed{}` equals the gold number, else 0 |
| rollout | 128 prompts x 16 samples per step, temperature 1.0, top_p 1.0, prompts up to 256 tokens, up to 1792 new tokens, rows of 2048 tokens |
| advantage | group mean baseline, no std normalization |
| loss | DAPO clip-higher, ratio clip [0.8, 1.28] |
| optimizer | AdamW, peak lr 2.5e-6, betas (0.9, 0.95), eps 1e-8, weight decay 0, grad-norm clip 1.0 |
| lr schedule | linear warmup over 35 steps, cosine decay to 0.1x over a 1170-step horizon, stopped at step 696 (lr then about 0.44x) |
| steps | 696 (348 also reported) |
| precision | fp32 master weights, bf16 compute; lm_head logits in fp32 |
| eval | 512 test problems, rows `(i * 1319) // 512` (`eval_subset`), 1 sample at temperature 0.7 |

The lr is an interior optimum among the four scales tried (348 steps, held-out pass@1, steps 304-344):

| peak lr | runs | held-out pass@1 | vs 1x |
|---|---|---|---|
| 0.5x (1.25e-6) | 5 | 82.55 ± 0.28 | -1.39 [-1.78, -1.01] |
| **1x (2.5e-6)** | 23 | **83.94 ± 0.63** | |
| 2x (5e-6) | 6 | 82.14 ± 0.91 | -1.80 [-2.76, -0.85] |
| 4x (1e-5) | 6 | 74.50 ± 2.52 | -9.44 [-12.08, -6.81] |

## Results

All runs: Qwen3-0.6B on one TPU v7x slice (64 chips), fp32 master weights, fp32 sampler `u`, pi_old = generator logprobs, peak lr 2.5e-6. Every row except the second kept bf16 logits and drew only `u` in fp32. The second row is the sampler this config runs: fp32 logits, fp32 arithmetic and fp32 `u`.

| run | runs | held-out pass@1 | collapsed |
|---|---|---|---|
| 696 steps (mean of the evals at steps 656-688) | 16 | **85.25** (85.44 ± 0.42 for the first 8; 85.06 for a fresh replicate of 8) | 0/16 |
| 696 steps, fp32 logits and arithmetic, fp32 `u` | 11 | 84.91 | 0/11 |
| 348 steps (mean of the evals at steps 304-344) | 10 | 84.30 ± 0.86 | 0/10 |
| 2 prompts per device (256 prompts x 16 per step), 348 steps | 8 | 84.72; +0.47 [-0.18, +1.11] vs a concurrent 1-prompt control: inconclusive | 0/8 |
| skip-solved prompt sampler + 2 prompts per device, 348 steps | 8 | 85.07; +0.55 [-0.23, +1.32] vs a concurrent 1-prompt control (84.52, 8 runs, 0/8 collapsed). This met the pre-registered criterion to advance to a confirmation (next row) | 0/8 |
| skip-solved prompt sampler + 2 prompts per device, fresh confirmation, 348 steps | 6 | 84.76 (strict 84.76): **not confirmed** against the pre-registered >= 85.00 bar. Across both sets (14 runs) the mean is 84.94, about +0.4 above the 1-prompt control, so >= 85 at 348 steps is not established | 0/6 |
| skip-solved prompt sampler alone (1 prompt per device), 348 steps | 8 | 84.06; -0.46 [-1.18, +0.26] vs the same control: inconclusive | 0/8 |
| any of the 2-prompt or skip-solved arms above, at 696 steps | - | **PENDING**: not yet run | - |

Values are mean ± SD across runs. CIs are Welch 95%, covering run-to-run noise only.

The skip-solved + 2-prompt gain should not be credited to skip-solved. In that arm the sampler did not engage as designed. The share of prompt groups with mixed rewards over steps 300-348 rose only from 0.20 to 0.29, short of the pre-registered +0.15 bar. Skip-solved alone did not help. So the gain is attributed mostly to the 2-prompts-per-device geometry. Neither change is in this config, which uses 128 prompts per step and no skip-solved sampler.

## Caveats

- **One model, one task.** Qwen3-0.6B on GSM8K, with one lr schedule. The bf16-master result depends on a small lr. At pretraining learning rates the ulp argument is weaker.
- **Eval subset used for decisions.** Every decision used the same 512 test problems, so they act as a validation set rather than a clean test set.
- **Lenient extractor.** Held-out answers were extracted leniently: `\boxed{}` first, then a "final answer" marker, then a trailing number in an answer context. This can only score a completion correct more often than a `\boxed{}`-only check would.
- **Builds.** Contrasts between different software builds are descriptive. The master-weight contrast above is within one build.

## How the numbers were measured

The runs used a standalone trainer: Hugging Face Qwen3 under FSDP2, plus an in-process TP=1 vLLM engine colocated on each of the slice's 128 TensorCore devices. Every rollout used the latest weights. It differs from this config in these ways:

| | measured trainer | this config |
|---|---|---|
| placement | trainer and generator colocated on every device | disjoint trainer and generator meshes |
| generator logits and sampler | headline runs (85.25, 85.44, 21/21 clean): bf16 lm_head logits, and a backend wrapper drew only the Gumbel `u` in fp32 | fp32 logits from `LMHeadCastConverter`, which is in the shared `model_spec`. The stock backend's `u` and sampler arithmetic are therefore fp32. Matching measured arm: 0/78 collapsed, 84.91 at 696 steps (11 runs) |
| loss normalization | per-sample token mean, then mean over samples; dual clip at 10x on negative advantages | token mean over all valid tokens; no dual clip |
| overlong shaping | `min(0, (1408 - len) / 256)` added to the reward before the advantage | none |
| zero-variance groups | kept (zero advantage) | dropped (`TrainingSampleBuilder.drop_zero_std_reward_groups`) |
| validation | sampled at temperature 0.7, lenient extractor | `ValidationConfig`: greedy, `\boxed{}` reward |
| warmup | lr 0 at step 1 | lr 1/35 of peak at step 1 (one step earlier); identical after warmup |

TODO: run this config end to end on a TPU slice and fill in its own results row.
