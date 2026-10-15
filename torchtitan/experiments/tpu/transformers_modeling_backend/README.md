# HF transformers backend on TPU

The TPU lane of the [HF transformers modeling backend](../../transformers_modeling_backend/README.md):
train a Hugging Face causal-LM checkpoint's architecture on TorchTPU with SimpleFSDP, without porting
the model. The architecture comes from the checkpoint's `config.json`; weights are random-init.

On top of the shared backend, this lane adds:

- **Attention** on the sbd splash kernel through HF's attention interface (`attention.py`): causal or
  bidirectional, grouped/multi-query KV without expansion, sliding windows, document masking from packed
  sequences, attention-logit soft-cap, and in-kernel dropout. Anything the kernel cannot express falls
  back to SDPA and is logged once per reason (outside Dynamo tracing, so run with `--compile.no-enable`
  to see the report).
- **Fused cross entropy** (Pallas), including the head's logit transform (Gemma soft-cap, Cohere scale,
  Granite divisor). Heads it cannot reproduce (a bias, a transform block) are refused.
- **Per-layer compile** and selective activation checkpointing that keeps the attention output.
- **MoE**: the HF MoE block is swapped for TorchTitan's grouped experts on TPU, with Pallas grouped
  matmuls and the TPU token dispatcher. No expert parallelism; SimpleFSDP shards the experts.

## Quick start

Single host (e.g. a v6e-4 VM), from the repo root, Qwen3-0.6B's architecture, 2 sequences of 2048
tokens per chip:

```bash
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.transformers_modeling_backend \
    --config=hf_full_max \
    --hf_model=Qwen/Qwen3-0.6B \
    --training.max_context_length=2048 \
    --training.num_tokens_per_microbatch_per_dp_rank=4096 \
    --training.steps=20
```

`--hf_model` takes a Hub id or a local directory containing `config.json` (only the config is read, so a
gated repo's config can be copied locally). The tokenizer defaults to the repo's test assets.

## Configs

Defined in `config_registry.py`.

| Config | Use |
|---|---|
| `hf_full_max` | Dense models: splash attention, fused CE, per-layer compile. The default choice. |
| `hf_full_max_plain_ce` | Same, with the model's own head and plain cross entropy (for heads the fused CE refuses). |
| `hf_moe_auto` | MoE checkpoints: the MoE swap with Pallas grouped matmuls and per-layer compile. |
| `hf_moe_auto_compile` / `hf_moe_auto_eager` | `hf_moe_auto` without the Pallas experts / without compile (A/B and debugging). |
| `hf_debugmodel`, `hf_debugmodel_max` | Two-layer debug models: a smoke test, and `hf_full_max`'s feature set at test scale. |

A MoE checkpoint run with a dense config keeps HF's own MoE block (used for experts the swap does not
reproduce, e.g. non-SiLU experts).

## Useful flags

| Flag | Effect |
|---|---|
| `--hf_config_overrides='{"num_hidden_layers": 4}'` | JSON overrides applied to the checkpoint's config. |
| `--compile.no-enable` | Eager mode (debugging, fallback reports). |
| `activation-checkpoint:full` | Full activation checkpointing, for models near the HBM limit. A subcommand: put it last. |
| `--loss.b_block_size=<n>` | Fused-CE token tile; must divide the per-rank token count (small batches). |

## Tests

| Test | What it checks |
|---|---|
| `tests/test_attention_masks.py` | The adapter's masks against HF's mask functions (CPU). |
| `tests/test_hf_reference_parity.py` | One training step on TPU against the raw HF model on CPU (loss and gradients). |
| `tests/test_hf_backend_distributed.py` | Multi-chip SimpleFSDP runs against a single-device CPU run, step for step. |
| `tests/test_pallas_experts.py` | The Pallas grouped-expert path of the MoE swap. |

Run on a TPU host with `python <test>.py --device=tpu` (absltest), or `pytest` for the CPU-only mask test.
