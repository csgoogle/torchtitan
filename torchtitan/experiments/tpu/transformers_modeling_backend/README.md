# HF transformers backend on TPU

Train decoder-only text models from the Hugging Face Hub on Cloud TPU with TorchTPU, without porting
the model. This is the TPU lane of the
[HF transformers modeling backend](../../transformers_modeling_backend/README.md): the model is built
from the checkpoint's `config.json` (weights are random-init), sharded with SimpleFSDP, and runs with
TPU kernels swapped in underneath HF's modeling code.

> **Status: experimental.** Like the upstream HF backend, this is backend coverage, not a set of
> recommended production training recipes. It is meant for bring-up, coverage and performance work on
> TPU: there is no expert parallelism yet, MoE models are bound by expert-weight communication, and
> some architectures fall back to slower paths. Check loss parity before relying on a new architecture.

## Coverage

Tested on 500+ of the most-downloaded text-generation checkpoints on the Hub, across multiple
architectures: Llama, Qwen, Mistral/Mixtral, Gemma, Phi, OLMo/OLMoE, Granite, DeepSeek, Falcon-H1, LFM2,
Mamba and more. Models up to 10B parameters were tested at full size on v6e-8, 10-35B at full size on
v6e-32, and larger ones at a reduced debug size (the architecture, not the full model).

## Optimizations enabled out of the box

With `hf_full_max` (dense) or `hf_moe_auto` (MoE), no flags needed:

- **Splash attention** on every HF architecture that uses the standard attention path, through HF's
  attention interface: document-aware block skipping for packed sequences, grouped/multi-query KV
  without expansion, sliding windows, attention-logit soft-cap, and in-kernel dropout. Anything the
  kernel cannot express falls back to SDPA and is logged once per reason.
- **Fused Pallas cross entropy** that never materializes the full logits, including the head's logit
  transform (Gemma soft-cap, Cohere scale, Granite divisor).
- **Per-layer `torch.compile`**, plus **selective activation checkpointing** that keeps the attention
  output instead of recomputing it.
- **SimpleFSDP** over all chips.
- **MoE**: the HF MoE block is swapped for TorchTitan's grouped experts, with **Pallas grouped
  matmuls** and a **TPU token dispatcher** whose backward gathers whole rows (3.6x to 65x faster than
  the standard dispatcher at equal batch on DeepSeek-V2-Lite, Qwen3-30B-A3B and OLMoE).

## Quick start

From the repo root on a single host (e.g. v6e-4):

```bash
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.transformers_modeling_backend \
    --config=hf_full_max \
    --hf_model=Qwen/Qwen3-0.6B \
    --training.max_context_length=2048 \
    --training.num_tokens_per_microbatch_per_dp_rank=4096 \
    --training.steps=20
```

`--hf_model` takes a Hub id or a local directory with a `config.json`. Use `hf_moe_auto` for MoE
checkpoints, `hf_full_max_plain_ce` for heads the fused CE refuses (e.g. a head bias), and add
`activation-checkpoint:full` (last on the command line) for models near the HBM limit.
