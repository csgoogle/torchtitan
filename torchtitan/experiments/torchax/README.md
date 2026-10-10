# torchax experiment

LLM training on TPU using [torchax](https://github.com/google/torchax) — a PyTorch-on-JAX
backend that lets you run PyTorch models on JAX devices with full XLA compilation.

## Overview

This experiment wraps the standard torchtitan training stack (config, dataloader,
models) with a torchax execution backend. Key features:

- **Single-process, multi-device**: JAX auto-discovers all TPU chips; no `torchrun` needed.
- **Scan-based layers**: `jax.lax.scan` over stacked transformer blocks — one compact XLA
  graph regardless of depth, dramatically reducing compile time.
- **Splash attention**: Pallas-based fused attention kernel for TPU.
- **FSDP + TP**: Parameters sharded via JAX named mesh (`fsdp`, `tp` axes).
- **CPU offload**: Optionally offload decoder layer inputs to host memory.
- **Models**: Llama3, Qwen3, DeepSeek-V3, Conformer.

## File layout

```
torchax/
├── train_minimal.py        # Entry point: parses flags, builds mesh, runs trainer
├── trainer.py              # TorchaxTrainer: model/optimizer/dataloader setup + train loop
├── torchax_job_config.py   # TorchaxConfig + TorchaxJobConfig dataclasses (tyro flags)
├── distributed.py          # JAX sharding utilities: apply_sharding_to_state, shard_input
├── splash_attn.py          # Builds splash attention callable from Pallas kernel
├── metrics.py              # JaxMetricsProcessor: TPS, TFLOPs, MFU logging
├── data_utils.py           # fake_dataloader, torch_loader_to_jax converter
├── gmm.py                  # Grouped matrix multiply (MoE)
├── moe.py                  # MoE utilities
├── jit_utils.py            # JIT helpers
├── llama3/                 # Llama3 model + sharding maps
├── qwen3/                  # Qwen3 model + sharding maps
└── deepseek_v3/            # DeepSeek-V3 model + sharding maps
```

## Quick start

```bash
python -m torchtitan.experiments.torchax.train_minimal \
    --model.name=llama3 \
    --model.flavor=8B \
    --training.dataset_path=tests/assets/c4_test \
    --model.hf_assets_path=tests/assets/tokenizer \
    --training.max_context_length=2048 \
    --training.num_tokens_per_train_step=8192 \
    --training.steps=10 \
    --torchax_config.use_scan \
    --activation_checkpoint.mode=full
```

Use `--training.dataset=fake` to skip the real dataset and use synthetic data.

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--torchax_config.use_scan` | False | Enable scan-based layers (smaller XLA graph) |
| `--torchax_config.tpu_megacore` | True | Enable megacore (v4/v5p) |
| `--torchax_config.model_layer_override` | None | Override n_layers for debugging |
| `--parallelism.tensor_parallel_degree` | 1 | TP degree (mesh tp axis size) |
| `--activation_checkpoint.mode` | full | `full`, `selective`, or `nothing` |
| `--training.enable_cpu_offload` | False | Offload decoder inputs to CPU |

## Parallelism

The mesh is shaped `(fsdp, tp)` where:
- `fsdp = num_devices // tp`
- `tp` is set by `--parallelism.tensor_parallel_degree`



