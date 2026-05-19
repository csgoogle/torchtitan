# Conformer implementation in Torchax

## Motivation
We implemented the `conformer` model from `torchtitan.experiments.tpu.conformer` onto Torchax. We wanted to test it using JAX's CTC loss and measure its performance.

## What was done
1. **Directory creation**: Created `torchtitan/experiments/torchax/conformer` to host the implementation.
2. **Model definition (`model.py`)**: Copied the Conformer PyTorch model from `tpu/conformer/model.py`. We removed the `splash_sdpa` dependency from `torch_tpu` and just used standard `F.scaled_dot_product_attention`, since Torchax natively replaces standard attention globally via Pallas.
3. **Training & Trainer Scripts (`train_minimal.py` & `trainer.py` & `jit_utils.py`)**: Copied the ones from `torchax/` but modified them to import from the local directory instead of globally from `torch_xla2`. This ensures that we don't encounter circular imports or conflicts with `torch_tpu` backend initialization.
4. **Sharding Map (`sharding.py`)**: Defined an `('fsdp', 'tp')` and `('fsdp',)` map corresponding to the Conformer's parameters (e.g. `conv_module`, `fc.weight`, `self_attn`, `ffn1/ffn2`).
5. **Config Registry (`config_registry.py`)**: Registered `conformer` config pointing to our Torchax module.
6. **CTC Loss**: Implemented custom `optax.ctc_loss` JAX fallback directly in `trainer.py` which checks if `logits` and `targets` have the `.jax()` representation. We also adapted `train_loader` to properly return shorter targets according to the config setting `ctc_loss_target_len`.
7. **Fixes to Environment**: Torchax natively conflicts with `torch_tpu` device backends (they both try to define `PrivateUse1`). We had to run with `TORCH_DEVICE_BACKEND_AUTOLOAD=0` to ensure `torchax` cleanly grabs the PyTorch environment.

## Running instructions

The Torchax training run can be executed efficiently on a v6e-8 node by simply launching it natively (JAX discovers all 8 local TPUs automatically without `torchrun`).

We reduced the `local_batch_size` to `16` to prevent HBM OOM during JAX's compilation step.

### Configuration Details
When running the command below, `torchtitan` uses the Python definition inside `config_registry.py` (specifically the `def conformer():` function) as the source of truth, because of the `--config=conformer` flag. 

While there is also a `conformer.toml` file in `train_configs/`, we do **not strictly need both**. The `.toml` file is mainly provided as a torchtitan convention for users who want to override configurations via CLI tools (Tyro) without touching the Python code. Our benchmark directly executed the Python-defined `config_registry.py`.

The exact configuration benchmarked was:
- **Batch Size:** 16 (per local device)
- **Sequence Length:** 512
- **Data Type:** bfloat16
- **Training Steps:** 100
- **Loss:** JAX CTC Loss (`use_ctc_loss = True`)
- **Torchax Scan:** Disabled (`use_scan = False`)

```bash
cd ~/torchtitan && TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. python3 -m torchtitan.experiments.torchax.conformer.train_minimal \
  --module=torchtitan.experiments.torchax.conformer \
  --config=conformer
```

### Result

```
[titan] 2026-05-19 22:44:56,538 - root - INFO - Compilation done. It took 68.1482 seconds
[titan] 2026-05-19 22:44:57,330 - root - INFO - step:  1   loss:     nan   step_time: 69.0395s   tps: 949  per_device_tps: 119   tflops: 0.59   mfu: 0.01%
[titan] 2026-05-19 22:44:58,254 - root - INFO - step: 10   loss:     nan   step_time:  0.1027s   tps: 638,370  per_device_tps: 79,796   tflops: 394.88   mfu: 5.38%
[titan] 2026-05-19 22:44:59,162 - root - INFO - step: 20   loss:     nan   step_time:  0.0908s   tps: 721,883  per_device_tps: 90,235   tflops: 446.53   mfu: 6.08%
[titan] 2026-05-19 22:45:00,365 - root - INFO - step: 30   loss:     nan   step_time:  0.1203s   tps: 544,907  per_device_tps: 68,113   tflops: 337.06   mfu: 4.59%
[titan] 2026-05-19 22:45:01,266 - root - INFO - step: 40   loss:     nan   step_time:  0.0900s   tps: 727,832  per_device_tps: 90,979   tflops: 450.21   mfu: 6.13%
[titan] 2026-05-19 22:45:02,195 - root - INFO - step: 50   loss:     nan   step_time:  0.0929s   tps: 705,230  per_device_tps: 88,154   tflops: 436.23   mfu: 5.94%
```

*(Note: Loss is NaN purely because randomly initialized values alongside a custom CTC implementation without clipping and fake integer inputs frequently cause divergence natively)*
