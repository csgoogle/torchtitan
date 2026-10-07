# Recipe for training Muse Glimmer 30B model on a 128 device TPU ironwood cluster (tpu7x-4x4x4)

These are instructions for deploying Muse Glimmer 30B training to a 128 device (64 chip) tpu7x cluster using Cluster Toolkit (`gcluster`).

For Cluster Toolkit (`gcluster`) installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE
```

## Notes

- **Devices vs chips**: an ironwood chip exposes two TensorCores, and torchtitan runs one process per core. `tpu7x-4x4x4` is therefore 128 devices on 64 chips, run as 16 hosts x 8 processes.
- **Fresh torch_tpu required**: the image must contain a torch_tpu built after
  google-pytorch/torch_tpu#4824 (2026-10-06), which provides
  `torch.tpu.distributed.spmd_safe`. Older builds log
  "Unable to find spmd_safe decorator" and are expected to regress performance.
- **Model settings follow upstream `muse_glimmer_30b`**: sequence 16384, local
  batch 1, AdamW lr 3e-4 with 200 warmup steps, 52 layers with the 3:1
  sliding-window-2048 (RoPE) / global (NoPE) pattern, GQA 32Q/2KV, soft-capped
  logits over a 202,048-token vocabulary.
- **GraphTrainer, Zero-3**: SimpleFSDP with parameters resharded after forward,
  full graph rematerialization, the optimizer compiled into the graph, and the
  `prefetch` FSDP schedule. Layers are not scanned (see "Scan" below).
- **Precision**: fp32 weights and AdamW state (default `training.dtype`), bf16
  forward/backward compute (`mixed_precision_param`), fp32 gradient
  reduce-scatter (`mixed_precision_reduce`).
- **Full logits by default**: at local batch 1 the `[16384, 202048]` logits
  are not what sets peak HBM, and plain cross entropy is ~2% faster than the
  fused kernel. `muse_glimmer_30b_fused_ce` runs the LM-head matmul, soft cap,
  and cross entropy as one Pallas kernel, one vocab tile at a time, so the
  logits are never materialized (the TPU counterpart of upstream's
  `ChunkedLossWrapper`). Use it once the logits dominate memory: larger local
  batch or longer sequences (at local batch 2 it saves ~36 GiB), the regular
  trainer (out of HBM with full logits), or when HBM headroom is needed.
- **Attention**: the JAX Pallas Splash Attention kernel
  (`jax.experimental.pallas.ops.tpu.splash_attention`, via
  `kernels/splash_attention.py`) with 1024 tiles: a local mask for the 39
  sliding-window layers and a causal mask for the 13 global ones. The tokamax
  Splash kernel (0.0.14) measured ~5% slower: its wrapper does not yet skip
  blocks that cross packed-document boundaries.
- **Config-carried flags**: exported before PJRT initialization, so the command
  below does not repeat them. Same libtpu set as `qwen3_32b`:
  - `xla_tpu_dvfs_p_state=7` (throughput power state)
  - `xla_tpu_scoped_vmem_limit_kib=65472`
  - `xla_tpu_enable_sparse_core_collective_aggregator=true`,
    `xla_tpu_enable_latency_hiding_layer_scheduler=true`
  - `xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false`,
    `xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true`,
    `xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true`
  - `xla_tpu_enable_diagnostic_coordinator=false` (TPU config default)
  - XLA flag `xla_enable_hlo_sharding_v3=false`

  "Default libtpu flags" in the table below means
  `--libtpu-init-flags=xla_tpu_enable_diagnostic_coordinator=false`, which
  replaces the set above with only the TPU config default; the XLA flag stays.
- **Tokenizer**: the default `hf_assets_path` is the test tokenizer in
  `./tests/assets/tokenizer`. Step time depends slightly on the data, so compare
  runs only with the same data and tokenizer.

## Training Command

```bash
export WORKLOAD_NAME="mg30b-tpu7x-4x4x4-$(date +%H%M%S)"

gcluster job submit \
  --name "$WORKLOAD_NAME" \
  --cluster "$CLUSTER_NAME" \
  --compute-type tpu7x \
  --topology 4x4x4 \
  --num-slices 1 \
  --location "$ZONE" \
  --project "$PROJECT_ID" \
  --image "$WORKLOAD_IMAGE" \
  --gke-disable-parallel-containers \
  --env="PYTHONUNBUFFERED=1" \
  --env="WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-main-job-0-0.${WORKLOAD_NAME}" \
  --env="WORKLOAD_NAME=$WORKLOAD_NAME" \
  --command "torchrun \
    --nnodes=16 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.muse_glimmer \
    --config=muse_glimmer_30b_full_logits \
    --training.steps=30"
```

## Expected Metrics

With this configuration (local batch 1, global batch 128, sequence 16384), you
should observe the following per-device metrics averaged over steps 10-30 (as
of 10/7/2026, measured with the command above, i.e. on the default `c4_test`
dataset and test tokenizer):

- **TPS (Tokens per second)**: ~2,913
- **TFLOPS**: ~505.3
- **MFU**: ~43.8%
- **Step Time (time_end_to_end)**: ~5.62s
- **Physical peak HBM**: 46.18GiB

Compile and initialization take about 5 minutes before the first step.

`grad_norm` is logged as 0.0 because the optimizer runs inside the compiled
graph; gradient clipping is still applied there.

## Alternative configurations

Measured the same way as above (10/7/2026, sequence 16384, steps 10-30, per
device):

| Configuration | Step time | TPS | MFU | HBM | Notes |
|---|---|---|---|---|---|
| `muse_glimmer_30b_full_logits` (this recipe) | ~5.62s | ~2,913 | ~43.8% | 46.18GiB | GraphTrainer, full logits |
| `muse_glimmer_30b_fused_ce` | ~5.75s | ~2,851 | ~42.9% | 46.12GiB | GraphTrainer, fused cross entropy |
| `reg_trainer_muse_glimmer_30b_fused_ce` | ~5.67s | ~2,889 | ~43.5% | — | Regular trainer (defer-and-fuse eager, per-layer compile); ~2.5 min startup; logs real `grad_norm`. Out of HBM at step 1 with full logits |
| `muse_glimmer_30b_full_logits`, local batch 2 | ~11.73s | ~2,794 | ~42.0% | 87.62GiB | `--training.num_tokens_per_microbatch_per_dp_rank=32768` |
| `muse_glimmer_30b_fused_ce`, local batch 2 | ~11.92s | ~2,748 | ~41.3% | 51.84GiB | |
| `muse_glimmer_30b_fused_ce`, tokamax Splash | ~6.05s | ~2,708 | ~40.7% | 46.17GiB | `--splash_attention_kernel.use_tokamax_splash_attention_kernel` |
| `muse_glimmer_30b_fused_ce`, default libtpu flags only | ~7.35s | ~2,229 | ~33.5% | 88.71GiB | The 4x4x4 flags are worth ~28% |

The regular trainer's memory column uses different accounting and is omitted.

## Scan

Layer scanning is not enabled yet. The 3:1 sliding-window/global layer pattern
means consecutive layers differ in masking and RoPE, so a single scanned layer
body does not apply directly.
