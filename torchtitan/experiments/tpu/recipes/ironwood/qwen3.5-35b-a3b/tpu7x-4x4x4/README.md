# Qwen3.5-35B-A3B on TPU7x 4x4x4

This recipe runs the FSDP-only Qwen3.5-35B-A3B hybrid MoE benchmark on 128 TPU
cores (16 hosts × 8 chips/host).
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
See the TPU [README](../../../../README.md) for XPK installation and cluster
setup.

Qwen3.5-35B-A3B is a 40-layer hybrid decoder combining 30 `GatedDeltaNet`
linear-attention layers with 10 `Qwen35Attention` full-attention layers (16
query heads, 2 KV heads, partial RoPE over 64 of 256 head dims) and 256 routed
experts (top-8 routing) plus 1 shared expert per layer.

The libtpu init flags this recipe needs ride on the torchtitan config
(`--libtpu-init-flags`), which exports them before PJRT initialization. The
SparseCore collective aggregator is required by the latency-hiding layer
scheduler for this workload, so keep those two enabled or disabled together.

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

# The nightly TorchTitan TPU container, built and validated by
# `.github/workflows/nightly-torchtitan-image.yml`. Note the region is
# `us-west1`: the `tpu-pytorch` project has no `us-central1` copy of this
# repository. If you are running unsubmitted Qwen3.5 changes, build and push
# your own image instead, per experiments/tpu/README.md, since the nightly
# image only contains code that has landed upstream.
export WORKLOAD_IMAGE=us-west1-docker.pkg.dev/tpu-pytorch/torchtitan-images/torchtitan-container:latest

# Let eligible reduce-scatter operations use both SparseCores. This is optimal
# for the Zero-2 FSDP schedule (`fsdp_reshard_after_forward="never"`), where
# backward has no parameter all-gathers competing with reduce-scatter.
# Request TPU7x performance state 7 for this throughput-oriented recipe.

LIBTPU_INIT_FLAGS="xla_tpu_enable_sparse_core_collective_aggregator=true \
xla_tpu_enable_latency_hiding_layer_scheduler=true \
xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false \
xla_tpu_dvfs_p_state=7"

xpk workload create \
--workload="$WORKLOAD_NAME" \
--cluster="$CLUSTER_NAME" \
--zone="$ZONE" \
--project="$PROJECT_ID" \
--tpu-type=tpu7x-4x4x4 \
--num-slices=1 \
--docker-image="$WORKLOAD_IMAGE" \
--env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
--command "\
    torchrun \
    --nnodes=16 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.qwen3_5 \
    --config=qwen35_moe_35b_fsdp_only_full_ac_benchmark \
    --libtpu-init-flags='$LIBTPU_INIT_FLAGS'"
```

## Fast 4-Layer Proxy Smoke Test (Optional)

Before running the full 40-layer compile on 128 chips, you can run the 4-layer
proxy (`qwen35_moe_35b_proxy`), which uses the exact same 128-chip sharding,
`TpuGraphTrainer` (`aot_fx_trace`), and `global_batch_size=768`
(`local_batch_size=6`), but truncates depth to 4 layers (3 `GatedDeltaNet` + 1
full attention) for rapid cluster and rendezvous verification:

```bash
--config=qwen35_moe_35b_proxy
```

## Benchmark Results

* **Run Date:** 9/18/2026
* **MFU:** 3.16%
* **TPS:** 1,954 (250,112 tokens/s cluster-wide)
* **TFLOPS:** 36.50
* **Memory Usage:** 88.25 GiB (91.93%)
* **Step Time:** 6.29s (steady state steps 10–30; step 30 `loss=2.77057`,
  `grad_norm=0.7211`)

## Architecture & Stability Notes

* **Full-Graph Tracing (`TpuGraphTrainer`)**: Uses `compile_mode="aot_fx_trace"`,
  `graph_trainer_memory_policy="full"`,
  `enable_graph_trainer_remat_schedule=True`, `compile_grad_norm=True`,
  `mixed_precision_reduce="float32"`, and
  `fsdp_allreduce_slice_parameter_patterns=("layers.*.attn.wo.weight",)` with
  hybrid FSDP layer bucketing (`delta_net`, `attention`, and `moe`).
* **Fused Pallas Cross-Entropy (`PallasCrossEntropyLoss`)**: Uses
  `PallasCrossEntropyLoss.Config()` with
  `torchtitan.experiments.tpu.overrides.fused_loss` to eliminate the
  materialized `[6, 2048, 248320]` logits tensor and its backward intermediates,
  reducing peak HBM from `114.17 GiB` (OOM) to `88.25 GiB` (`91.93%`).
* **Splash Attention (`SplashAttentionKernelConfig`)**: Enables Pallas Splash
  Attention across the 10 full-attention layers (`SEQ_MINOR`,
  `use_vmap_bwd=True`, `sa_use_document_aware_block_skipping=False`). Backward
  `*_dkv` tiles (`sa_block_dkv`, `sa_block_q_dkv`, `sa_block_kv_dkv`) are set to
  `1024` (halved from Qwen3-30B's `2048`) because Qwen3.5 has `head_dim=256`
  (2× Qwen3-30B's `128`), and `2048` overflows the 33 MiB per-kernel scoped
  VMEM limit.
* **Pallas Grouped-Matmul Experts (`TpuGroupedExperts`)**: Uses
  `TpuTokenDispatcherConverter` (all-to-all token dispatcher) and
  `TpuGroupedExpertsConverter` with `use_pallas_grouped_mm=True`,
  `use_pallas_paired_grouped_mm=True`, `pallas_grouped_mm_tile_m=256`,
  `pallas_grouped_mm_tgmm_tiling=(256, 0, 0)`, and
  `use_kernel_native_weight_layout=True`.
* **Chunkwise Linear Attention (`tpu_chunked`)**: `GatedDeltaNet` uses
  chunkwise-parallel recurrence (`chunk_size=64`) with explicit IEEE `float32`
  VPU unit-lower-triangular forward substitution
  (`_solve_unit_lower_triangular`) to maintain finite gradients across all 40
  layers without custom-call partitioner failures.
* **Batch Size**: Configured for `local_batch_size=6` (`global_batch_size=768`
  across 128 chips at `seq_len=2048`).
