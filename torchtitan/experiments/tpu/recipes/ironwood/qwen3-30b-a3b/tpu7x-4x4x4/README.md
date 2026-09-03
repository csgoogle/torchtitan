# Qwen3-30B-A3B on TPU7x 4x4x4

This recipe runs the FSDP-only Qwen3-30B-A3B benchmark on 128 TPU cores.
See the TPU [README](../../../../README.md) for XPK installation and cluster
setup.

The `LIBTPU_INIT_ARGS` below must be set before PJRT initialization. The
SparseCore collective aggregator is required by the latency-hiding layer
scheduler for this workload, so keep the flags enabled or disabled together.

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE

# Let eligible reduce-scatter operations use both SparseCores. This is optimal
# for the current Zero-2 schedule, where backward has no parameter all-gathers
# competing with reduce-scatter. It is also optimal for the dependency-ordered
# Zero-3 interleave in PR #328, which cannot overlap its all-gathers and
# reduce-scatters. Re-evaluate this choice for Zero-3 prefetch or future
# schedules that can make those collectives concurrent.
# Request TPU7x performance state 7 for this throughput-oriented recipe. This
# trades higher device power for faster compute rather than changing the graph.
export LIBTPU_INIT_ARGS='--xla_tpu_enable_sparse_core_collective_aggregator=true \
    --xla_tpu_enable_latency_hiding_layer_scheduler=true \
    --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false \
    --xla_tpu_dvfs_p_state=7'

xpk workload create \
--workload="$WORKLOAD_NAME" \
--cluster="$CLUSTER_NAME" \
--zone="$ZONE" \
--project="$PROJECT_ID" \
--tpu-type=tpu7x-4x4x4 \
--num-slices=1 \
--docker-image="$WORKLOAD_IMAGE" \
--env LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS" \
--env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
--command "\
    torchrun \
    --nnodes=16 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.qwen3 \
    --config=qwen3_moe_30b_fsdp_only_full_ac_benchmark"
```

The scheduler is opt-in: omit the aggregator and scheduler flags to run the
same TorchTitan configuration without it. The dual-SparseCore reduce-scatter
flag is an independent recipe choice and may need to be disabled for schedules
that overlap all-gather and reduce-scatter. The p-state flag is also independent
and selects the throughput-oriented TPU7x operating point. These flags are
launcher-owned because libtpu reads them before the TorchTitan configuration is
constructed.

In matched 30-step TPU7x runs, enabling the pair improved throughput by about
5% at both GBS768/2K and GBS1024/2K. The runs retained FP32 FSDP reductions and
completed with finite loss and gradient norms on all 16 hosts.
