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

- **Model**: upstream `muse_glimmer_30b` settings (sequence 16384, local batch
  1, AdamW lr 3e-4, 200 warmup steps).
- **Parallelism**: GraphTrainer with SimpleFSDP Zero-3, the `prefetch` FSDP
  schedule, full rematerialization, and the optimizer compiled into the graph.
- **Precision**: fp32 weights and optimizer state, bf16 compute.
- **Kernels**: JAX Pallas Splash attention (1024 tiles) with a local mask for
  the sliding-window layers, and plain cross entropy over the full logits.
  `muse_glimmer_30b_fused_ce` swaps in the fused soft-capped cross-entropy
  kernel, which is ~2% slower here but saves memory when raising tokens per
  microbatch.
- **Flags**: the tpu7x-4x4x4 libtpu flags (p-state 7, scoped VMEM, SparseCore
  latency-hiding scheduler and hierarchical reduce-scatter) are carried on the
  config, the same set as `qwen3_32b`.

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

`grad_norm` is logged as 0.0 because the optimizer runs inside the compiled
graph; gradient clipping is still applied there.
