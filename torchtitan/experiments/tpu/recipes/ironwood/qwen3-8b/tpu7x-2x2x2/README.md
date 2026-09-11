# Recipe for training Qwen3-8B model on a 16 device TPU ironwood cluster (tpu7x-2x2x2)

These are instructions for deploying Qwen3-8B training to a 16 device (8 chip) tpu7x cluster using `xpk`.

For `xpk` installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export TORCHTITAN_IMAGE=YOUR_TORCHTITAN_IMAGE # Artifact Registry repo holding torchtitan-container
export SCRIPT_DIR=YOUR_SCRIPT_DIR # Path to your torchtitan directory
```

## Notes

- **Devices vs chips**: an ironwood chip exposes two TensorCores, and torchtitan runs one process per core. `tpu7x-16` is therefore 16 devices on 8 chips, run as 2 hosts x 8 processes. 
- **Model Configuration**: `qwen3_8b_tpu7x` from [`qwen3/config_registry.py`](../../../../qwen3/config_registry.py).
  The config carries the training flags and the libtpu init flags, so the launch
  command passes nothing but `--config`.
  A name without a slice is shared with the other slices of the same
  hardware, which need no settings of their own.
- **SparseCore Offloading**: This configuration achieves high MFU by utilizing
  SparseCore collective offloading for FSDP, requested by the config's
  `libtpu_init_flags` (SparseCore collective aggregator, latency-hiding layer
  scheduler, dual-SparseCore reduce-scatter, TPU7x p-state 7 and a 65472 KiB
  vmem limit).
- **Graph Capture**: The optimizer and backward pass are fully captured into a single XLA graph.
- **Splash Attention**: Splash Attention block sizes are manually tuned to 1024 across the board for optimal tile sizing.

## Training Command

```bash
export WORKLOAD_NAME="qwen3-8b-tpu7x16-$(date +%Y%m%d-%H%M%S)"

xpk workload create \
  --workload="$WORKLOAD_NAME" \
  --cluster="$CLUSTER_NAME" \
  --tpu-type="tpu7x-16" \
  --num-slices=1 \
  --zone="$ZONE" \
  --project="$PROJECT_ID" \
  --base-docker-image="$TORCHTITAN_IMAGE" \
  --script-dir="$SCRIPT_DIR" \
  --env="PYTHONUNBUFFERED=1" \
  --env="WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-slice-job-0-0.${WORKLOAD_NAME}" \
  --env="WORKLOAD_NAME=$WORKLOAD_NAME" \
  --command="torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.qwen3 \
    --config=qwen3_8b_tpu7x"
```

## Expected Metrics

With this configuration, you should observe the following metrics around step 30 (as of 9/8/2026):

- **TPS (Tokens per second)**: ~10,971
- **TFLOPS**: ~518.07
- **MFU**: ~44.91%
- **Step Time (time_end_to_end)**: ~1.11s
- **Memory**: 78.47GiB (81.74%)
