# Recipe for training Qwen3-8B model on 8 TPU trillium cluster (v6e-8)

> **Note (8/30/2026):** This recipe uses the Regular Trainer configuration for Qwen3-8B. The Graph Trainer currently still encounters Out Of Memory (OOM) errors on v6e.

These are instructions for deploying Qwen3-8B training using the regular trainer to an 8 chip v6e cluster using `xpk`.

For `xpk` installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export SCRIPT_DIR=YOUR_SCRIPT_DIR # Path to your torchtitan directory
```

## Notes

- **Model Configuration**: `reg_trainer_qwen3_8b` from `config_registry.py`.
- **Memory/Barrier Tuning**: `LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_scoped_vmem_limit_kib=65536"`
- **Dataset**: `c4_test` (synthetic/test dataset)

## Training Configuration Details
We are running with the following key optimizations built into the configuration:
- **Sequence Length**: 2048
- **Local Batch Size**: 6 (Global batch size = 6 * 8 = 48)
- **Parallelism**: Simple FSDP with Topology Aware Mesh (`enable_tpu_topo_mesh=True`) mapped to the 2x4 ICI physical layout.
- **Activation Checkpointing**: FullAC (`SelectiveAC` causes OOM on v6e due to XLA activation limits).
- **Compile Mode**: Layer-by-layer compilation (`compile_mode="layer"`).
- **Precision**: `float32` reduction precision.
- **Loss Kernel & Splash Attention**: Both enabled (SA block sizes updated to 1024 for all blocks).

## Training Command
```bash
export WORKLOAD_NAME="qwen3-8b-v6e8-$(date +%Y%m%d-%H%M%S)"

xpk workload create \
  --workload="$WORKLOAD_NAME" \
  --cluster="$CLUSTER_NAME" \
  --tpu-type="v6e-8" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --base-docker-image="us-west1-docker.pkg.dev/tpu-pytorch/torchtitan-images/torchtitan-container:latest" \
  --script-dir="$SCRIPT_DIR" \
  --env="LIBTPU_INIT_ARGS=--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_scoped_vmem_limit_kib=65536" \
  --env="PYTHONUNBUFFERED=1" \
  --env="WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-slice-job-0-0.${WORKLOAD_NAME}" \
  --env="WORKLOAD_NAME=$WORKLOAD_NAME" \
  --command="export PYTHONPATH=/app:\$PYTHONPATH && cd /app && WORKERS=(${TPU_WORKER_HOSTNAMES//,/ }) && MASTER_ADDR=\$WORKERS_0_HOSTNAME && torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.qwen3 \
    --config=reg_trainer_qwen3_8b"
```

## Expected Metrics

With this configuration, you should observe the following metrics at step 15 (run from 8/31/2026):

- **TPS (Tokens per second)**: 5,642
- **TFLOPS**: 276.65
- **MFU**: 30.14%
- **Step Time (time_end_to_end)**: 2.18s
- **Loss**: 7.82703
- **Grad Norm**: 31.0801
- **Memory**: 2.01GiB (6.28%)
