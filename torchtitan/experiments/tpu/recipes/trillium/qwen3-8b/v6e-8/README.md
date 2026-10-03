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
- **Memory/Barrier Tuning**: `--libtpu-init-flags='xla_tpu_use_enhanced_launch_barrier=false xla_tpu_scoped_vmem_limit_kib=65536'`. These are libtpu init flags,
  carried by the torchtitan config and exported before PJRT initialization
  rather than set by the launcher.
- **Dataset**: `c4_test` (synthetic/test dataset)

## Training Configuration Details
We are running with the following key optimizations built into the configuration:
- **Sequence Length**: 2048 (passed as `--training.max_context_length=2048`)
- **Local Batch Size**: 6 sequences per data-parallel rank (Global batch size =
  6 * 8 = 48 sequences). Batch size is expressed in tokens on the command line:
  `--training.num_tokens_per_microbatch_per_dp_rank=12288` (6 sequences * 2048
  tokens).
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
    --config=reg_trainer_qwen3_8b \
    --training.num_tokens_per_microbatch_per_dp_rank=12288 \
    --training.max_context_length=2048 \
    --training.mixed_precision_reduce=float32 \
    --compile.enable \
    --parallelism.use_simple_fsdp \
    --parallelism.enable_tpu_topo_mesh \
    --splash_attention_kernel.use_splash_attention_kernel \
    --splash_attention_kernel.sa_block_q=1024 \
    --splash_attention_kernel.sa_block_kv=1024 \
    --splash_attention_kernel.sa_block_dkv=1024 \
    --splash_attention_kernel.sa_block_kv_compute=1024 \
    --splash_attention_kernel.sa_block_q_dkv=1024 \
    --splash_attention_kernel.sa_block_kv_dkv=1024 \
    --splash_attention_kernel.sa_block_kv_dkv_compute=1024 \
    --splash_attention_kernel.sa_block_q_dq=1024 \
    --splash_attention_kernel.sa_block_kv_dq=1024 \
    --loss_kernel.use_loss_kernel \
    --tpu_config.eager_mode=DEFER_AND_FUSE \
    --tpu_config.use_graph_split \
    --tpu_config.compile_mode=layer \
    --libtpu-init-flags='xla_tpu_use_enhanced_launch_barrier=false xla_tpu_scoped_vmem_limit_kib=65536'"
```

## Expected Metrics

With this configuration, we observe the following at step 15 (run from 8/31/2026):

- **TPS (Tokens per second)**: 5,642
- **TFLOPS**: 276.65
- **MFU**: 30.14%
- **Step Time (time_end_to_end)**: 2.18s
- **Loss**: 7.82703
- **Grad Norm**: 31.0801
- **Memory**: 2.01GiB (6.28%)
