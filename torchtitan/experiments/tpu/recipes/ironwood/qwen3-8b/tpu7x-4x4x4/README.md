# Recipe for training Qwen3-8B model on a 128 device TPU ironwood cluster (tpu7x-4x4x4)

These are instructions for deploying Qwen3-8B training to a 128 device (64 chip) tpu7x cluster using `xpk`.

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

- **Devices vs chips**: an ironwood chip exposes two TensorCores, and torchtitan runs one process per core. `tpu7x-4x4x4` is therefore 128 devices on 64 chips, run as 16 hosts x 8 processes. 
- **SparseCore Offloading**: This configuration achieves high MFU by utilizing
  SparseCore collective offloading for FSDP, requested through the config's
  `--libtpu-init-flags` and exported before PJRT initialization.
- **Graph Capture**: The optimizer and backward pass are fully captured into a single XLA graph via `--tpu_config.compile_optimizer` and `--tpu_config.optimizer_in_graph`.
- **Splash Attention**: Q/K/V layouts and backward tiles are tuned for this
  model and topology.

## Training Command

```bash
export WORKLOAD_NAME="qwen3-8b-tpu7x-4x4x4-$(date +%Y%m%d-%H%M%S)"

xpk workload create \
  --workload="$WORKLOAD_NAME" \
  --cluster="$CLUSTER_NAME" \
  --tpu-type="tpu7x-4x4x4" \
  --num-slices=1 \
  --zone="$ZONE" \
  --project="$PROJECT_ID" \
  --base-docker-image="$TORCHTITAN_IMAGE" \
  --script-dir="$SCRIPT_DIR" \
  --env="PYTHONUNBUFFERED=1" \
  --env="WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-slice-job-0-0.${WORKLOAD_NAME}" \
  --env="WORKLOAD_NAME=$WORKLOAD_NAME" \
  --command="torchrun \
    --nnodes=16 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.qwen3 \
    --config=qwen3_8b \
    --training.local_batch_size=6 \
    --training.seq_len=2048 \
    --training.steps=30 \
    --training.mixed_precision_reduce=float32 \
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
    --tpu_config.use_graph_trainer \
    --tpu_config.compile_mode=aot_fx_trace \
    --tpu_config.compile_optimizer \
    --tpu_config.optimizer_in_graph \
    --libtpu-init-flags='xla_tpu_dvfs_p_state=7 xla_tpu_scoped_vmem_limit_kib=65472 xla_tpu_enable_sparse_core_collective_aggregator=true xla_tpu_enable_latency_hiding_layer_scheduler=true xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false'"
```

## Expected Metrics

With this configuration, you should observe the following metrics around step 30 (as of 9/16/2026):

- **TPS (Tokens per second)**: ~12,389
- **TFLOPS**: ~585.01
- **MFU**: ~50.72%
- **Step Time (time_end_to_end)**: ~0.992s
- **Physical peak HBM (XProf)**: 86.81GiB

## Alternative configurations

The Qwen3 TPU module also provides two tuned 8B configurations. Both use the
same model, batch shape, optimizer, and attention implementation as
`qwen3_8b`.

| Configuration | Purpose | Main differences from `qwen3_8b` |
|---|---|---|
| `qwen3_8b_zero3` | Memory-oriented configuration | ZeRO-3, full rematerialization, interleaved FSDP scheduling, and tuned Splash policy |
| `qwen3_8b_maxtext_parity` | MaxText policy comparison | Tuned ZeRO-3 plus BF16 FSDP reductions and BF16 RoPE; scan remains disabled |

Keep canonical FP32-reduction results separate from MaxText-policy results.
The latter deliberately changes reduction precision and RoPE arithmetic.
