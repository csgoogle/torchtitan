# Recipe for training Llama3-8B model on 8 TPU trillium cluster (v6e-8)

These are instructions for deploying Llama3-8B training to an 8 chip v6e cluster using `xpk`.

For `xpk` installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export SCRIPT_DIR=YOUR_SCRIPT_DIR # Path to your torchtitan directory
```

## Notes

- **Master weights are float32** (config_registry default). Compute runs in bf16 via `mixed_precision_param=bfloat16`.
- VMEM tuning: `--libtpu-init-flags xla_tpu_scoped_vmem_limit_kib=131072`, a
  libtpu init flag the config exports before PJRT initialization.
- Dataset: `c4_test` (synthetic/test dataset).
- Tokenizer: `tests/assets/tokenizer`.
- Model configuration used is `llama3_8b` from `config_registry.py`.

## Training Configuration Details

We are running with the following key configurations (explicitly passed as flags):
- **Sequence Length**: 8192
- **Local Batch Size**: 2 (Global batch size = 2 * 8 = 16)
- **Torch Compile**: Enabled for `model` component.
- **Simple FSDP**: Enabled.
- **Splash Attention**: Enabled with block size 2048.
- **Loss Kernel**: Enabled.
- **Eager Mode**: `DEFER_AND_FUSE` (for non-compiled parts if any).

## Training Command

```bash
export WORKLOAD_NAME="llama3-8b-v6e8-$(date +%Y%m%d-%H%M%S)"

xpk workload create \
  --workload="$WORKLOAD_NAME" \
  --cluster="$CLUSTER_NAME" \
  --tpu-type="v6e-8" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --base-docker-image="us-west1-docker.pkg.dev/tpu-pytorch/torchtitan-images/torchtitan-container:latest" \
  --script-dir="$SCRIPT_DIR" \
  --env="WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-slice-job-0-0.${WORKLOAD_NAME}" \
  --env="WORKLOAD_NAME=$WORKLOAD_NAME" \
  --command="torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.llama3 \
    --config=llama3_8b \
    --hf_assets_path=tests/assets/tokenizer \
    --dataloader.dataset_path=tests/assets/c4_test \
    --training.steps=20 \
    --training.local_batch_size=2 \
    --training.seq_len=8192 \
    --compile.enable \
    --parallelism.use_simple_fsdp \
    --splash_attention_kernel.use_splash_attention_kernel \
    --splash_attention_kernel.sa_block_kv_compute=2048 \
    --loss_kernel.use_loss_kernel \
    --tpu_config.eager_mode=DEFER_AND_FUSE \
    --libtpu-init-flags xla_tpu_scoped_vmem_limit_kib=131072"
```

## Expected Metrics

With this configuration you should observe the following metrics at step 15:

- TPS/chip: **6,139**
- MFU: **38.73%**
- Total TPS (8 chips): 49,112
