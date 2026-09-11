# Recipe for training Llama3-8B model on 256 TPU trillium cluster (v6e-256)

These are instructions for deploying Llama3-8B training to a 256 chip v6e cluster using `xpk`.

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
- Dataset: `c4_test` (synthetic/test dataset).
- Tokenizer: `tests/assets/tokenizer`.
- Model configuration used is `llama3_8b_v6e_256` from [`llama3/config_registry.py`](../../../../llama3/config_registry.py).
  It carries the training flags and the libtpu init flags
  (`--xla_tpu_scoped_vmem_limit_kib=131072`), which the trainer exports before
  PJRT initialization -- so the launch command passes nothing but `--config`.

## Training Configuration Details

We are running with the following key configurations, all built into the config:
- **Sequence Length**: 8192
- **Local Batch Size**: 4 (Global batch size = 4 * 256 = 1024)
- **Torch Compile**: Enabled for `model` component.
- **Simple FSDP**: Enabled.
- **Splash Attention**: Enabled with block size 2048.
- **Loss Kernel**: Enabled.
- **Eager Mode**: `DEFER_AND_FUSE` (for non-compiled parts if any).

## Training Command

```bash
export WORKLOAD_NAME="llama3-8b-v6e256-$(date +%Y%m%d-%H%M%S)"

xpk workload create \
  --workload="$WORKLOAD_NAME" \
  --cluster="$CLUSTER_NAME" \
  --tpu-type="v6e-256" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --base-docker-image="us-west1-docker.pkg.dev/tpu-pytorch/torchtitan-images/torchtitan-container:latest" \
  --script-dir="$SCRIPT_DIR" \
  --env="WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-slice-job-0-0.${WORKLOAD_NAME}" \
  --env="WORKLOAD_NAME=$WORKLOAD_NAME" \
  --command="torchrun \
    --nnodes=64 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.llama3 \
    --config=llama3_8b_v6e_256"
```

## Expected Metrics

With this configuration you should observe the following metrics at step 15:

- TPS/chip: **6,765**
- MFU: **42.68%**
- Total TPS (256 chips): 1,731,840
