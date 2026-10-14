# Recipe for training Llama3-8B model on 8 TPU trillium cluster (v6e-8)

These are instructions for deploying Llama3-8B training to an 8 chip v6e cluster using Cluster Toolkit (`gcluster`).

For Cluster Toolkit (`gcluster`) installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE
```

## Notes

- **Master weights are float32** (config_registry default). Compute runs in bf16 via `mixed_precision_param=bfloat16`.
- VMEM tuning: `--libtpu-init-flags=xla_tpu_scoped_vmem_limit_kib=131072`, a
  libtpu init flag the config exports before PJRT initialization.
- `tpu_program_stack_shrink_after_unload=false` keeps the TPU program stack at
  its high-water mark. HBM is nearly full in this recipe, so letting the stack
  shrink makes later program loads trigger HBM defragmentation stalls (~0.6%
  slower step time on libtpu 0.0.50).
- Dataset: `c4_test` (synthetic/test dataset).
- Tokenizer: `tests/assets/tokenizer`.
- Model configuration used is `llama3_8b` from `config_registry.py`.

## Training Configuration Details

We are running with the following key configurations (explicitly passed as flags):
- **Sequence Length**: 8192 (passed as `--training.max_context_length=8192`)
- **Local Batch Size**: 2 sequences per data-parallel rank (Global batch size =
  2 * 8 = 16 sequences). Batch size is expressed in tokens on the command line:
  `--training.num_tokens_per_microbatch_per_dp_rank=16384` (2 sequences * 8192
  tokens).
- **Torch Compile**: Enabled for `model` component.
- **Simple FSDP**: Enabled.
- **Splash Attention**: Enabled with block size 2048.
- **Loss Kernel**: Enabled.
- **Eager Mode**: `DEFER_AND_FUSE` (for non-compiled parts if any).

## Training Command

```bash
export WORKLOAD_NAME="llama3-8b-v6e8-$(date +%H%M%S)"

gcluster job submit \
  --name "$WORKLOAD_NAME" \
  --cluster "$CLUSTER_NAME" \
  --compute-type ct6e-standard-4t \
  --topology 2x4 \
  --num-slices 1 \
  --location "$ZONE" \
  --project "$PROJECT_ID" \
  --image "$WORKLOAD_IMAGE" \
  --env="WORKERS_0_HOSTNAME=${WORKLOAD_NAME}-main-job-0-0.${WORKLOAD_NAME}" \
  --env="WORKLOAD_NAME=$WORKLOAD_NAME" \
  --command "torchrun \
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
    --training.num_tokens_per_microbatch_per_dp_rank=16384 \
    --training.max_context_length=8192 \
    --compile.enable \
    --parallelism.use_simple_fsdp \
    --splash_attention_kernel.use_splash_attention_kernel \
    --splash_attention_kernel.sa_block_kv_compute=2048 \
    --tpu_config.eager_mode=DEFER_AND_FUSE \
    --libtpu-init-flags='xla_tpu_scoped_vmem_limit_kib=131072 tpu_program_stack_shrink_after_unload=false'"
```

## Expected Metrics

With this configuration you should observe the following metrics (as of 9/29/26):

- TPS/chip: **5,383**
- MFU: **33.96%**
- Total TPS (8 chips): 43,068
