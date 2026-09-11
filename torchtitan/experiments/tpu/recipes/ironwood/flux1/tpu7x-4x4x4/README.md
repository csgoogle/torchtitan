# Recipe for training FLUX.1-dev model on 64 TPU ironwood cluster (v7x-64)

These are instructions for deploying FLUX.1-dev training (all 12B parameters trained) to an 64 chip v7x cluster using `xpk`.

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
For `xpk` installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export REPOSITORY=YOUR_REPOSITORY # Repository containing docker image
```

## SimpleFSDP + SplashAttention + torch.compile

Configs: `flux_dev_tpu7x_4x4x4_float32` and `flux_dev_tpu7x_4x4x4_bfloat16` in
[`flux/config_registry.py`](../../../../flux/config_registry.py). Each carries
the whole recipe, including the libtpu init flags -- `--xla_tpu_scoped_vmem_limit_kib=65536` --
which the trainer exports before PJRT initialization, so the launch command
passes nothing but `--config`.

### float32 + MixedPrecision bfloat16

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

xpk workload create \
  --workload="$WORKLOAD_NAME" \
  --cluster="$CLUSTER_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT_ID" \
  --tpu-type="tpu7x-4x4x4" \
  --num-slices=1 \
  --docker-image=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest \
  --env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
  --command "\
    torchrun \
    --nnodes=16 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.flux \
    --config=flux_dev_tpu7x_4x4x4_float32"
```

**6/11/26: With this configuration you should observe the following metrics**

- Average Step time: **2.50s**
- Average Step time / PDB: **0.83s**
- Average MFU: **21.01%**

### bfloat16

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

xpk workload create \
  --workload="$WORKLOAD_NAME" \
  --cluster="$CLUSTER_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT_ID" \
  --tpu-type="tpu7x-4x4x4" \
  --num-slices=1 \
  --docker-image=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest \
  --env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
  --command "\
    torchrun \
    --nnodes=16 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.flux \
    --config=flux_dev_tpu7x_4x4x4_bfloat16"
```

**6/11/26: With this configuration you should observe the following metrics**

- Average Step time: **4.00s**
- Average Step time / PDB: **0.80s**
- Average MFU: **21.80%**
