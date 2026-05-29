# Recipe for training FLUX.1-dev model on 8 TPU trillium cluster (v6e-8)

These are instructions for deploying FLUX.1-dev training (all 12B
parameters trained) to an 8 chip v6e cluster using `xpk`.

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

### float32 + MixedPrecision bfloat16

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

xpk workload create \
--workload="$WORKLOAD_NAME" \
--cluster="$CLUSTER_NAME" \
--zone="$ZONE" \
--project="$PROJECT_ID" \
--tpu-type="v6e-8" \
--num-slices=1 \
--docker-image=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest \
--env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
--env LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536" \
--command "\
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=flux \
    --config=flux_dev \
    --optimizer.implementation=foreach \
    --metrics.log_freq=1 \
    --training.steps=40 \
    --training.dtype=float32 \
    --training.local_batch_size=2 \
    --dataloader.img_size=1024 \
    --flux.random_dataset \
    --parallelism.use_simple_fsdp \
    --splash-attention-kernel.use-splash-attention-kernel \
    --compile.enable \
    --compile.backend=tpu \
    --activation-checkpoint.no-preserve_rng_state"
```

**5/28/26: With this configuration you should observe the following metrics**

- Average Step time: **2.44s**
- Average Step time / PDB: **1.22s**

### bfloat16

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

xpk workload create \
--workload="$WORKLOAD_NAME" \
--cluster="$CLUSTER_NAME" \
--zone="$ZONE" \
--project="$PROJECT_ID" \
--tpu-type="v6e-8" \
--num-slices=1 \
--docker-image=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest \
--env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
--env LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536" \
--command "\
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=flux \
    --config=flux_dev \
    --optimizer.implementation=foreach \
    --metrics.log_freq=1 \
    --training.steps=40 \
    --training.dtype=bfloat16 \
    --training.local_batch_size=3 \
    --dataloader.img_size=1024 \
    --flux.random_dataset \
    --parallelism.use_simple_fsdp \
    --splash-attention-kernel.use-splash-attention-kernel \
    --compile.enable \
    --compile.backend=tpu \
    --activation-checkpoint.no-preserve_rng_state"
```

**5/28/26: With this configuration you should observe the following metrics**

- Average Step time: **2.87s**
- Average Step time / PDB: **0.95s**
