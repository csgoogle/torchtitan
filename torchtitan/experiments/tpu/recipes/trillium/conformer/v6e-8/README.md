# Recipe for training Conformer model on 8 TPU trillium cluster (v6e-8)

These are instructions for deploying Conformer training to an 8 chip v6e cluster using `xpk`.

For `xpk` installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export REPOSITORY=YOUR_REPOSITORY # Repository containing docker image
```

## Notes as of 5/21/26
- `--tpu_config.use_simple_fsdp` is enabled.
- `--compile.enable` is enabled with `--tpu_config.compile_mode=layer`.

## DDP with torch.compile

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

xpk workload create \
--workload=$WORKLOAD_NAME \
--cluster=$CLUSTER_NAME \
--zone=$ZONE \
--project=$PROJECT_ID \
--tpu-type="v6e-8" \
--num-slices=1 \
--docker-image=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest \
--env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
--command "
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.conformer.train_minimal \
    --module=torchtitan.experiments.tpu.conformer \
    --config=conformer_test \
    --parallelism.data_parallel_shard_degree=1 \
    --parallelism.data_parallel_replicate_degree=-1 \
    --activation_checkpoint.mode=full \
    --training.local_batch_size=1024 \
    --tpu_config.use_simple_fsdp \
    --tpu_config.compile_mode=layer \
    --splash_attention_kernel.use_splash_attention_kernel \
    --metrics.log_freq=1 \
    --conformer.use-ctc-loss \
    --compile.enable \
    --dataloader.dataset=random \
    --training.steps=20
"
```

**5/21/26: With this configuration you should observe approximately the following metrics**

- Average TPS/chip: **90,300**
- Average MFU: **6.09%**
- Total TPS (8 chips): **722,700**



## DDP eager mode

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

xpk workload create \
--workload=$WORKLOAD_NAME \
--cluster=$CLUSTER_NAME \
--zone=$ZONE \
--project=$PROJECT_ID \
--tpu-type="v6e-8" \
--num-slices=1 \
--docker-image=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest \
--env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
--command "
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.conformer.train_minimal \
    --module=torchtitan.experiments.tpu.conformer \
    --config=conformer_test \
    --parallelism.data_parallel_shard_degree=1 \
    --parallelism.data_parallel_replicate_degree=-1 \
    --activation_checkpoint.mode=full \
    --training.local_batch_size=1024 \
    --tpu_config.use_simple_fsdp \
    --tpu_config.eager_mode=DEFER_AND_FUSE \
    --splash_attention_kernel.use_splash_attention_kernel \
    --metrics.log_freq=1 \
    --conformer.use-ctc-loss \
    --dataloader.dataset=random \
    --training.steps=20
"
```

**5/21/26: With this configuration you should observe approximately the following metrics**

- Average TPS/chip: **52,100**
- Average MFU: **3.51%**
- Total TPS (8 chips): **416,900**

