# Qwen3-30B-A3B on TPU7x 4x4x4

This recipe runs the FSDP-only Qwen3-30B-A3B benchmark on 128 TPU cores.
See the TPU [README](../../../../README.md) for XPK installation and cluster
setup.

The libtpu init flags this recipe needs live in the config
(`qwen3_moe_30b_tpu7x_4x4x4` in [`qwen3/config_registry.py`](../../../../qwen3/config_registry.py)),
which the trainer exports before PJRT initialization -- so the launch command
passes nothing but `--config`. The SparseCore collective aggregator is required
by the latency-hiding layer scheduler for this workload, so those flags move
together as one bundle.

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME
export WORKLOAD_IMAGE=YOUR_TORCHTITAN_IMAGE

xpk workload create \
--workload="$WORKLOAD_NAME" \
--cluster="$CLUSTER_NAME" \
--zone="$ZONE" \
--project="$PROJECT_ID" \
--tpu-type=tpu7x-4x4x4 \
--num-slices=1 \
--docker-image="$WORKLOAD_IMAGE" \
--env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
--command "\
    torchrun \
    --nnodes=16 \
    --nproc_per_node=8 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.qwen3 \
    --config=qwen3_moe_30b_tpu7x_4x4x4"
```

The scheduler is opt-in: run
`qwen3_moe_30b_fsdp_only_full_ac_benchmark` instead for the same TorchTitan
configuration without the flags, or override individual flags on the command
line, e.g. `--libtpu-init-flags.xla-tpu-dvfs-p-state 5`. The dual-SparseCore
reduce-scatter flag is an independent recipe choice and may need to be disabled
for schedules that overlap all-gather and reduce-scatter. The p-state flag is
also independent and selects the throughput-oriented TPU7x operating point.
Exporting `LIBTPU_INIT_ARGS` at launch still works and wins over the config for
any flag it names.

In matched 30-step TPU7x runs, enabling the pair improved throughput by about
5% at both GBS768/2K and GBS1024/2K. The runs retained FP32 FSDP reductions and
completed with finite loss and gradient norms on all 16 hosts.

## Benchmark Results
Run Date: 9/8/2026
* **MFU:** 12.76%
* **TPS (Tokens Per Second):** 7,121
* **TFLOPS:** 147.16
* **Memory Usage:** 65.33 GiB (68.05%)
* **Step Time (end-to-end):** 1.73s
