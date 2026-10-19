# Qwen3.5-35B-A3B on Ironwood (TPU7x)

Qwen3.5-35B-A3B (text only): 40 layers in a 3:1 cycle of `GatedDeltaNet`
(linear attention) and full-attention layers, 256 routed experts (top-8) plus a
gated shared expert in every layer; 34.66B parameters, 2.95B active per token.
`qwen35_moe_35b_split_tokamax_benchmark` trains it on any TPU7x slice from one
`tpu7x-2x2x1` host (4 chips, 8 TensorCore ranks) to `tpu7x-4x4x4` (64 chips,
128 ranks). `qwen35_moe_35b_proxy_split_tokamax_benchmark` is its 4-layer proxy
(one 3:1 cycle, same layer shapes) for quick single-host runs.

## Launch

One `tpu7x-2x2x1` host:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m torchtitan.experiments.tpu.train \
  --module=torchtitan.experiments.tpu.qwen3_5 \
  --config=qwen35_moe_35b_split_tokamax_benchmark \
  --hf_assets_path=tests/assets/tokenizer \
  --dataloader.dataset=c4_test \
  --dataloader.dataset_path=tests/assets/c4_test
```

`tpu7x-4x4x4` (16 hosts; see the TPU [README](../../../../README.md) for
cluster setup), with 64K tokens per rank:

```bash
torchrun --nnodes=16 --nproc_per_node=8 \
  --rdzv_backend=static --rdzv_endpoint=$WORKERS_0_HOSTNAME:29501 \
  --node_rank=$TPU_WORKER_ID \
  -m torchtitan.experiments.tpu.train \
  --module=torchtitan.experiments.tpu.qwen3_5 \
  --config=qwen35_moe_35b_split_tokamax_benchmark \
  --training.num_tokens_per_microbatch_per_dp_rank=65536 \
  --hf_assets_path=tests/assets/tokenizer \
  --dataloader.dataset=c4_test \
  --dataloader.dataset_path=tests/assets/c4_test
```

The image needs a Tokamax build with the Kimi Delta Attention op
(`tokamax._src.ops.kda`, [openxla/tokamax](https://github.com/openxla/tokamax)
HEAD): the tokamax wheels up to 0.1.0 do not have it.

## Recipe

* **Precision and sharding**: bf16 parameters, gradients and AdamW moments (8
  bytes per parameter, 258 GiB in total), ZeRO-3 SimpleFSDP
  (`fsdp_reshard_after_forward=always`) with full rematerialization.
* **GraphTrainer with graph split**: as one step graph, the 40-layer model
  neither compiles within an hour on 8 host CPUs nor fits in HBM on one host
  (XLA keeps most layers' unsharded expert gradients alive until late in the
  step). Per-layer executables bound both, and repeated layers hit the
  compilation cache (first step after 4-6 minutes on 224-CPU GKE hosts). Two
  layers per forward and backward executable (each pairing a full-attention
  layer with a `GatedDeltaNet` layer) with `backward_partition_mode=
  fsdp_pipeline`: 3.51 s/step vs 4.65 s/step with one layer per executable
  (one host, 32K tokens per rank).
* **AdamW**: torch_tpu's fused AdamW outside the step graph
  (`optimizer_in_graph=False`, `compile_grad_norm=True`).
* **`GatedDeltaNet`**: Tokamax's Mosaic TPU KDA kernels
  (`TpuGatedDeltaKernel.Config.backend="tokamax"`), with the forward returning
  the backward kernels' residuals; the causal convolution and its input
  gradient are computed from shifted slices.
* **MoE**: Pallas paired grouped matmuls with kernel-native expert weights
  and the fused route-score backward.
* **Attention**: Splash attention with segment ids from `positions`; 1024
  blocks keep `head_dim=256` within the scoped VMEM limit.
* **Loss**: Pallas fused linear cross-entropy (no `[tokens, 248320]` logits in
  HBM).
* **Logging** every 5 steps: a per-step `loss.item()` keeps the host from
  dispatching the next step (+28-36% on a 12-layer proxy).
* **libtpu flags**: `xla_tpu_dvfs_p_state=7` (throughput operating point);
  the SparseCore collective aggregator with the latency-hiding layer scheduler
  (overlaps the FSDP collectives with compute); SparseCore-offloaded async
  all-gathers; reduce-scatters on both SparseCores; and
  `xla_tpu_scoped_vmem_limit_kib=65472` for the Pallas kernels.

## Benchmark Results

Full 40-layer model on GKE TPU7x slices, sequence length 2048, 20 steps
(the numbers at step 20; steps 5-20 agree within 1%):

| Slice (ranks) | Tokens/rank | Step time | Tokens/s/rank | MFU | Loss @ steps 1/10/20 |
| --- | --- | --- | --- | --- | --- |
| `2x2x1` (8) | 32,768 | 3.54 s | 9,255 | 15.2% | 12.76 / 4.12 / 2.70 |
| `2x2x2` (16) | 32,768 | 3.59 s | 9,133 | 15.0% | 12.79 / 3.98 / 2.70 |
| `4x4x4` (128) | 32,768 | 4.25 s | 7,707 | 12.7% | 12.78 / 4.12 / 2.69 |
| `4x4x4` (128) | 65,536 | 6.89 s | 9,506 | **15.6%** | 12.78 / 4.19 / 2.81 |

At 128 ranks the ZeRO-3 all-gathers and reduce-scatters take longer than at
8-16 ranks; 64K tokens per rank hides them again.

## Expert Parallelism

`--parallelism.expert_parallel_degree=<EP>` (or
`qwen35_moe_35b_split_tokamax_ep_benchmark`, EP=4) shards the routed experts
over `EP` ranks and FSDP-shards them over the remaining `efsdp` ranks; all
other parameters stay sharded over `fsdp`:

* `TpuQwen35TokenDispatcher` dispatches and combines tokens with static-shape
  all-to-alls (no host syncs), which `TpuGraphTrainer` can trace. Each rank
  sends at most `ceil(1.25 * T * top_k / EP)` tokens to each other rank;
  tokens beyond that are dropped (zero expert output), unlike the dropless
  `EP=1` path.
* The experts use the unpaired Pallas grouped matmuls (`ExpertParallel` shards
  the expert weights on dim 0).
* The optimizer gets one param group per device mesh, and the gradient norm
  is computed eagerly over both meshes (`compile_grad_norm=False`).
* Graph split uses one layer per executable with `by_layer` backward
  executables (3 distinct backward executables to compile instead of 6).

Full 40-layer model, `4x4x4` (128 ranks), 32K tokens per rank:

| EP | Step time | Tokens/s/rank | MFU | Memory (XLA reservation) |
| --- | --- | --- | --- | --- |
| 1 | 4.25 s | 7,707 | 12.7% | 20.8 GiB |
| 4 | 9.61 s | 3,410 | 5.6% | 17.3 GiB |
| 8 | 8.61 s | 3,808 | 6.3% | 17.2 GiB |
| 16 | 10.46 s | 3,134 | 5.2% | 17.1 GiB |

At this model size EP is about 2x slower than `EP=1`: the all-to-alls, the
unpaired grouped matmuls on smaller per-expert groups and the one-layer
executables cost more than the smaller expert all-gathers save. It is meant
for scaling beyond what FSDP alone fits.

## Notes

* `GatedDeltaNet` runs one recurrence per `[B, S]` row: documents packed into
  a row share the causal convolution and the delta-rule state (there are no
  varlen `cu_seqlens` on TPU); attention masks documents via `positions`.
* The Tokamax kernels clamp the per-token log decay at -5
  (`tokamax_min_log_decay`): the kernel's sub-chunk factoring overflows below
  it, which produced NaN gradients with the model's initialization (log
  decays down to about -34).
* The auxiliary-loss-free expert-bias balancer is off
  (`load_balance_coeff=None`): HF checkpoints carry no expert biases.
