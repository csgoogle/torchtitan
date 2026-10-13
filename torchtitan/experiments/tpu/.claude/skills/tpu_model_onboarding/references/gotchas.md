# Gotchas

Each entry is something that cost a previous onboarding a day or more. The
first line is the symptom; the second is the cause and the check.

## Silent wrong numerics (trains fine, wrong model)

- **Packed batch, no segment ids.** Loss looks normal, throughput is half of
  expected. The attention wrapper did not forward `positions` to the inner
  attention, so splash saw one long causal problem per row. Check the
  custom_call operands in HLO.
- **Sliding window not applied on the plain SDPA path**, or off by one under
  splash (Muse window W is splash window W - 1). Test with HF's definition:
  query q attends key k iff 0 <= q - k < W.
- **Config override builds the wrong shape.** `--config=<x> --hf-model=<y>`
  can build a hardcoded architecture for any checkpoint and every log line
  looks right. Assert per-layer replacement counts and heads / kv_heads /
  vocab against the checkpoint config.
- **Config patcher matched nothing.** A traverse that finds no module reads as
  "the knob does not help". Assert the match count.
- **Unit-initialized scales hide missing scaling.** Norm weights at 1.0 and
  learned scales at 1.0 make a dropped multiply invisible; perturb them in
  the parity test.
- **Fused kernels are not bit-identical to the compiled graph.** A kernel that
  is bit-exact in isolation can shift the step-1 loss at 1e-5 relative because
  whole-program layout changes reduction order. Label it rounding-order; do not
  claim bit-identity from an isolated test.
- **Empty expert groups.** `torch._grouped_mm` on libtpu < 0.0.48 left the
  weight gradient for empty groups uninitialized; one clean run proves nothing
  for this class.

## Loud failures with misleading messages

- **`init_device_mesh` empty process-group name** right after `import
  torch_tpu` succeeded: the wheel's torch glue does not match the image's torch
  nightly. Three-run bisect: new image, base image, base image + new source.
- **Dynamo marks a dim symbolic on the second layer type** and the TPU backend
  raises `NotImplementedError` about dynamic shapes: heterogeneous layers share
  a block class; compile per block with `dynamic=False`.
- **Dynamo `recompile_limit` hit at layer 9+.** Per-layer SimpleFSDP classes
  count as distinct; the 8-layer debugmodel passes exactly. Raise the limit in
  the recipe and test with a deeper flavor.
- **`UncapturedHigherOrderOpError` at scan trace**: the layer body writes a
  module attribute or captured buffer (MoE counters, routing metrics). Guard
  them behind config; `functional_call` buffer overrides for the rest.
- **A tuple reaches plain cross entropy** on GraphTrainer: the legacy
  `--loss_kernel.use_loss_kernel` path plus the graph trainer. Use
  `PallasCrossEntropyLoss.Config()` + the fused-loss override instead.
- **Config validation fails with a message that reads like a scan error**:
  `num_tokens_per_train_step` != ranks x `num_tokens_per_microbatch_per_dp_rank`.
- **Shardy segfault in `buildTensorFactorShardings`** on GraphTrainer with
  libtpu 0.0.48+: set `xla_enable_hlo_sharding_v3=False`.
- **Compile-time VMEM OOM naming a kernel that is not the one under test**:
  the scoped VMEM default differs by generation (16 MB v5e, 64 MB v6e / v7x);
  read both numbers in the error line before blaming the kernel.

## Environment and launch

- `TORCHTITAN_DEVICE_TYPE=tpu`; `JAX_PLATFORMS=cpu` when JAX is importable;
  `LD_LIBRARY_PATH=<env>/lib`; `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for CPU tests;
  import `torch` before `spmd_types`.
- `rm -f /tmp/libtpu_lockfile` after any libtpu crash or the chip stays wedged.
- xpk: explicit `--script-dir`, single-line `--command`, workload name under
  40 characters, one `--env` per flag, nodepool pinned for multi-host.
- Build images locally; check the base image's pip pins before reusing it.
- `pgrep -f` / `pkill -f` match their own shell; assemble the pattern at runtime.

## Process

- "Got further than before" is not working; 2-4 steps is not a verdict; wait
  6+ steps past warmup.
- A fixed defect is an instance; sweep the class (every call site, every
  model variant, `.claude/rules/models.md`).
- Parity after optimizer updates is seed- and chip-dependent; bound step 0
  tightly, later steps loosely, and keep checks independent.
- Reference bars drift with the reference's stack. Reproduce the published
  recipe at its pinned versions before claiming a beat.
- Do not nudge a running experiment; fix the process that produced it.
