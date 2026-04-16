# performance-autoresearch (TorchTitan JAX Llama3 8B, TPU v6e)

Variant of Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) adapted for **performance optimization** instead of model-quality search. An AI agent modifies the TorchTitan JAX Llama3 8B training pipeline to maximize throughput on TPU v6e, using MaxText's Llama3.1-8B run as a reference ceiling and [xprof_mcp](https://github.com/openxla/xprof) for profile-driven analysis.

**The model's output distribution is invariant.** The agent optimizes *how* the same computation runs — sharding, attention kernel, remat policy, XLA flags, layer implementation, training config — never the Llama3 8B architecture itself.

## Setup

1. **Pick a tag** of the form `v6e4-<YYYYMMDD>-<NN>` (e.g. `v6e4-20260416-01`), where `NN` is a two-digit session counter. The branch `perfautoresearch/<tag>` must not already exist.
2. **Create the branch** in the TorchTitan repo: `git checkout -b perfautoresearch/<tag>`.
3. **Read the in-scope context** (once, then consult as needed):
   - This `program.md`.
   - `torchtitan/experiments/jax/train_minimal.py` and the full `torchtitan/experiments/jax/` subtree.
   - TPU-wiki references: `profiling.md`, `xprof` entity, `xprof_mcp OSS` source, `roofline-analysis.md`, `activation-checkpointing.md`, `flash-attention.md`, `splash-attention.md`, `scaling-book.md`, `torchtitan` and `MaxText` entity pages.
   - MaxText reference source: `raw/code/maxtext/` — particularly `layers/attentions.py`, `layers/models.py`, the sharding setup, and the XLA flag wiring.
4. **Verify tooling**: `py312` and `maxtext_py312` conda envs resolve; `/mnt/disks/persist/torch-tpu/dump_folder/` (or whichever `--job.dump_folder` the agent chooses) is writable; xprof_mcp MCP tools respond (`list_runs` succeeds).
5. **Capture the reference profile** (one-shot, before the loop):
   - Run MaxText with the baseline command in the project notes.
   - Call xprof_mcp: `get_overview`, `get_top_hlo_ops(limit=20)`, `get_memory_profile`, `aggregate_xplane_events(plane_regex='/device:TPU:0')`, `list_hlo_modules`. If relevant, `get_hlo_neighborhood` on top ops.
   - Record MaxText TPS, MFU, step_time, peak HBM, top-10 ops, bottleneck category, notable HLO patterns (flash attention kernel name, collective placement, remat fingerprint) to `OBSERVATIONS.md` under `## reference: MaxText llama3.1-8b`.
6. **Initialize `RESULTS.tsv`** with header only (already scaffolded).
7. **Initialize `OBSERVATIONS.md`** with sections `## reference: MaxText ...`, `## baseline: TorchTitan ...`, `## experiments`, `## approach evolution` (already scaffolded).
8. **Confirm and go.**

## Experimentation

Each experiment starts from the current branch tip and runs the TorchTitan command. The **baseline** command is:

```bash
XLA_FLAGS="--xla_dump_to=/mnt/disks/persist/torch-tpu/dump_folder/hlo_dumps --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*" \
LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=131072" \
/home/alekseyv_google_com/miniconda3/envs/py312/bin/python -m \
      torchtitan.experiments.jax.train_minimal \
    --model.name=llama3 \
    --model.flavor=8B \
    --training.dataset_path=tests/assets/c4_test \
    --model.hf_assets_path=tests/assets/tokenizer \
    --training.seq_len=2048 \
    --training.global_batch_size=4 \
    --training.steps=15 \
    --jax_config.use_scan \
    --activation_checkpoint.mode=full \
    --profiling.enable_profiling \
    --profiling.profile_freq=10 \
    --profiling.profiler_warmup=0 \
    --profiling.profiler_active=4 \
    --profiling.save_traces_folder="profile_trace" \
    --job.dump_folder=/mnt/disks/persist/torch-tpu/dump_folder
```

Any flag on this command is tunable (including `--training.seq_len`, `--training.global_batch_size`, `--jax_config.use_scan`, `--activation_checkpoint.mode`, the XLA dump path, `--job.dump_folder`, `--profiling.save_traces_folder`). The goal is to find the configuration *and* code that maximize TPS for Llama3 8B on TPU v6e; the right config for TorchTitan may differ from MaxText's.

### The contract

The final trained model must be equivalent to a baseline-trained Llama3 8B — same output distribution (bitwise up to floating-point rounding for a fixed seed/config). Everything about *how* that computation is expressed, scheduled, sharded, stored, or compiled is tunable.

### What you CAN do

Anything preserving the output-distribution contract:

- Modify any file inside `torchtitan/experiments/jax/` and any TorchTitan internals it imports.
- **Training config**: `global_batch_size`, per-device batch, `seq_len`, grad accumulation, `use_scan` on/off and unroll factor, `activation_checkpoint.mode`, `--profiling.*`, `--job.dump_folder`, `--profiling.save_traces_folder`.
- **Environment/compiler**: any XLA flag or `LIBTPU_INIT_ARGS` value. Change `XLA_FLAGS`'s `--xla_dump_to=<path>` when a clean HLO dump per experiment is useful. Set compilation-cache env vars (`JAX_COMPILATION_CACHE_DIR`, persistent cache paths).
- **Sharding**: mesh shape, axis names, `PartitionSpec` choices, FSDP/TP/DP mix — any valid sharding on v6e-4.
- **Attention kernel**: dense → splash/flash, swap between JAX native, `pallas.flash_attention`, `splash_attention`, custom Pallas — as long as numerically equivalent.
- **Remat policy**: `full` / `selective` / `none`, specific save lists, host-offload of checkpointed tensors, `jax.checkpoint_policies.*`.
- **Optimizer implementation**: fusion, state dtype, host-offload of optimizer state (as MaxText does via `optimizer_memory_host_offload`). Gradient-clipping formulation if mathematically equivalent.
- **Precision mix**: bf16 mixed precision is baseline; fp32 norms + bf16 compute fine; fp32 master weights + bf16 gradients fine; fp8 acceptable if the loss trajectory matches on the sanity run.
- **Module structure**: fused QKV vs separate projections; different RoPE implementation; different residual fusion; flat vs nested modules — provided math is preserved.
- **Numerical nits**: norm epsilon, weight init (they don't change trained-model equivalence under fixed seed, but be careful — if you touch seed or init, run the sanity check).
- **JIT boundaries**: where `jit` / `pjit` / `scan` are placed, `donate_argnums`, `static_argnums`, `inline`, unroll factors.
- **Data pipeline**: prefetch depth, host workers, caching, async host→device transfer, packing.
- **Pallas kernels** for any op, provided numerical equivalence.
- **Pre-compile shapes**, adjust AOT compilation behavior.

### What you CANNOT do

- Change Llama3 8B architecture: `n_layers` (32), `n_heads` (32), `n_kv_heads` (8), `hidden_dim` (4096), `ffn_hidden` (14336), `vocab_size` (128256), `head_dim`, RoPE frequency base/theta, RMSNorm positioning, residual structure, tokenizer. These define the 8B parameter count.
- Change `--model.name` or `--model.flavor` (stays `llama3` / `8B`).
- Quantize trained weights below bf16 (int8/int4 weight quantization is off-limits for this loop; activation int8 is also off-limits).
- Disable or approximate attention (linear attention, sliding window on a dense Llama are off-limits; swapping to a numerically-equivalent kernel is fine).
- Skip or approximate the optimizer step, norm layers, or residual connections.
- Edit `train_minimal.py`'s flag parsing to silently override a blocked invariant.
- Modify anything outside `torchtitan/`. MaxText, xprof_mcp, and the wiki are strictly read-only.
- Cheat the metric: skip or warp step-timing, fake the profile, log crashes as `discard`.

**If a MaxText source change seems necessary**: stop the loop and ask the user. Explain what you'd change, why, and what TorchTitan experiment it unblocks. Do not proceed until authorized.

**Loss sanity check** — required after changes that can affect numerics (attention kernel swap, precision change, custom Pallas kernel, optimizer internals, init/seed change). Run a 30-step side experiment and verify the loss trajectory is within ±2% of the current branch's baseline loss at the same step. If it diverges, the change is a cheat — revert.

**Simplicity criterion** (unchanged from autoresearch):

- TPS win from **deleting** code → clear keep.
- ~1% TPS win with 200 lines of fragile new code → probably discard; log the idea as a follow-up.
- No-op-or-faster plus simpler → always keep.

**The first run**: baseline. Run the TorchTitan command as-is on the fresh branch. Record TPS, MFU, step_time, peak HBM, top-10 ops, bottleneck category.

## Metric

**Primary: TPS** (tokens per second) = `global_batch_size × seq_len / median_step_time_s`. Higher is better. Invariant under batch/seq tuning, so config changes are fairly comparable.

**Secondary: MFU** — sanity check. If TPS goes up but MFU goes down, we're moving more tokens without better hardware use — probably fine on v6e-4, but flag it in observations.

**Diagnostic: step_time** — human-readable signal. Not a decision metric.

**Measurement protocol**: median step_time over **steps 6–15** (skip steps 1–5 to exclude compile and first-iteration warmup). If a run has fewer than 10 usable steady-state steps, rerun with `--training.steps` bumped to 20 and still measure over steps 6–15. Compile time is recorded separately for awareness but does not enter the metric.

## Output format

Extract after each run (from TorchTitan's step-log JSON and xprof_mcp `get_overview`):

```
tps:             <global_batch × seq_len / median step_time>
mfu_percent:     <from TorchTitan metrics or get_overview>
step_time_ms:    <median steps 6-15>
peak_hbm_gib:    <from get_memory_profile>
compile_seconds: <time to first step — diagnostic only>
config:          <key flags that differ from baseline, one-line>
```

## Logging results

Append one row to `RESULTS.tsv` per run (tab-separated, **untracked**):

```
commit	tps	mfu_percent	step_time_ms	peak_hbm_gib	config	status	description
```

- `commit` — short 7-char hash
- `tps` — `0.00` on crash
- `mfu_percent` — `0.0` on crash
- `step_time_ms` — `0.00` on crash
- `peak_hbm_gib` — `0.0` on crash
- `config` — pipe-separated diffs from baseline config, e.g. `seq_len=4096|batch=8|attn=splash`
- `status` — `keep` / `discard` / `crash`
- `description` — one-line change summary

Example:

```
commit	tps	mfu_percent	step_time_ms	peak_hbm_gib	config	status	description
a1b2c3d	3420.1	28.4	2394.8	58.1	baseline	keep	baseline
b2c3d4e	4180.5	31.1	1959.3	58.3	attn=splash	keep	splash attention kernel for QKV
c3d4e5f	3422.0	28.4	2393.5	58.1	tp=4	discard	4-way TP within ICI, no overlap gain
d4e5f6g	0.0	0.0	0.00	0.0	seq=4096	crash	seq_len=4096 OOM during attn fwd
e5f6g7h	4320.2	32.0	3794.0	62.1	seq=4096|batch=2|attn=splash|remat=selective	keep	larger seq with selective remat + splash
```

`RESULTS.tsv` is not committed.

## Observations log — detailed report per experiment

After each experiment, append a full block to `OBSERVATIONS.md` **before** deciding keep/discard. The template is deliberately verbose — reasoning is the durable output; the TSV is just the ledger. Every field is mandatory (use "n/a" if truly not applicable, but this should be rare).

```markdown
### <commit>  <one-line summary>

**Config**:
- Command diff from previous keep: <full flag diffs + code change summary>
- Profile path: `<dump_folder>/<run_name>/`
- HLO dump path: `<xla_dump_to path>`

**Hypothesis**:
<2-4 sentences. What do you expect will improve and why.
Cite specific profile signals or wiki pages that motivated this.
Link to the prior observation that raised this as a follow-up, if any.>

**Changes made**:
- <code change 1 — file, function, what was swapped, mathematical justification for equivalence>
- <code change 2 — ...>
- <config flag change — old value → new value, why>

**Expected outcome**:
<What TPS / MFU change you predict, and through what mechanism —
e.g. "splash_attention should fuse the softmax and avoid the HBM
round-trip seen in top-op N, expect ~15% TPS win with no memory change">

**Actual outcome**:
- TPS: X → Y (Δ Z%)
- MFU: A → B
- Step time: P → Q ms
- Peak HBM: M → N GiB
- Compile time: C → D sec

**Profile signals**:
- Bottleneck category (from `get_overview`): <compute | memory | host | collective>
- Top ops change (from `get_top_hlo_ops`): which ops moved up/down in the ranking
- HLO patterns of note: <fusion formed/broke, collective placement, pattern in `get_hlo_neighborhood` output>
- XLA pass diff (if relevant, from `diff_hlo_stages`): what the compiler did differently

**Analysis**:
<If actual matched expected: why the mechanism worked as predicted.
If not: what really happened, what confounds were in play,
what this tells us about the underlying system.
If crash: the root cause, and whether it invalidates the hypothesis
or just the implementation.
If a loss-sanity run was required, report baseline-loss vs experiment-loss
at step 30 here.>

**Decision**: keep | discard | crash

**Follow-ups** (ideas this surfaced):
- <idea 1 — what, why it's plausible>
- <idea 2 — ...>
```

`OBSERVATIONS.md` is the reasoning log. Commit it alongside code on each `keep`. Failed-and-reverted experiments also write a full block — they're valuable context for the loop.

## Improving the approach itself

If during experimentation the agent discovers a generalizable improvement to this methodology — a better measurement protocol, a richer profile-reading heuristic, a new anti-cheat case worth banning, a reference page worth consulting — it **commits the improvement to this `program.md`** on the research branch, and adds an entry to `OBSERVATIONS.md` under `## approach evolution`:

```markdown
### approach update — <commit>
<what changed in program.md, why, and the experiment that revealed it>
```

The human reviews these at session end to decide what promotes to the canonical `program.md` on the main branch.

## The experiment loop

LOOP FOREVER:

1. Check git state and read the tail of `OBSERVATIONS.md` for recent context.
2. Generate a hypothesis, priority order:
   a. **Profile-driven**: highest-signal gap vs MaxText from the most recent profile (top slow op / missing fusion / worse collective placement / memory pressure).
   b. **Follow-up**: an unexplored idea from a previous observation's "follow-ups".
   c. **MaxText cross-check**: find something MaxText does that TorchTitan doesn't, and try porting the technique. Re-run MaxText with altered command-line params if needed to isolate the technique's contribution (MaxText source itself stays read-only).
   d. **Wiki-driven**: consult scaling-book / profiling / roofline pages if the profile suggests a known gotcha.
3. Edit the TorchTitan code to implement the hypothesis.
4. `git commit` with a message stating the hypothesis in one line.
5. Run the experiment command (`>run.log 2>&1`).
6. If run.log has no step summary → crash. `tail -n 80 run.log`; dumb bug → fix and rerun; fundamentally broken → crash row, reset, move on.
7. Extract metrics; call xprof_mcp on the new profile:
   - `list_runs()` → find the new run
   - `get_overview`, `get_top_hlo_ops(limit=20)`, `get_memory_profile`
   - For HLO-level changes: `list_hlo_modules`, `get_hlo_module_content`, `get_hlo_neighborhood` on the relevant op
   - If the change targets a compiler pass: `list_hlo_dump_modules`, `diff_hlo_stages`
8. Write the full detailed report to `OBSERVATIONS.md` (template above). Append row to `RESULTS.tsv`.
9. If TPS improved (higher) → advance (keep commit + observations). Else → `git reset --hard` back to the prior tip (observations stay — failures teach).
10. Loop.

**Timeout**: kill any experiment running > 15 minutes and treat it as a crash.

**Crashes**: same judgment as autoresearch. Easy bug → fix. Fundamentally broken idea → log and skip.

**NEVER STOP**: once the loop begins, do not pause to ask. Idea well running dry: re-read MaxText's Llama layers, diff structural differences, consult wiki concept pages. Re-run MaxText with tweaked params if a hypothesis needs isolation. Only stop to ask if you need a MaxText source change — explain why, wait for authorization.

## Reference material at a glance

- Bottleneck diagnosis: `tpu_wiki/wiki/concepts/profiling.md`, `tpu_wiki/wiki/concepts/roofline-analysis.md`
- Known gotchas: `tpu_wiki/wiki/sources/2025-scaling-book.md` Ch 1, 3, 7
- Attention kernels: `tpu_wiki/wiki/concepts/flash-attention.md`, `tpu_wiki/wiki/concepts/splash-attention.md`
- Sharding: `tpu_wiki/wiki/concepts/distributed.md`, `tpu_wiki/wiki/concepts/fsdp.md`, `tpu_wiki/wiki/concepts/gspmd.md`, `tpu_wiki/wiki/concepts/matmul-sharding.md`
- Scan + remat: `tpu_wiki/wiki/concepts/scan-layers.md`, `tpu_wiki/wiki/concepts/activation-checkpointing.md`, `tpu_wiki/wiki/concepts/rematerialization.md`
- Reference implementation: `tpu_wiki/raw/code/maxtext/` — `layers/attentions.py`, `layers/models.py`, sharding module, XLA flag setup.
- MCP tools: `tpu_wiki/wiki/sources/2026-04-xprof-mcp-oss-repo.md` — 17 tools + `discovery_flow`.
