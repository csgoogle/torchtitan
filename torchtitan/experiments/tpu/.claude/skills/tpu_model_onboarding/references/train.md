# Acceptance criteria and numerics classes

`run.md` says where and how a run happens on each execution target and how to
read it; `torchtitan/experiments/tpu/README.md` has the environment, wheel
build, cluster and image steps. This file is what counts as "it trains".

## Environment, before the first run

- Wheel and image torch must match: a torch_tpu wheel built against one torch
  nightly on an image with another imports fine and then dies in
  `init_device_mesh` with an empty process-group name. Build with
  `--config=local_torch --repo_env=TORCH_SOURCE=<the image's torch>` or pin
  the image torch to the wheel's.
- `TORCHTITAN_DEVICE_TYPE=tpu` selects the device (there is no `--device` CLI
  flag on the trainer). `JAX_PLATFORMS=cpu` whenever JAX is importable in the
  env, or it takes the TPU before torch_tpu does. `LD_LIBRARY_PATH=<env>/lib`
  if tokamax or xprof imports fail on ICU symbols.
- libtpu >= 0.0.48 if the model uses `torch._grouped_mm` (empty expert groups
  left the weight gradient uninitialized before that). On 0.0.48+ GraphTrainer
  recipes set `xla_enable_hlo_sharding_v3=False`.
- After a libtpu crash, `rm -f /tmp/libtpu_lockfile` before the next run.
- Build images locally (`FROM <latest image>` + `COPY` for Python-only
  changes); the GitHub workflow build is slow and its pins drift.

## Single host (TPU VM, v6e-4 or v6e-8)

Debug model, three trainer modes, 10 steps each:

```bash
# eager per-op (dispatch-bound; correctness baseline only)
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.<name> --config=<name>_debugmodel \
    --training.steps=10 --compile.no-enable

# per-block torch.compile (the PR 1 performance path)
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.<name> --config=<name>_debugmodel \
    --training.steps=10

# GraphTrainer, once the wrapper PR has landed
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.<name> --config=<name>_debugmodel \
    --training.steps=10 --tpu_config.use_graph_trainer \
    --tpu_config.compile_mode=aot_fx_trace
```

`train_minimal.py` is the single-process local entry when the full trainer
cannot initialize PJRT on the VM. Keep `max_context_length` large enough for
the architecture (longer than any sliding window).

## Target size (GKE)

Use the recipe README's `gcluster job submit` block (Cluster Toolkit; the
README links the cluster and image setup) with the pre-dispatch gate in
`run.md`. If the cluster is driven by xpk instead, the rules that bit before:
`--script-dir` explicit (it tars the cwd otherwise), single-line `--command`,
workload names under 40 characters, one `--env` per flag, multi-host JobSets
pinned to a nodepool. Either way `torchrun --redirects=3 --tee=3` so
tracebacks reach the logs.

## Acceptance criteria for "it trains"

Record every item in the PR description with the command that produced it.

```
Check                        | Bar
-----------------------------|------------------------------------------------------------
Steps                        | >= 10 completed; verdicts from steps past warmup only
Loss                         | decreasing over the run, no NaN/inf, no step-to-step spikes
Determinism                  | two runs with --debug.seed=42 --debug.deterministic give
                             | bit-identical loss and grad_norm (TensorBoard, not stdout:
                             | stdout prints 5 digits). Never --debug.deterministic_warn_only
Step-1 loss oracle           | record the deterministic step-1 loss and grad_norm; every
                             | later PR compares against it
Trainer-mode agreement       | eager / per-block compile / GraphTrainer step 1-10 losses
                             | within ~1e-4 on debugmodel (llama3, muse: 4 decimals)
Structure                    | per-layer patch/replacement counts in the log equal the
                             | config's layer count; heads / kv_heads / vocab in the log
                             | match the checkpoint config; HLO contains the custom_calls
                             | the enabled kernels should emit
Checkpoint                   | save and load round trip on the debug recipe
Memory and MFU               | --tpu_config.enable_mfu_benchmark with
                             | mfu_benchmark_start_step past warmup; peak HBM from the
                             | metrics log; FLOP accounting verified (model.md)
```

Eager log lines from kernels do not fire under `torch.compile`; verify kernel
presence in the HLO dump, not the log.

## Reference trajectory (before Step 3 closes)

A unit test on a tiny config does not prove the real size trains the same as
the reference. Before calling the target recipe done, compare 200 steps
against a reference run of the same flavor (upstream torchtitan on GPU,
MaxText, or the reference framework's own trainer). The reference run is an
input: ask for it, or for its event files, with the settings below pinned; it
is not something the TPU side produces. Pin optimizer and schedule, global batch in sequences,
gradient-reduce dtype, seed, dataset and tokenizer, MoE balancer flavor. Compare
post-warmup medians of the per-step loss (step >= 40), not means and not early
steps. Qwen3-235B: median |diff| 0.06 nats vs 64 B200 eager was accepted; a
persistent offset of that size is reported, not hidden.

## Numerics classes

Every change after PR 1 gets one of these labels in its PR:

- **bit-identical**: step-1 loss and grad_norm equal to the oracle.
- **rounding-order**: differs at fp32 round-off (tile changes, fused kernels,
  reduction reorders). Accepted on speed, but a frontier recipe needs the
  200-step trajectory check above. Do not promise bit-identity from an isolated
  kernel test: whole-program layout changes the reduction order.
- **semantic**: changes what is computed (bf16 gradient reduce, dropped tokens,
  different balancer). Not a performance lever; it is a separate, labelled
  recipe (`*_maxtext_parity`) or it is rejected.

## Reading results

- "Got further than before" is not "works"; 2-4 steps is not a verdict.
- One clean run never clears an uninitialized-memory class of bug; rerun.
- Loss values are only comparable within the same `--training.steps`: the LR
  schedule is a fraction of total steps.
- Step-1 loss identical across two learning rates means a later divergence is
  step-size, not a model bug.
- Before building a probe, run the cheap control: the native model on the same
  pods, the CPU reproduction, the non-MoE flavor.
