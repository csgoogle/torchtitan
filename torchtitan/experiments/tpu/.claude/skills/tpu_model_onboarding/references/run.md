# Running the model: where, how, reading the result, when it fails

`train.md` says what counts as "it trains". This file is how a run happens
on each execution target, how to get the evidence back, and the order in which
to look when it does not work.

## 1. Execution target (Step 0, question 1)

```
Target                          | Commands run       | Chips                   | Artifacts come back as          | You need from the user
--------------------------------|--------------------|-------------------------|---------------------------------|-------------------------------------
A. this machine is the TPU VM   | directly           | torch.accelerator       | files in --job.dump_folder      | nothing; verify the chips yourself
                                |                    | .device_count()         |                                 |
B. a TPU VM reachable over ssh  | gcloud compute tpus| the VM's; detect over   | rsync/scp of dump_folder, or    | TPU name, zone, project, checkout
                                | tpu-vm ssh <name>  | ssh with the same call  | tail of the remote log          | path on the VM (or leave to rsync
                                | --zone=<z>         |                         |                                 | the branch there)
                                | --command='...'    |                         |                                 |
C. GKE via Cluster Toolkit      | gcluster job submit| the slice in the        | GCS paths from the launch block | project, cluster, blueprint /
   (gcluster)                   | per the recipe     | blueprint               | (profile, HLO, metrics); job    | nodepool, image registry; may you
                                | README             |                         | logs via kubectl                | build+push images; dispatch budget
```

Rules that follow from the answer:

- **Rung gating.** Rung 1 (correctness, parity test, 10 deterministic steps on
  `debugmodel`) closes on A or B with 1-4 chips. Kernel-parity rungs close on
  one chip. Only Step 3 (target-size recipe, benchmark table) needs C.
- **C only:** run every CPU-closable check locally first (section 2), then
  treat each dispatch as expensive: pre-dispatch gate (section 5), at most the
  agreed number of jobs, then stop and report.
- **No silent fallback.** A reports no chip, or B's ssh fails, or C's image is
  missing: stop and say so. Do not switch to CPU and mark run-based rows done.
  This is the same rule the test harness follows (`--device=tpu` on a host
  without a TPU is an error, not a skip).
- **Verify the chip before trusting it.**
  `python -c "import torch, torch_tpu; print(torch.accelerator.device_count())"`
  (with `LD_LIBRARY_PATH=<env>/lib`; `torch.tpu.device_count()` is
  TorchTPU-internal and raises). On a shared host also check nothing else holds
  `/dev/vfio/*` (`ls -l /proc/*/fd 2>/dev/null | grep vfio`).
- **xpk** instead of gcluster: same procedure; the traps are in `train.md`
  (explicit `--script-dir`, single-line `--command`, workload names under 40
  characters, one `--env` per flag, nodepool pinned for multi-host).

## 2. What closes without a TPU

```
Check                                           | CPU | 1 chip | 4 chips | slice
------------------------------------------------|-----|--------|---------|------
reference-parity test (in-process or golden)    | yes | yes    |         |
state-dict round trip, flavor / config checks   | yes |        |         |
mask-semantics test                             | yes |        |         |
test_train_distributed cases (debugmodel, 3     | yes*| yes    | yes     |
steps; *CPU with --device=cpu on tiny flavors)  |     |        |         |
10-step run, determinism repeat, step-1 oracle  |     | yes    | yes     |
kernel parity (splash, fused CE, MoE)           |     | yes    |         |
FSDP / SimpleFSDP end to end                    |     |        | yes     |
HLO custom_call evidence                        |     | yes    |         |
target recipe, benchmark table, trajectory      |     |        |         | yes
```

CPU runs: `TORCH_DEVICE_BACKEND_AUTOLOAD=0 JAX_PLATFORMS=cpu`; a test that
imports `torch_tpu` unconditionally breaks the CPU shard.

## 3. Launching

Single host (A or B), `debugmodel`, 10 steps, the three trainer modes:

```bash
export TORCHTITAN_DEVICE_TYPE=tpu JAX_PLATFORMS=cpu LD_LIBRARY_PATH=<env>/lib
# eager per-op (dispatch-bound; correctness baseline only)
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.<name> --config=<name>_debugmodel \
    --training.steps=10 --compile.no-enable --job.dump_folder=<dir>/eager
# per-block torch.compile (the rung-1 performance path)
torchrun ... --training.steps=10 --job.dump_folder=<dir>/compile
# GraphTrainer, once the wrapper rung has landed
torchrun ... --training.steps=10 --tpu_config.use_graph_trainer \
    --tpu_config.compile_mode=aot_fx_trace --job.dump_folder=<dir>/gt
```

Over ssh (B), wrap the same line:
`gcloud compute tpus tpu-vm ssh <name> --zone=<zone> --project=<project> --worker=all --command='cd <checkout> && <the torchrun line>'`
and fetch results with
`gcloud compute tpus tpu-vm scp --recurse <name>:<dir> <local> --zone=<zone>`.
Sync code before each run (`git push` + `git checkout` on the VM, or rsync
the worktree); record the VM-side commit in the run notes.

Cluster (C): use the recipe README's `gcluster job submit` block verbatim,
with the image you built (`FROM <latest torchtitan image>` + `COPY` for
Python-only changes; do not use the GitHub image workflow, it is slow and its
pins drift). Add `--training.steps`, `--job.dump_folder=<gcs path>` and
`--profiling` flags from the README. Keep `torchrun --redirects=3 --tee=3` so
tracebacks reach the logs.

`train_minimal.py` is the single-process local entry when the full trainer
cannot initialize PJRT on the VM.

## 4. Reading a run

Everything the acceptance table in `train.md` asks for is in
`--job.dump_folder`; bring it back to where you work (A: it is there; B: scp;
C: `gsutil -m cp -r`) and read it there, so all three targets end in the same
procedure.

```
Evidence                      | Where                                           | How to read
------------------------------|-------------------------------------------------|---------------------------------------------
per-step loss, grad_norm      | <dump_folder>/tb/*/events.out.tfevents.*        | TensorBoard event file, full precision.
                              |                                                 | stdout prints 5 digits: never compare runs
                              |                                                 | from stdout.
peak HBM, step time, TPS, MFU | the metrics log lines ("memory:", "tps:",       | "memory:" is the XLA temp region, not the
                              | "mfu:") and --tpu_config.enable_mfu_benchmark   | heap; for the heap read buffer-assignment.txt
                              |                                                 | in the HLO dump or the recipe's "Physical
                              |                                                 | peak HBM" method
HLO                           | --extra-libtpu-init-flags=xla_dump_to=<dir>     | grep custom_call names for the enabled
                              | (TPU README, "flags"); after-optimizations .txt | kernels; grep input_output_alias for donation;
                              | per program. An exported LIBTPU_INIT_ARGS /     | count programs per step
                              | XLA_FLAGS overrides the recipe's flags per key  |
structure                     | startup log                                     | per-layer patch / replacement counts == layer
                              |                                                 | count; heads / kv_heads / vocab == checkpoint
determinism                   | two dump folders                                | compare event files step by step (below)
```

Comparing two runs, step by step, at full precision:

```python
from tensorboard.backend.event_processing import event_accumulator as ea
def series(folder, tag="loss_metrics/global_avg_loss"):
    acc = ea.EventAccumulator(folder); acc.Reload()
    return {e.step: e.value for e in acc.Scalars(tag)}
a, b = series("<run1>/tb/<subdir>"), series("<run2>/tb/<subdir>")
for s in sorted(a):
    print(s, a[s], b.get(s), None if b.get(s) is None else a[s] - b[s])
```

Bars: determinism means every step equal (`a[s] == b[s]`, not "close"); the
trainer-mode agreement in `train.md` is ~1e-4 on `debugmodel`; "no spikes"
means no step-to-step increase larger than the largest decrease seen so far
after warmup. Verdicts come from steps past warmup only; 2-4 steps is not a
verdict.

## 5. Pre-dispatch gate (before any multi-host or cluster launch)

All of these, every time; a 64-chip dispatch that dies at step 2 on a defect
a CPU run would have shown is the most expensive mistake in this process
(it has happened: six dispatches across two experiments, one measurement).

1. The exact config parses: `python -c "from torchtitan.config.manager import ConfigManager; ConfigManager().parse_args([...])"`
   with the same `--module/--config` and overrides as the launch.
2. The exact config runs 3 steps somewhere cheaper: CPU on a tiny flavor if
   the recipe has one, else one chip, else 4 chips. Same flags.
3. The image contains the commit you think it does (`docker run <image> git -C
   /workspace rev-parse HEAD`, or the equivalent for your image layout).
4. The optimizer path is one thing (`compile_optimizer` xor
   `optimizer_in_graph` xor eager); `compile_grad_norm` not combined with
   `optimizer_in_graph`; `use_scan_layers` only with GraphTrainer.
5. Token budget consistent: `num_tokens_per_train_step` == ranks x
   `num_tokens_per_microbatch_per_dp_rank`.
6. Flags from the command line are copied into the recipe (left on the command
   line they get forgotten; the scan recipe measured 10% slow that way).

Budget: agree the number of dispatches with the user before the first one;
when it is spent, write the ladder table and stop.

## 6. When it fails: look in this order

Each row: symptom, first check, cheapest tier that reproduces it. Reproduce
at the cheapest tier before changing anything on the expensive one.

```
Symptom                                              | First check                                           | Reproduce on
-----------------------------------------------------|-------------------------------------------------------|-------------
import torch_tpu ok, init_device_mesh dies with an   | wheel's torch glue vs the image's torch nightly        | 1 process
empty process-group name                             | (pip show torch, torch_tpu build info)                 |
"TPU initialization failed" / vbar.sock / lockfile   | another process on the chips; rm -f                    | 1 chip
                                                     | /tmp/libtpu_lockfile after a crash                     |
config ValueError that reads like a model error      | the budget identity in section 5; the optimizer-path   | CPU
                                                     | exclusions                                             |
compile error on a new op / dynamic shape            | heterogeneous layers sharing a block class: compile    | CPU trace
                                                     | per block with dynamic=False (#491 8f6e12ffa)          | (make_fx)
recompile_limit at layer 9+                          | per-layer SimpleFSDP classes count as distinct; raise  | 1 chip
                                                     | the limit, test a deeper flavor                        |
OOM at compile or step 1                             | memory policy vs reshard policy (guide section 8);     | 4 chips
                                                     | activation budget; batch; is a kernel's VMEM tile the  |
                                                     | wrong generation's default                             |
NaN / inf at step k                                  | fp32 norms? mixed-precision reduce dtype? run eager    | 1 chip
                                                     | per-op for the same steps and compare the step where   |
                                                     | they diverge                                           |
loss fine, throughput half of expected                | attention got no segment ids (packed batch attends     | HLO on
                                                     | across documents); check custom_call operands          | 1 chip
loss fine, eager vs compile disagree > 1e-4           | a kernel or patch silently off in one mode; count      | 1 chip
                                                     | replacements in the startup log; HLO custom_calls      |
hang in a collective                                 | all ranks in the same program? --redirects=3 --tee=3;  | 4 chips
                                                     | kill after 3 polls with no step progress                |
determinism repeat differs                           | --debug.deterministic set on both? reading stdout      | 1 chip
                                                     | instead of event files? a kernel with nondeterministic |
                                                     | reduction (label rounding-order, not a bug)            |
"got further than before"                            | not a result; run 6+ steps past warmup                 |
```

If the first check does not explain it, run the cheap control before building
a probe: the native (un-ported) model on the same chips, the CPU reproduction,
the non-MoE flavor. Then `/numerics_debugging` for a first-diverging-op search.
