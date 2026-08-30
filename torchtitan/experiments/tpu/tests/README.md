# TPU experiments — test coverage

What each test module protects, and why it exists. Fifteen modules, **144
tests**. Nine run on CPU, four need a TPU, one runs on both, and one needs
eight chips.

The organising idea: most of this feature set fails *silently* rather than
loudly. A scan body that saves the wrong tensors still computes the right
answer — it just uses 100 GB more memory. An FSDP gather hoisted out of the
loop is still numerically correct. An activation-checkpoint policy that gets
erased still trains. So the tests are split in two: **semantic** tests that
compare against an unrolled reference, and **structural** tests that assert on
the shape of the traced graph, because that is the only place the silent
failures are visible.

Last full run: 2026-08-26, image
`torchtitan-container:llama3-gtscan-v013b-338fd5e6`, on a v6e-8 host, against
tree `e4034997`. All fifteen modules pass — 93 CPU-tier, 47 single-chip, 3
distributed. One CPU test has been added since
(`PublicApiTest::test_the_docstrings_worked_example_actually_binds`, taking the
CPU tier to 94); it needs no device and was verified outside the container. The only failures are the two documented CPU-only ones below, both
of which pass on a device.

Two cases skip on that image, both correctly:

| Skipped case | Reason |
|---|---|
| `test_scan_lowering.py::ArgumentValidationTest::test_short_argument_list_raises` | `v013b` carries the scan-dispatch fix but not the two `compiled_mode.cc` diagnosability commits, so a bad argument list still aborts rather than returning a status. Passes on `v015`. |
| `test_scan_reverse.py::BackwardRewriteTest::test_gradients_match_stock_backward_tpu` | Eager autograd-through-scan is broken upstream; there is no stock backward to compare against. |

## Running them

```bash
# CPU
python -m torchtitan.experiments.tpu.tests.test_scan_over_layers
python -m torchtitan.experiments.tpu.tests.test_scan_memory_policy
python -m torchtitan.experiments.tpu.tests.test_scan_fsdp_lowering
python -m torchtitan.experiments.tpu.tests.test_scan_body_ac_shim
python -m torchtitan.experiments.tpu.tests.test_rope_real_arithmetic
python -m torchtitan.experiments.tpu.tests.test_parallelize_scan
python -m torchtitan.experiments.tpu.tests.test_torch_tpu_fix
python -m torchtitan.experiments.tpu.tests.test_workarounds
python -m torchtitan.experiments.tpu.graph_trainer.tests.test_tpu_graph_trainer_config

# TPU, one chip
python -m torchtitan.experiments.tpu.tests.test_scan_lowering
python -m torchtitan.experiments.tpu.tests.test_scan_reverse
python -m torchtitan.experiments.tpu.tests.test_scan_compiled_hlo
python -m torchtitan.experiments.tpu.graph_trainer.tests.test_tpu_graph_trainer_unit
python -m torchtitan.experiments.tpu.compiled_optimizers.tests.test_functional --device=tpu

# TPU, eight chips
python -m torchtitan.experiments.tpu.graph_trainer.tests.test_tpu_graph_trainer_distributed
```

To run the lot, including the two device tiers, see
[`run_in_container.sh`](run_in_container.sh) below.

**Run them in the container, not in a host conda env.** These modules import
`torch.distributed.fsdp` symbols that moved between torch releases; a host env
one minor version off the image fails at import with something that looks like
a code bug and is not (`ImportError: cannot import name 'DataParallelMeshDims'`
is the usual one). The image is the only environment these numbers were taken
in.

The device modules need a chip and a torch_tpu build. If you have a container
image with both, [`run_in_container.sh`](run_in_container.sh) will set up the
slice topology torch_tpu's PjRt client requires and run one module inside it:

```bash
IMG=<your torch_tpu image> ./run_in_container.sh \
  torchtitan.experiments.tpu.tests.test_scan_lowering

# the distributed module wants the whole host
IMG=<your torch_tpu image> NPROC=8 ./run_in_container.sh \
  torchtitan.experiments.tpu.graph_trainer.tests.test_tpu_graph_trainer_distributed
```

`--device` is accepted by the five `BaseAcceleratorDeviceTest` modules —
`test_functional.py`, the three TPU scan modules, and
`test_tpu_graph_trainer_unit.py` — and rejected by the CPU-only ones, where
passing it is a flags-parsing error. The three TPU scan modules skip cleanly
under `--device=cpu` rather than failing; that is what makes them safe for a CI
runner that has no TPU, though CI does not schedule them there (see below).

Two modules have cases that cannot pass without an accelerator, and both fail
the same way — `AssertionError: If capturable=True, params and state_steps must
be on supported devices`. That is `torch.optim`'s own guard, not a bug here:

| Module | On CPU | On TPU |
|---|---|---|
| `compiled_optimizers/tests/test_functional.py` | 7 pass, **6 fail** | 13 pass |
| `graph_trainer/tests/test_tpu_graph_trainer_unit.py` | 5 pass, **1 fail** | 6 pass |

### Which shard CI picks

`.ci/discover_tpu_tests.py` classifies by **base-class name**: a
`BaseAcceleratorDeviceTest` goes to the single-TPU shard, a
`BaseDistributedDeviceTest` to the distributed shard, and anything else to a CPU
runner with no accelerator attached. A TPU test class that forgets to inherit
from one of those two is not skipped — it is scheduled on a machine that cannot
run it.

The classifier reads the base class as written in the `class` statement, and it
accepts both spellings — `ast.Name` for `BaseAcceleratorDeviceTest` and
`ast.Attribute` for `base_device_test.BaseAcceleratorDeviceTest`. Prefer the
dotted one. Importing the base class into a test module's namespace makes
unittest collect **the base class itself** as a test of that run, and its
inherited `test_gpu_available` then executes before `setUpModule`. That is
mostly cosmetic — it inflates the count by one, which is why
`test_tpu_graph_trainer_unit.py` reports six tests for five of its own — but in
`test_scan_compiled_hlo.py` it was fatal, because that module's `setUpModule`
needs the chip still to be free.

## Where each module runs, and why

The split is not a preference; each module runs at the cheapest tier that can
actually observe the failure it guards. Nine modules land on CPU because what
they assert is a property of the **traced graph or the Python plumbing**, which
is identical on both devices — running them on a chip would cost scarce CI
minutes and observe nothing extra.

| Tier | Modules | Why this tier is the right one |
|---|---|---|
| **CPU** | `test_scan_over_layers`, `test_scan_memory_policy`, `test_scan_body_ac_shim`, `test_parallelize_scan`, `test_torch_tpu_fix`, `test_workarounds`, `test_rope_real_arithmetic`, `test_scan_fsdp_lowering`, `test_tpu_graph_trainer_config` | The assertion is about an fx graph, a tag set, a call order, a source-text detector, or an FQN — none of which a device changes. Two are subtler: `test_rope_real_arithmetic` guards a **trace-time** `NotImplementedError`, which fires before any device is involved and asserts bit-exactness at `atol=0` that a device's fused arithmetic would break; `test_scan_fsdp_lowering` runs a real gloo world — of one for the placement checks, and of **four** for `InBodyCollectiveTest`, where the size is load-bearing (on a mesh of one, SimpleFSDP's `redistribute(Shard(1) -> Replicate)` emits no collective at all, so the checks would pass against the broken read) — because the structural claim is *where the collective sits in the graph*, not what it computes. |
| **One chip** | `test_scan_lowering`, `test_scan_compiled_hlo`, `test_scan_reverse`, `test_tpu_graph_trainer_unit` | These assert on what the **compiler** did — StableHLO, post-optimization HLO, the `kReverse` lowering — and there is no compiler without a device. |
| **CPU + one chip** | `compiled_optimizers/tests/test_functional` | The math is device-independent; the capturable path is not. |
| **Eight chips** | `graph_trainer/tests/test_tpu_graph_trainer_distributed` | Sharding, collectives, and the composition of the whole target stack. |

The gap that this split *did* leave was composition. Every piece of the 70B
target — scan, the in-body gather, `aot_fx_trace`, the compiled optimizer — had
its own module, and none of them ran together. The
`llama3_scan_compiled_optimizer` case in the distributed module closes that;
see its section below.

## Coverage by module

### `test_scan_over_layers.py` — 15 tests, CPU

The stack itself: does `ScannedLayerStack` compute what the Python loop
computed, and does it actually collapse the graph.

| Group | Protects |
|---|---|
| Structure (7) | Stacked storage is one `[n_layers, *shape]` parameter per layer-relative FQN, **not** one per layer — the property the whole memory argument rests on. Applying the transform is idempotent. Layer stacks with mismatched shapes or FQNs are rejected rather than silently mis-stacked. Round-trip to and from an unrolled `state_dict` is lossless, and an incomplete `state_dict` is rejected — checkpoints must stay interchangeable between the two forms. |
| Semantics (4) | Forward and backward both match the unrolled loop. Gradients **differ across layers** — a scan that broadcast one layer's gradient to all N would pass a naive parity check, so this is asserted separately. A carry that doesn't require grad raises instead of silently producing zeros. |
| Graph (2) | The traced graph contains a `scan` op, and **graph size is independent of depth**. This is the compile-time claim stated as a test; without it a regression to unrolling is invisible. |

### `test_scan_memory_policy.py` — 12 tests, CPU

What the graph trainer's own SAC pass does — and does *not* do — once the
layers are scanned.

The pass forces `MUST_SAVE` at layer boundaries, and finds those boundaries by
parsing an integer layer index out of each node's `module_fqn` (`layers.7.…`).
Scan replaces the N per-layer modules with one `layer_template`, so that integer
no longer exists: every node reports `_NOT_IN_LAYERS` and the boundary pass can
never fire, with no error and no log line saying so.

**This does not mean activation checkpointing is absent under scan.** It comes
from `ScannedLayerStack.set_recompute_body`, whose ambient `current_meta`
window tags the body for the HOP's own partitioner — see `test_scan_body_ac_shim.py`. What
is inert is specifically this pass's layer-boundary `MUST_SAVE`. The two
mechanisms must not be assumed to compose, and that is what this module pins.

Covers FQN translation under both forms, and whether boundary saves fire under
each. One test is an **expected failure** asserting the behaviour we would want
if the heuristic stopped keying on FQN integers — it documents the gap rather
than hiding it, and if it ever passes unexpectedly the decorator should come
off. Widening the comparison from `>` to `!=` does not fix it: both sides of a
scan boundary report the same sentinel, so both spellings are false. The
missing layer index is the defect, not the operator.

### `test_scan_body_ac_shim.py` — 11 tests, CPU

The mechanism that *does* provide activation checkpointing inside a scan body.
`torch.utils.checkpoint` is a silent no-op there — the HOP re-traces the body
and discards the tags — so the tag is instead delivered through
`fx_traceback.current_meta`, the ambient channel the tracer copies into every
node it creates, and which therefore survives that re-trace.

| Group | Protects |
|---|---|
| Ambient window (3) | Joint-tracing a real `scan` under the window tags the body's compute; without the window the body is untagged; and the window does not outlive the call. The first is the claim the whole mechanism rests on, and it is asserted through a real HOP rather than a hand-built graph precisely because the interesting part is surviving the re-trace. The trace is `aot_export_joint_simple`, not `torch.compile` — dynamo would trace the window block instead of executing it. |
| Strip (4) | The shim removes the tag from `placeholder` nodes (min-cut raises `NetworkXError` on them) and from the outer `scan` node (left tagged, min-cut recomputes the whole layer loop in backward — the forward runs twice per step, silently, while the saved-set metric *improves*). Equally, it keeps the tag on real body compute and leaves untagged nodes alone. Over-stripping is as wrong as under-stripping and much harder to notice. |
| Lifecycle (4) | Install replaces and uninstall restores *the same object*; install is idempotent; uninstall without install is a no-op; a failed install does not leak the wrapper. |

### `test_parallelize_scan.py` — 19 tests, CPU

The graph_trainer lane's `parallelize_fn`, `parallelize_llama_scan`.

| Group | Protects |
|---|---|
| Step order (4) | All four steps run, in order, with compile last, and FSDP ignores the stacked parameters. Order is load-bearing — a step that runs after compile does nothing. |
| Memory policy (4) | `recompute_body` is on for `full` and off for `default` and `none`, and a compile config without a memory policy defaults to off. This is the single line that connects the CLI flag to activation checkpointing on this lane. |
| Guards (7) | Scan here requires simple FSDP and raises a named `ValueError` without it — **and the guard runs before the model is rewritten**, so a rejected config leaves nothing half-transformed. A seq len the mesh cannot divide is rejected, each of TP / CP / PP is rejected by name, and the guards run before any torch_tpu patch is installed. |
| Public API (4) | The documented entry points import, the aliases are the underlying functions, everything in `__all__` resolves, and the package docstring's worked example **binds against the real signature**. The fourth is there because the first three all passed while the example raised `TypeError` — name resolution is not the same as being callable the way the docs call it. |

### `test_torch_tpu_fix.py` — 9 tests, CPU

The temporary torch_tpu workaround in `scan/torch_tpu_fix.py`, and in
particular its **end-of-life detector** — the thing that is supposed to make
the patch no-op itself once torch_tpu ships the fix.

Writing this module found a real defect: the detector did not fire, so the
patch would have kept reapplying itself over a fixed torch_tpu indefinitely.
That is the whole reason a temporary patch needs a test — its failure mode is
silently outliving its reason to exist, and nothing else would have caught it.

| Group | Protects |
|---|---|
| Defect detection (5) | The detector sees the defect, and sees the upstream fix. Reformatting the defective source alone does **not** read as a fix. A missing implementation or an unreadable source reports *unknown*, not *fixed* — guessing "fixed" would silently disable the patch on a broken wheel. |
| Install (3) | Installing twice is idempotent, install reports `False` rather than raising when torch_tpu is absent, and the replacement mirrors the upstream signature. |
| Contract (1) | Run against whatever torch_tpu is actually installed: the patch state must match the detected defect state. This is the test that flips on its own when the wheel is upgraded. |

### `test_workarounds.py` — 4 tests, CPU

`workarounds.py`'s RoPE TPU optimization patch, over three dtype /
complex-input combinations: the patched form matches the stock one. The fourth
case is a gather-versus-2D-indexing benchmark that records why the patch
exists.

### `graph_trainer/tests/test_tpu_graph_trainer_config.py` — 9 tests, CPU

Two groups, and the second is the important one.

`TpuGraphTrainerConfigTest` (5) covers the config translation: the CLI memory
policy is propagated, an unsupported one is rejected at conversion rather than
at run time, and qwen3-8b explicitly takes the default graph policy.

**`ScanWiringTest` (4)** asserts that `to_tpu_graph_trainer_config` installs the
scan `parallelize_fn` onto the spec copy it builds. That copy is taken *before*
`train.py` installs its own `patched_parallelize` on the original spec, so
anything written onto the original afterwards never arrives here — which is
exactly how `--tpu-config.use-scan` came to be accepted, logged as enabled, and
ignored, and how a 70B run came to ask for 166.72 GiB of a 94.75 GiB device
while every log line said scan was on.

It compares the scanned conversion against the un-scanned one rather than
against a function name, because the failure being guarded is "the graph_trainer
default survived" and only the two side by side show that. It also pins that
`TPU_SCAN_PEEL_FIRST_LAYER` is resolved into the closure **at conversion time**
— a value read late would attribute the peel's cost to the wrong arm. Three of
the four cases fail against the pre-fix wiring.

### `graph_trainer/tests/test_tpu_graph_trainer_unit.py` — 6 tests, **TPU** (5 of 6 also pass on CPU)

The TPU graph-trainer glue: `to_graph_trainer_config` resolves the right target
class, the Pallas CE-loss override applies, `make_fx` traces a synthetic model,
and forward + backward + AdamW + gradient clipping trace as one graph.

Only that last case needs a device — it traces the functional AdamW, which
`torch.optim` refuses to run with `capturable=True` off an accelerator. The
other four are device-independent. Six is also one more than this module
defines: it imports `BaseAcceleratorDeviceTest` by name, so unittest collects
the base class as a test too.

### `test_scan_fsdp_lowering.py` — 11 tests, CPU (gloo)

The 136.9 GB trap. SimpleFSDP installs its all-gather as a class property, so a
single `getattr` on a stacked parameter gathers **all N layers at once**.

The load-bearing test is `test_forward_does_not_read_through_the_parametrization`
— both spellings are numerically correct, so nothing fails if this regresses;
only a structural assertion catches it. `test_in_body` then asserts the
collectives appear **inside** the loop body and not above it.

Also: the layer axis is never sharded, the stack shards on dim 1, plans that
would shard the scan axis or use a multi-dimensional mesh are rejected,
sharded placements plan a gather while replicated ones plan a cast only, and
an unsharded parameter still honours the compute dtype.

### `test_rope_real_arithmetic.py` — 4 tests, CPU

`view_as_complex` becomes `view_as_complex_copy` inside a scan body, which has
no autograd formula for complex outputs — tracing fails on every attention
layer. The patch rewrites the complex multiply in real arithmetic.

Asserts the rewrite matches the complex form **bit-for-bit in fp32**, in
forward and in backward, and that installing it is idempotent. Bit-exactness
is the right bar here: PyTorch's complex multiply evaluates exactly the
expression the patch writes out, so anything less would mean a real difference.

### `test_scan_lowering.py` — 10 tests (1 skipped), **TPU**

That the structural claims survive export to StableHLO: exactly one
`while` loop, one copy of the layer in the body, and a lowered module whose
size is independent of depth. Plus `test_device_forward_matches_device_unrolled`
for numerical agreement on device.

`test_short_argument_list_raises` asserts a bad `fx_to_mlir` argument list
raises rather than aborts. Against a torch_tpu that still routes this through
`ABSL_CHECK_OK` the failure is SIGABRT, which cannot be asserted on from
inside the process it kills — so the call is made in a **subprocess** and the
test only reads its exit code. This is the regression test for torch_tpu commit
"return, don't CHECK, on a bad traverse_and_compile argument list"; expect it
to skip against a stock wheel and pass against a patched one.

Two details of that probe are load-bearing, and both were got wrong once. It
runs from `setUpModule`, because the subprocess needs a TPU of its own and this
process owns the device the moment any test touches it — after that the probe
can only fail with "The TPU is already in use". And it matches the **message**,
not just the exception type: that already-in-use failure is also a
`RuntimeError`, so a bare `except RuntimeError` scores it as "the wheel
validates properly" and the test then makes the aborting call in-process,
taking the whole module down with it.

### `test_scan_compiled_hlo.py` — 5 tests, **TPU**

`test_scan_lowering.py` above asserts one `stablehlo.while` in the module
torch_tpu *emits*. That is the compiler's input, not its output, and XLA is free
to unroll a `while` — a `while` it unrolls costs exactly what no scan costs:
compile time linear in depth, and every layer's activations live at once. Until
this module, the only check on the compiled side ran at 70B on four slices, as a
grep for ` while(` in the payload's HLO dump. Right assertion, wrong scale. This
makes it on one chip in a minute.

The mechanism is XLA's own dump. `XLA_FLAGS=--xla_dump_to=…` is read when PJRT
initialises and cannot be set afterwards, so each compile runs in a subprocess
with the flag in its environment and the parent reads the files back — which is
why the compiles happen in `setUpModule`, the last moment the chip is free.

| Test | Protects |
|---|---|
| Loop survives | The compiled HLO still contains a `while(`. |
| Unrolled control | The *unscanned* stack compiles to **zero** `while(`. Without it the first test cannot distinguish "scan made a loop" from "this toy model always makes one". |
| Count is depth-invariant | Depth 4 and depth 16 give the same `while` count — catches the partial unroll a full-unroll check misses. |
| Body is depth-invariant | The loop-bearing module grows 1.05× while depth grows 4×. A single peeled-out layer would land far above the 2× bar. |

Line counts are taken over loop-bearing modules only. A compile emits a dozen
unrelated modules — RNG, dtype casts, the `.cpu()` transfer — whose sizes drift
with anything and would swamp a whole-dump count.

### `test_scan_reverse.py` — 13 tests (1 skipped), **TPU**

The 7.75 GiB `torch.flip` elision. Three claims in dependency order:

1. `scan_reverse_op` means what it says — checked against a **plain Python
   loop**, so the definition is pinned by something with no shared machinery.
   `test_ys_is_indexed_by_input_position_not_iteration_order` isolates the one
   property the whole rewrite rests on.
2. The TPU lowering (`ScanDirection.kReverse`) agrees with that definition —
   a separate implementation, and nothing in the stack exercised it before.
3. **The rewrite changes no gradient.** A memory optimisation that perturbs
   gradients is not an optimisation.

Plus: the forward scan is unaffected, install/uninstall are idempotent and
restore the stock backward, and the elision **does not eat a `torch.flip` the
model itself performs** — it matches on tensor identity, not shape, and there
is a test with a body that flips its own carry to catch a regression to shape
matching.

One skip: eager autograd-through-`scan` on device is broken upstream
(unrelated to this work), so device-side gradient equivalence is established
by the training run's loss trajectory instead.

### `compiled_optimizers/tests/test_functional.py` — 13 tests, CPU + TPU

The functional AdamW used by the compiled-optimizer path. Matches
`torch.optim.AdamW` over multiple steps; does not mutate its inputs; returns
fresh tensors rather than aliases; advances the step counter by exactly one;
casts grads to the param dtype. Plus parameter lookup that **bypasses the
parametrization getter** (the same class-property hazard as the FSDP tests) and
resolves nested FQNs, and buffer writeback that writes params and both
optimizer-state buffers in place so buffer identity survives.

### `graph_trainer/tests/test_tpu_graph_trainer_distributed.py` — 3 tests, **TPU × 8**

The only module that runs the real trainer entry point across real chips. Two
parameterized cases, three steps each on a debug model, plus the inherited
availability check.

`qwen3_simple_fsdp` is the pre-existing case: graph trainer + `aot_fx_trace` +
simple FSDP.

**`llama3_scan_compiled_optimizer` is the 70B target configuration**, shrunk to
a debug model and one host — graph trainer, `aot_fx_trace`, `use_scan`,
`compile_optimizer`, simple FSDP and `graph_trainer_memory_policy=full`
together. Every one of those is covered somewhere above in isolation; nothing
ran them as a group, and the group is the only form the target actually uses. So
until this case, the first check that the combination trains at all was a 70B
job on four slices — half an hour, a cluster, and a reviewer reading a log. It
now costs 70 seconds. Loss falls 7.67 → 6.83 → 6.30.

The case passes both simple-FSDP spellings. The graph-trainer lane's scan guard
reads `parallelism.use_simple_fsdp` (the in-body gather is a simple_fsdp
parametrization plan) while the TPU trainer reads its own `tpu_config` copy;
`llama3_70b` sets the first from its config function and `llama3_debugmodel`
does not, so the target's shape has to be asked for explicitly here.

## What these tests do *not* cover

- **End-to-end performance.** Deliberately. Step time is measured on a cluster
  and lives in the benchmark record, not here.
- **70B.** Everything above runs at toy, debug or 8B scale. The distributed
  case proves the target composition *runs*; it says nothing about what it costs
  at 80 layers and seq 8192.
- **Loss parity against a full unrolled training run.** The CPU semantic tests
  cover the math at small scale; full-scale parity is a training-run
  observation, and at 70B it is still open.
- **Multi-host.** The distributed module uses the eight chips of one host. Slice
  topology, cross-host collectives and the rendezvous are cluster-only.
- **TP / CP / PP**, which the feature rejects rather than supports.

## See also

- [`../scan/README.md`](../scan/README.md) — what the feature does and how the
  mechanisms work.
