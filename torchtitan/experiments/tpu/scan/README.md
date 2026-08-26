# Scan over layers (llama3, TPU)

A decoder is N structurally identical blocks. torchtitan runs them as a Python
`for` loop, so the compiler sees N copies of one program. This feature replaces
that loop with a single `scan` higher-order op, so the stack lowers to one
`stablehlo.while` whose body does not grow with depth.

Enable it with `--tpu-config.use-scan`.

## Why

Three costs scale with N when the loop is unrolled, and stop scaling when it
is not:

- **Compile time and compiler memory.** The XLA module carries N copies of the
  block. At 70B that is 80 copies of a program that only needs to exist once.
- **Weight residency.** Unrolled, the natural place for FSDP's all-gather is
  above the loop, which means every layer's gathered weights are live at once.
  With one loop body, the gather happens *inside* it, so only the current
  layer's weights are resident.
- **Graph-level work that is per-copy rather than per-layer** — parallelization
  passes, sharding propagation, and scheduling all run over N copies.

What it does *not* change is the math. The scanned stack computes the same
function as the unrolled one, in the same order, at the same precision.

## Measured

| Model | Hardware | seq | local batch | Unrolled | Scan | Delta |
|---|---|---|---|---|---|---|
| llama3 8B | v6e-256 | 8192 | 4 | 37.44% MFU | 39.89% MFU | **+2.45 pp** |
| llama3 8B | v6e-8 | 8192 | 1 | 26.51% MFU | 29.51% MFU | **+3.00 pp** (graph-trainer lane) |
| llama3 8B | v6e-8 | 8192 | 1 | 26.51% MFU | 27.41% MFU | **+0.90 pp** (fullgraph lane) |

Loss is identical to the unrolled run at step 1 and tracks to ~1e-3 by step 20,
which is bf16 reduction-order noise.

The two single-host rows are at local batch **1**, not 4: on one v6e-8 host the
scan arms do not fit at batch 4 — the graph-trainer arm hits
`RESOURCE_EXHAUSTED` after step 1, where the unrolled arm runs (34.83% MFU).
That is the expected direction for this lane and the reason the headline row is
the 256-chip one, where the per-host share is small.

All three unrolled/scan pairs above are same-tree, same-harness, same-day
comparisons. The two v6e-8 rows were re-measured against the current tree on
2026-08-25; loss, grad-norm and peak memory came back bit-identical to the
original 2026-08-20 readings on all four ablation arms, and the graph-trainer
delta moved from +3.30 pp to +3.00 pp — the whole of that shift is in the
unrolled arm, whose own three-run spread on the day was 0.04 pp.

### Are the two arms activation-checkpointing-matched?

Yes, at the transformer block — which is where all but ~2% of the flops are.
MFU is `num_flops_per_token × tokens/s / peak_flops` and `num_flops_per_token`
is an analytic constant that **excludes recompute**, so an arm that recomputes
less would be credited the same numerator for less work. That is worth checking
rather than assuming, because the two arms reach full AC by different routes.

The unrolled arm gets `FullAC`, which is `ptd_checkpoint_wrapper(early_stop=
False)` around each block: the whole block is recomputed in backward and only
its input is saved. The scanned arm cannot use that wrapper —
`torch.utils.checkpoint` inside a scan body is a silent no-op, since the HOP
re-traces the body and drops the recompute tags. It gets the equivalent from
`install_scan_collective_recompute(recompute_body=True)` in `fsdp.py`,
which tags every value-producing forward node in the body `MUST_RECOMPUTE`
before the body's min-cut partitioner runs. `memory_policy=full` is what sets
that flag.

Both halves are observable in a run's logs, and both were checked at 8B:

- the tagging pass fires — `tagged 135 node(s) MUST_RECOMPUTE out of 262 graph
  nodes (recompute_body=True)`;
- and the saved-set audit (`SCAN_FSDP_AUDIT_SAVED_SET=1`) shows what actually
  survives the cut: **12 tensors** — the carry in, the carry out, the seven
  weight shards, the RoPE table, the position tensor, and one scalar. No
  intra-block activation crosses it. That is the `FullAC` saved set plus the
  carry-out.

So the recompute sets match, and the residual mismatch runs the other way: the
graph trainer's `full` policy also tags the *non-layer* region —
`Layer non-layer: 0 MUST_SAVE, 169 RECOMPUTE` — so the scanned arm additionally
recomputes the embedding, the final norm and the output projection, which
`FullAC` leaves alone. **+2.45 pp is therefore a slight under-statement of the
scan win, not an upper bound**, and the 70B deficit below is a slight
over-statement.

The peak-HBM gap between the arms (9.36 GiB vs 2.01 GiB at 8B; 59.72 vs 8.23
GiB at 70B) is not evidence against any of this. It is not a recompute-set
difference: what survives the cut stacks into one contiguous `[n_layers, ...]`
`ys` buffer where `FullAC` keeps `n_layers` independently-placeable tensors.
And the metric is narrower than a footprint on both lanes — 2.01 GiB is below
the 8B optimizer-state floor alone, so it is a reservation reading, not a total.

One limit: the two log lines above were read at 8B. The 70B runs did not
capture trace-level logging, so the same check has not been made there.
`recompute_body` is a boolean derived from a flag and the block structure is
identical, so this is a small gap rather than an open question.

70B is not in this table on purpose. Scan runs at 70B and the memory argument
holds, but the end-to-end step time is not a win there yet: measured 1.6 %
slower per step than the unrolled arm at seq 8192, local batch 3 on tpu7x-32,
reproduced in both rung orders within a single job. Do not read the 8B result
as a 70B result. The cause is identified and specific — see *The 70B layout
regression* below.

### The 70B layout regression

On tpu7x-32 at seq 8192, local batch 3, the scan arm spends **4.93 % of the
step** in a single async layout conversion that the unrolled arm never emits.
That one instruction accounts for **102 % of the `async-done` gap** between the
two arms (+4.85 pp, within which every other bucket nets to −0.08 pp).

Note the denominator: 4.93 % is the fraction of the *step*, and it is about
three times the 1.6 % end-to-end deficit quoted above — the two are not the same
number and this instruction is not "the whole step-time gap". The compute
buckets outside `async-done` net *favorably* for the scan arm by ~3.3 pp, which
is why a 4.93 % penalty surfaces as only a 1.6 % end-to-end deficit. Read the
other way: removing the conversion does not bring the scan arm to parity, it
puts it roughly **3 % ahead** of the unrolled baseline.

It converts the **dV gradient** — `bf16[3, 8, 8192, 128]`, produced by the
splash-attention backward kernel — into the layout the fused-QKV gradient
concat wants. The reason it exists only under scan is structural:

- The backward kernel runs in a `while` loop over the batch, nested inside the
  scan's backward `while`. dV is a **loop-carried accumulator** of that inner
  loop, and layout assignment cannot rewrite the layout of a carried buffer —
  the loop's carry signature fixes it. Unrolled, the carried layout happens to
  be the one the consumer wants, and the conversion is free.
- dQ and dK reach the same concat through the RoPE-inverse elementwise fusion,
  which absorbs the layout change for free. RoPE applies to Q and K only, so dV
  is the one gradient with no intervening op to fold the relayout into.
- The peeled first layer (`TPU_SCAN_PEEL_FIRST_LAYER`) emits the *cheap*
  conversion, since it is outside the scan — which confirms the expensive form
  is a consequence of the loop structure and not of a missing compiler feature.

Worth stating explicitly because it is the intuitive guess: **it is not
collective overlap.** Measured on the same traces, FSDP collectives are
*cheaper* under scan than unrolled — unhidden all-gather wait 0.30 % vs 0.52 %,
reduce-scatter 1.59 % vs 1.82 %. Every large weight all-gather in the scan body
is fully hidden behind the MLP matmuls. Rescheduling is also not the fix: the
window around the conversion does not contain enough compute to hide it behind.
The conversion has to be made cheap or removed, which means giving layout
assignment somewhere to absorb it.

## How to turn it on

Two lanes support it. Pick one:

```bash
# fullgraph lane
--tpu-config.use-scan --tpu-config.compile-mode=fullgraph

# graph-trainer lane (joint fwd+loss+bwd via aot_fx_trace)
--tpu-config.use-scan --tpu-config.use-graph-trainer
```

Constraints, all enforced with a specific error at parallelization time rather
than a silent fallback:

- llama3 only.
- `compile_mode=layer` is rejected — compiling each block separately is exactly
  what the scan removes.
- `activation_checkpoint` must be `none` or `full` (see *Activation
  checkpointing* below).
- TP, CP and PP are not supported. The stacked parameters carry a leading layer
  axis that the existing parallel plans do not describe.

Two environment variables tune it:

| Variable | Default | Effect |
|---|---|---|
| `TPU_SCAN_PEEL_FIRST_LAYER` | auto | Override the peel heuristic (see *Pallas kernels*) |
| `TPU_SCAN_REVERSE_BACKWARD` | off | Use a native reverse scan in the backward (see *Backward direction*) |
| `SCAN_FSDP_AUDIT_SAVED_SET` | off | Log which tensors cross forward→backward |

## How it works

### The stack (`layer_stack.py`)

`ScannedLayerStack` replaces `model.layers`. It deliberately quacks like the
`ModuleDict` it replaces — `.values()` returns `[self]`, and it is itself
callable with the block signature — so **the decoder's layer loop is not
touched**: it runs exactly once and the single "layer" it gets is the whole
scanned stack. Reverting is `model.layers = original`. (`decoder.py` does carry
one unrelated change on this branch, to how tied embeddings are re-pointed
under a SimpleFSDP parametrization; it has nothing to do with the loop.)

Parameters are stored **stacked**: one `[n_layers, *param_shape]` tensor per
layer-relative name, which is what `scan` consumes as `xs`. The per-layer
modules survive as *shadows* with their parameters moved to `meta`, so they
cost no HBM. Layer 0 doubles as the template that `torch.func.functional_call`
is invoked on with a slice of the stack.

The shadows are not dead weight — they carry each layer's `_param_init`
closures, so initialization binds layer *i*'s parameters to a view of
`stacked[i]` and runs the stock per-layer initializer straight into the stack.
Initialization is bit-identical to the unrolled model's, per layer, with no
extra allocation.

### FSDP inside the body (`fsdp.py`)

This is the part that matters most at scale, and the easiest to break silently.

SimpleFSDP installs its all-gather as a **class property** per parameter name.
So a single `getattr` on a stacked parameter gathers *all N layers at once* —
at 70B that is 136.9 GB of bf16 weights against a 96 GB budget — before the
loop the gather was supposed to happen inside.

The fix has two halves, and both are required:

1. `forward` reads through `self._parameters[mangled]`, never `getattr`, and
   hands `scan` the local shard. The combine_fn gathers one layer's slice,
   so peak gathered weight is 1.71 GB rather than 136.9 GB.
2. The gather and its view-like descendants are forced to `MUST_RECOMPUTE`, or
   the partitioner saves the *gathered* weights across the forward→backward
   boundary and hands back the memory the first half just saved.

Both spellings of (1) are numerically correct, so nothing fails if this
regresses — which is why `tests/test_scan_fsdp_lowering.py` asserts the
structure directly. Set `SCAN_FSDP_AUDIT_SAVED_SET=1` to log the saved set.

The all-gather is bf16 and the reduce-scatter is fp32, matching SimpleFSDP's
existing asymmetry rather than introducing a new one.

### Activation checkpointing

`torch.utils.checkpoint` inside a scan body is a **silent no-op**. The HOP
re-traces the body through `materialize_as_graph`, which erases the recompute
tags; you get min-cut's saved set with no warning and no error.

So the block-wrapping AC policies cannot be used, and passing one raises.
Activation memory for the scanned stack comes instead from an FX tagging pass
over the body's joint graph, which is why only `full` and `none` are accepted —
`full` is the only policy with such a pass.

### Backward direction (`reverse.py`)

`scan`'s backward is a reverse scan. Upstream has no `reverse` argument, so
`ScanAutogradImpl.call_backward` emulates one by calling `torch.flip` on the
saved forward intermediates. On TPU that flip is a materialized copy of the
entire residual stack — at 8B, seq 8192, local batch 4 it is a second
`bf16[31,4,8192,4096]` buffer, 7.75 GiB, 28% of the arena, holding nothing that
is not already resident. Upstream knows: the call site carries
`# TODO: torch.flip copies the tensor, we should optimize it away`.

torch_tpu can lower a genuine reverse scan and writes its `ys` aligned to the
input index rather than to iteration order, so the input flip and the output
flip cancel and both can be dropped. `TPU_SCAN_REVERSE_BACKWARD=1` installs a
sibling HOP that does this. It is a sibling rather than a new argument because
`ScanOp.__call__` and `ScanOp.gen_schema` share a fixed four-argument signature
that nine dispatch registrations are written against.

### RoPE (`rope_real_arithmetic.py`)

`ComplexRoPE.apply_rotary_emb` uses `view_as_complex`, which inside a scan body
becomes `view_as_complex_copy` — and that has no autograd formula for complex
outputs, so tracing fails on every attention layer.

The patch rewrites the same complex multiply in real arithmetic:
`(a + bi)(c + di) = (ac − bd) + (ad + bc)i`, same pairing, same interleaved
output layout. PyTorch's complex multiply evaluates exactly that expression, so
the two agree bit-for-bit in fp32 — asserted by
`tests/test_rope_real_arithmetic.py`.

### Pallas kernels and `peel_first_layer`

A Pallas kernel enters the graph as a `func.call` to a symbol defined at module
scope. When the body is cloned into the loop region, that callee can become
unresolvable. Peeling layer 0 out of the scan leaves one un-scanned copy of the
block at module scope where the call resolves, at the cost of one unrolled
layer. The heuristic turns it on when a JAX kernel is in the body;
`TPU_SCAN_PEEL_FIRST_LAYER` overrides it.

## torch_tpu dependency

Executing a scan on TPU needs fixes that live in torch_tpu, not here.
`torch_tpu_fix.py` carries Python-side workarounds so this work is not
blocked on an external release; it is **temporary** and should be deleted once
the fixes land upstream. Its docstring diagnoses each defect.

One defect is C++-only and cannot be worked around from Python: `PyCreateScanOp`
creates its deferred buffer list directly instead of going through the helper
that records the node on the events queue, so the node never enters a traversal.
This only bites lanes that replay ops eagerly; both compiled lanes build the
scan inside compiled mode, where every buffer is a placeholder.

## Tests

```bash
python -m torchtitan.experiments.tpu.tests.test_scan_over_layers      # CPU
python -m torchtitan.experiments.tpu.tests.test_scan_memory_policy    # CPU
python -m torchtitan.experiments.tpu.tests.test_scan_fsdp_lowering    # CPU (gloo)
python -m torchtitan.experiments.tpu.tests.test_rope_real_arithmetic  # CPU
python -m torchtitan.experiments.tpu.tests.test_scan_lowering         # TPU
python -m torchtitan.experiments.tpu.tests.test_scan_reverse          # TPU
```

The CPU suites prove the structural claims — one copy of the body in the FX
graph, the all-gathers inside it and none above it, the saved set. The TPU
suites prove those survive the torch_tpu export and that the lowered loop
agrees numerically with the unrolled one.

`test_scan_lowering` skips one test against a torch_tpu that still routes a bad
`fx_to_mlir` argument list through `ABSL_CHECK_OK`; there the failure is
SIGABRT, which cannot be asserted on from inside the process it kills.

[`../tests/README.md`](../tests/README.md) walks every module test by test and
says what each one protects — read that before changing any of this code, since
most of what these tests catch fails silently rather than loudly.

## Known gaps

- 70B end-to-end step time is not yet a win — 1.6 % slower per step. The cause
  is the dV layout conversion described under *The 70B layout regression*,
  worth ~4.93 % of the step; no fix landed yet.
- The AC match between the two arms was verified from 8B logs, not 70B ones —
  the 70B runs captured metrics only. See *Are the two arms
  activation-checkpointing-matched?*.
- On tpu7x the graph-trainer lane can abort inside the torch_tpu compile. This
  is separate from the scan defects above and still open.
- TP / CP / PP unsupported.
- `torch_tpu_fix.py` is a temporary shim.
