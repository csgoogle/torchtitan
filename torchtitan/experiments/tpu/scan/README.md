# Scan over layers (llama3, TPU)

A decoder is N structurally identical blocks. torchtitan runs them as a Python
`for` loop, so the compiler sees N copies of one program. This package replaces
that loop with a single `scan` higher-order op, so the stack lowers to one
`stablehlo.while` whose body does not grow with depth.

Turn it on with `--tpu-config.use-scan`.

## Why

Three costs scale with N when the loop is unrolled, and stop scaling when it
is not:

- **Compile time and compiler memory.** The XLA module carries N copies of the
  block. At 70B that is 80 copies of a program that only needs to exist once.
- **Weight residency.** Unrolled, the natural place for FSDP's all-gather is
  above the loop, so every layer's gathered weights are live at once. With one
  loop body the gather happens *inside* it, and only the current layer's
  weights are resident.
- **Per-copy graph work.** Parallelization passes, sharding propagation and
  scheduling all run over N copies.

What it does *not* change is the math. The scanned stack computes the same
function as the unrolled one, in the same order, at the same precision.

## How to turn it on

```bash
# graph-trainer lane -- joint fwd + loss + bwd (+ optimizer) via aot_fx_trace
--tpu-config.use-scan --tpu-config.use-graph-trainer
```

Constraints. Each is enforced with a specific error at parallelization time,
never a silent fallback:

- llama3 on the graph-trainer lane only. Any other model or lane is rejected;
  a lane that merely ignored the flag would run unrolled while logging that
  scan was on.
- Activation checkpointing is `compile.memory_policy=default|full`. See
  *Activation checkpointing*.
- TP, CP and PP are unsupported. The stacked parameters carry a leading layer
  axis that the existing parallel plans do not describe.

One environment variable tunes it:

| Variable | Default | Effect |
|---|---|---|
| `TPU_SCAN_PEEL_FIRST_LAYER` | auto | Override the peel heuristic. See *Pallas kernels*. |

## Adopting it in another model

`scan.over_layers` is the whole API. It is a context manager because the three
steps have to interleave with the caller's own parallelization:

```python
from torchtitan.experiments.tpu import scan

with scan.over_layers(
    model,
    parallel_dims=parallel_dims,
    dp_mesh=dp_mesh, dp_mode=dp_mode, mp_policy=mp_policy,
    recompute_body=True,
) as stack:
    # your own annotation / TP / FSDP passes go here; they see the stack,
    # not the N per-layer modules.
    model = apply_simple_fsdp(
        model, ..., ignored_params=set(stack.stacked_parameters())
    )
# leaving the block installs the in-body all-gather on the stacked parameters
```

Entering stacks the layers; leaving shards the stack. Your FSDP wrap has to run
between the two, and it has to skip the stacked parameters -- they shard on
dim 1 and are gathered one layer at a time inside the body instead. Getting
that order wrong is silent, which is why it is a `with` block and not two
calls.

Your model must satisfy:

- `model.layers` is a `ModuleDict` keyed `"0"`, `"1"`, ... of identical blocks;
- the block's forward takes the hidden state first and loop-invariant arguments
  after it -- `(h, attention_masks, positions, segment_ids)` for llama3. The
  stack forwards the trailing arguments unchanged, so that list can grow without
  touching this package; an argument that differs per layer could not, and would
  have to be stacked into the scan's `xs`;
- the layer loop iterates `model.layers.values()` (`ScannedLayerStack.values()`
  yields `[self]`, so the loop runs once over the whole scan and the decoder
  needs no edit);
- no complex-dtype gradient ops in the block. See *RoPE*.

`torchtitan/experiments/tpu/graph_trainer/parallelize_scan.py` is the worked
example.

## How it works

One file per problem. Each is one commit on this branch.

### The stack -- `layer_stack.py`

`ScannedLayerStack` replaces `model.layers`. It deliberately quacks like the
`ModuleDict` it replaces -- `.values()` returns `[self]`, and it is itself
callable with the block signature -- so **the decoder's layer loop is not
touched**: it runs exactly once and the single "layer" it gets is the whole
scanned stack. Reverting is `model.layers = original`.

Parameters are stored **stacked**: one `[n_layers, *param_shape]` tensor per
layer-relative name, which is what `scan` consumes as `xs`. The per-layer
modules survive as *shadows* with their parameters moved to `meta`, so they
cost no HBM. Layer 0 doubles as the template that `torch.func.functional_call`
is invoked on with a slice of the stack.

The shadows are not dead weight -- they carry each layer's `_param_init`
closures, so initialization binds layer *i*'s parameters to a view of
`stacked[i]` and runs the stock per-layer initializer straight into the stack.
Initialization is bit-identical to the unrolled model's, per layer, with no
extra allocation.

### FSDP inside the body -- `layer_stack.py`

This is the part that matters most at scale, and the easiest to break silently.

SimpleFSDP installs its all-gather as a **class property** per parameter name.
A single `getattr` on a stacked parameter therefore gathers *all N layers at
once* -- at 70B, 136.9 GB of bf16 weights against a 96 GB budget -- before the
loop the gather was supposed to happen inside.

So `ScannedLayerStack.forward` reads through `self._parameters[name]`, never
`getattr`, and hands `scan` the local shard. `_AllGatherReduceScatter` gathers
one layer's slice inside the combine_fn, so the collective lowers into the
`stablehlo.while` body and peak gathered weight is one layer rather than eighty.

Calling SimpleFSDP's `ReplicateComputation` there instead -- which would delete
the hand-rolled collective -- was tried and does not work. Two HOP-contract
limits block it, both measured on 8 ranks:

* a closed-over `DeviceMesh` is lifted as an extra HOP operand, and scan's
  additional-inputs contract accepts only `Tensor | int | SymInt`, so the mesh
  arrives as a `FakeScriptObject` and the call dies;
* `replicate_compute` reads `x._spec` to choose its branch, which Dynamo cannot
  wrap while tracing the sourceless combine_fn -- *"SourcelessBuilder.create
  does not know how to wrap DTensorSpec"*.

`torch.distributed._functional_collectives` is used instead: its group is a
plain `str`, which bakes into the graph as a constant. The **dtypes** are still
SimpleFSDP's, read off `module._simple_fsdp_parametrization_list[0]` rather than
restated, so the deliberate bf16-gather / fp32-reduce asymmetry has one
definition.

`shard_stacked_layers` is the wrap-time half: `data_parallel(..., shard_dim=1)`
for the sharding (dim 0 is the scan axis), then `slice_specs()` to reject any
mesh or placement the in-body gather cannot express before the first step runs.

### Activation checkpointing -- `recompute.py`

`torch.utils.checkpoint` inside a scan body is a **silent no-op**, and the
reason is narrower than "the context is lost". Vanilla AC is tagged by Dynamo,
in `TagActivationCheckpoint.tag_nodes` (`torch/_higher_order_ops/wrap.py:288`).
The HOP re-traces the body through `materialize_as_graph`, whose inner tracer
runs under `@torch._dynamo.disable(recursive=True)`
(`torch/_higher_order_ops/utils.py:1271`). Dynamo never sees the body, so the
tagging pass never runs; you get min-cut's saved set with no warning and no
error. This is why the block-wrapping AC policies are rejected rather than
ignored.

That matters because without a recompute tag the partitioner saves the
*gathered* weights across the forward/backward boundary and hands back
everything the in-body gather just saved.

What does reach the body is the tracer's ambient metadata. `fx/proxy.py` copies
`fx_traceback.current_meta["recompute"]` onto every node the tracer creates,
including the body's, so `body_checkpointed()` -- a `with` block around the
`scan` call -- is ordinary AC expressed one level down. The tag is
`CheckpointPolicy.MUST_RECOMPUTE`, which `ban_recomputation_if_allowed` honours
ahead of both the collective ban and the allowlist.

The window is also open while the outer tracer creates the `scan` node and the
body's placeholders, and neither may keep the tag: a tagged placeholder makes
min-cut raise `NetworkXError`, and a tagged `scan` node recomputes the whole
layer loop in backward -- silently, while the saved-set metric improves.
`install()` wraps the module-level `min_cut_rematerialization_partition` (which
`HopGraphMinCutPartitioner` calls directly; there is no per-HOP hook) to strip
those two back off. Note that Dynamo's own tagger has the same rule and solves
it at the source: `tag_nodes` tags only `call_function` / `call_method` /
`call_module`, never a placeholder.

Upstream asks 1 and 2 delete this file and its monkeypatch respectively.

### Backward direction -- `reverse.py`

`scan`'s backward is a reverse scan. Upstream has no `reverse` argument, so
`ScanAutogradImpl.call_backward` emulates one with `torch.flip` on the saved
forward intermediates. On TPU that flip is a materialized copy of the entire
residual stack -- at 8B, seq 8192, local batch 4 it is a second
`bf16[31, 4, 8192, 4096]` buffer, 7.75 GiB, 28 % of the arena, holding nothing
that is not already resident. Upstream knows: the call site carries
`# TODO: torch.flip copies the tensor, we should optimize it away`.

torch_tpu can lower a genuine reverse scan and writes its `ys` aligned to the
input index rather than to iteration order, so the input flip and the output
flip cancel and both can be dropped. This file installs a sibling HOP that does
that. It is a sibling rather than a new argument because `ScanOp.__call__` and
`ScanOp.gen_schema` share a fixed four-argument signature that nine dispatch
registrations are written against.

Upstream ask 3 deletes this file.

### Pallas kernels and `peel_first_layer` -- `api.py`

A Pallas kernel enters the graph as a `func.call` to a symbol defined at module
scope. When the body is cloned into the loop region, that callee can become
unresolvable. Peeling layer 0 out of the scan leaves one un-scanned copy of the
block at module scope where the call resolves, at the cost of one unrolled
layer. `resolve_peel_first_layer` turns it on when a splash-attention kernel is
configured; `TPU_SCAN_PEEL_FIRST_LAYER=0|1` overrides it either way.

### RoPE -- `torchtitan/models/common/rope.py` (outside this package)

`ComplexRoPE.apply_rotary_emb` used `view_as_complex`, which inside a scan body
becomes `view_as_complex_copy` -- and that has no autograd formula for complex
outputs, so tracing failed on every attention layer. The rotation is now written
in real arithmetic: `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`, same pairing,
same interleaved output layout. PyTorch's complex multiply evaluates exactly
that expression, so the two agree bit-for-bit in fp32.

`torchtitan/models/common/decoder.py` carries the other out-of-package change:
tied embeddings are re-pointed through `_parameters`, so the tie survives a
SimpleFSDP parametrization.

## torch_tpu dependency

Executing a scan on TPU needs fixes that live in torch_tpu, not here. **This
requires a torch_tpu newer than 0.1.1**; on 0.1.1 it does not run at all, so the
dependency is declared rather than patched around. Three defects:

| # | Where | Symptom |
|---|---|---|
| A | `_internal/compile/scan.py`, `_handle_scan_impl` | The body's operands are passed to `fx_to_mlir` as live device tensors rather than placeholders, so compiled mode `CHECK`s and the process aborts ("argument tensor has deferred ops"). |
| B | `_internal/compile/dispatch_scan.cc`, `PyCreateScanOp` | The deferred buffer list is created directly instead of through the helper that records the node on the events queue, so the node never enters a traversal. **C++-only -- not shimmable from Python.** |
| C | `_internal/compile/dispatch_scan.cc`, output shapes | For a rank-0 combine_fn output, the declared `ys` shape comes from the torch dummy while the accumulator is built from the body's MLIR result type, so the compiled `stablehlo.while` returns one rank too many and output reconstruction dies in `force_strides`. Not exotic: SDPA returns `philox_seed`/`philox_offset` as rank-0 int64 and scan stacks every saved intermediate. |

Because B is in compiled C++, no configuration of this feature has ever run on
stock 0.1.1. Carrying a Python shim would only make a hard dependency look
optional, so there is none.

## Upstream asks

Four changes to PyTorch would delete code here. **None is required to use the
feature** -- everything below works today against stock torch. They are listed
so a reviewer can see which lines are a design and which are a workaround with
a known expiry.

Verified against `pytorch/pytorch` v2.12.0 (`0d62256a`); line numbers are that
tag.

| # | Upstream change | Deletes here | Prod | Test |
|---|---|---|---:|---:|
| 1 | AC works inside a HOP body | `recompute.py` | 149 | ~170 |
| 2 | `partition_fn` argument | `recompute.install()`'s monkeypatch | (in 1) | -- |
| 3 | `reverse=` on the scan HOP | `reverse.py` | 359 | ~220 |
| 4 | `ScriptObject` in HOP additional inputs | `_AllGatherReduceScatter` | ~50 | -- |

Total if all four land: ~558 of 1,197 production lines, and both of the tree's
two monkeypatches.

1. **Make `torch.utils.checkpoint` work inside a HOP body.** Today it is a
   silent no-op, because vanilla AC is tagged by Dynamo
   (`torch/_higher_order_ops/wrap.py:288`) and `materialize_as_graph` traces the
   body under `@torch._dynamo.disable(recursive=True)`
   (`torch/_higher_order_ops/utils.py:1271`). This is not a metadata-plumbing
   bug: the meta is never set rather than dropped in transit. The fix touches
   every HOP -- `cond`, `while_loop`, `scan`, `associative_scan`,
   `flex_attention` -- so it wants a design discussion before a diff. The
   reproducer is fifteen lines on CPU and needs no accelerator. Deletes
   `recompute.py`.

   Worth testing before asking: the *selective* AC path
   (`create_selective_checkpoint_contexts`) does not go through Dynamo. It tags
   from a `TorchDispatchMode` that writes `fx_traceback.current_meta`
   (`torch/utils/checkpoint.py:1345`), and `materialize_as_graph` calls
   `_disable_current_modes()` *before* running the body, so a mode entered
   inside the body should survive. Tagging by dispatch would also hit exactly
   the body's `call_function` nodes -- no placeholders, no outer `scan` node,
   so no strip. If that works, most of `recompute.py` goes away with no
   upstream change at all.

2. **Let `HopGraphMinCutPartitioner.create_partitioned_graph` take a
   `partition_fn`.** `HopJointGraph.partition` already accepts one
   (`torch/_higher_order_ops/partitioner.py:280`); only the caller two frames up
   hardcodes `min_cut_rematerialization_partition` (lines 357-380). Adding the
   argument and forwarding it is ~10 lines, purely additive, and turns
   `recompute.install()`'s process-global monkeypatch into a function argument.
   This is the smallest and most landable of the four, and it is worth doing
   even if ask 1 never happens.

3. **A `reverse` argument on the scan HOP.** `scan()` the Python API already
   takes `reverse` (`torch/_higher_order_ops/scan.py:86`) but implements it by
   flipping the inputs and the outputs (lines 194, 219); the HOP itself has a
   fixed four-argument signature (line 229). Deletes `reverse.py` and fixes the
   `torch.flip` TODO upstream already carries. Most of this file's 359 lines are
   scaffolding that exists only because the patch lives outside torch, so the
   upstream version is the smaller one.

4. **Let the HOP's additional inputs carry a lifted `ScriptObject`.** A
   `DeviceMesh` would then survive into the body, and SimpleFSDP's own
   `ReplicateComputation` could replace `_AllGatherReduceScatter`. Needs a
   Dynamo fix alongside it: `SourcelessBuilder` cannot wrap a `DTensorSpec`, and
   `replicate_compute` reads one to pick its branch.

## Tests

All of these run on TPU:

```bash
cd torchtitan/experiments/tpu/tests
IMG=<image> TT=<repo root> NPROC=1 ./run_in_container.sh \
    torchtitan.experiments.tpu.tests.test_scan_stack        --device=tpu
IMG=<image> TT=<repo root> NPROC=8 ./run_in_container.sh \
    torchtitan.experiments.tpu.tests.test_scan_fsdp         --device=tpu --num_devices=8
IMG=<image> TT=<repo root> NPROC=8 ./run_in_container.sh \
    torchtitan.experiments.tpu.tests.test_scan_recompute    --device=tpu --num_devices=8
IMG=<image> TT=<repo root> NPROC=1 ./run_in_container.sh \
    torchtitan.experiments.tpu.tests.test_scan_reverse      --device=tpu
IMG=<image> TT=<repo root> NPROC=1 ./run_in_container.sh \
    torchtitan.experiments.tpu.tests.test_scan_integration  --device=tpu
IMG=<image> TT=<repo root> NPROC=1 ./run_in_container.sh \
    torchtitan.experiments.tpu.tests.test_rope_real_arithmetic --device=tpu
```

| Module | Protects |
|---|---|
| `test_scan_stack` | one copy of the body in the graph; stacked parameters; init parity; the `.values()` compatibility contract |
| `test_scan_fsdp` | gathers inside the body and none above it; real 8-rank gradient parity against a global-batch reference |
| `test_scan_recompute` | the gathered weights do not cross the forward/backward boundary; the ambient tag reaches the body and is stripped off the nodes that must not carry it |
| `test_scan_reverse` | reverse-scan semantics vs a Python loop; the TPU lowering agrees; the rewrite is gradient-neutral |
| `test_scan_integration` | the adoption API's ordering and guards; `--tpu-config.use-scan` reaching both lanes |
| `test_rope_real_arithmetic` | the real rewrite is bit-identical to the complex multiply |

Most of what these catch fails **silently** -- a gather that moved above the
loop, a recompute tag that got dropped, a config flag that was accepted and
ignored. Read the module docstring before changing any of this code.

## Known gaps

- TP / CP / PP unsupported.
- llama3 only; other models need the `over_layers` contract above.
- Requires torch_tpu > 0.1.1.
- At 70B the memory argument holds, but end-to-end step time was ~1.6 % slower
  than the unrolled arm on the predecessor branch, traced to a single async
  layout conversion of the splash-attention `dV` gradient (~4.93 % of the step)
  that only the scan arm emits: `dV` is a loop-carried accumulator of the
  attention backward's inner `while`, and layout assignment cannot rewrite a
  carried buffer's layout. `dQ`/`dK` avoid it because the RoPE-inverse fusion
  absorbs the relayout for free. Not collective overlap -- FSDP collectives
  measured *cheaper* under scan. No fix landed.
