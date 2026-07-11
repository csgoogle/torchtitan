# Guide to optimizing LLMs on TorchTPU

> ⚠️ **Work in progress.** TorchTPU and its tooling are evolving quickly — some recommendations here
> will change, and defaults/APIs may shift under you. Treat this as current best-practice guidance,
> not a stable contract, and always measure on your own setup.

A practical guide to getting good MFU (model FLOPs utilization) out of a dense decoder-only LLM on
**TorchTPU** — PyTorch running on Cloud TPU. It can execute op-by-op (convenient for development)
or, under `torch.compile`, fuse ops into compiled XLA programs (the performant path). Getting good
MFU is largely about staying in the compiled path and feeding it well — the playbook below.

**Worked example.** For a demonstration of high-performance dense LLM training on TorchTPU, see the
**llama3 sample** — `llama3` (configs in
[`llama3/config_registry.py`](llama3/config_registry.py), e.g. `llama3_8b()`) — which runs the
llama3 model ([`../../models/llama3/model.py`](../../models/llama3/model.py)) with the compile +
fused-kernel + FSDP recipe described here. Use it as the reference for how the pieces fit together.

It references two companion libraries you will lean on for kernels and parallelism:

- **[TorchTitan](https://github.com/pytorch/torchtitan)** — its `experiments/tpu` kernels
  (splash attention, fused cross-entropy), `apply_compile` helper, SimpleFSDP, and the
  `num_flops_per_token` MFU formula.
- **[tokamax](https://github.com/openxla/tokamax)** — mosaic-TPU Pallas kernels (e.g. the fused
  linear + softmax-cross-entropy kernel) that TorchTitan's TPU kernels build on.

---

## Scope — what applies to what

Most of this guide is **general TorchTPU** optimization (true of any model); a few sections are
**dense-specific**. MoE is out of scope for now — see §10 (TBD).

| Section | Applies to |
|---------|-----------|
| §1 Mental model | General |
| §2 Profile first | General |
| §3 Arithmetic intensity (dim, batch) | Dense-specific |
| §4 Reduce dispatch (compile, fused kernels) | General |
| §5 Memory & parallelism (SimpleFSDP + mixed precision) | General |
| §6 Precision, layout | General |
| §7 The dispatch ceiling | General |
| §8 Mixture-of-Experts | MoE — TBD |

---

## 1. The mental model: compile, or be dispatch-bound  *(general)*

Every XLA program TorchTPU runs has to be dispatched by the host, and the TPU can sit idle between
launches. **Run uncompiled and each op becomes its own program** — so a naive model spends much of
every step idle between many tiny programs (the eager, dispatch-bound regime). `torch.compile` compiles
multiple ops into one program per compiled region, collapsing those gaps.

The practical consequence: for a small or naively-configured model you will often find the TPU is
**idle a large fraction of every step**, not compute-limited. When the matmuls actually run, the
MXU is typically fine; the time is lost *between* programs.

So the entire optimization strategy reduces to two ideas:

1. **Raise the work done per launched program** — bigger matmuls (model dim), bigger batch. This
   raises the compute-to-dispatch ratio.
2. **Reduce the number of host dispatches** — compile transformer blocks; use fused kernels.

Everything below is an instance of one of these two.

> **Eager mode is for development and debugging only.** Run uncompiled while you are building the
> model, checking numerics, or stepping through a bug — the eager per-op dispatch makes errors easy
> to localize. But an uncompiled model is **dispatch-bound and leaves most of the TPU idle**; it is
> *not* a performance configuration. **For any performance-sensitive run the model must be compiled
> with `torch.compile` (see §4).** Treat "compile it" as step zero of performance, not an optional
> lever — everything else in this guide assumes a compiled model.

---

## 2. Always profile first — classify the bottleneck  *(general)*

Do not optimize blind. Capture an **xprof** profile over a few steady-state steps and read the
**idle vs compute** split from the op-profile:

- **High idle (dispatch-bound)** — the common case for small models / small batch. The levers in
  §3–§4 apply directly.
- **Matmuls at a high fraction of peak (compute-bound)** — you are near the eager ceiling for that
  config; further gains need bigger problems (§3), and the residual dispatch glue is the ceiling (§7).
- **High HBM traffic / OOM (memory-bound)** — reach for the fused CE kernel and FSDP (§4, §5).

The single most useful number is the idle percentage. Optimize it down, or accept it as the
ceiling and move on.

---

## 3. Raise arithmetic intensity (the biggest lever)  *(dense-specific)*

### Batch size

A TPU is easily starved at small batch. Increasing the per-device batch is often the **single
largest MFU lever at small scale** — it directly raises the compute per launched program. Push
batch as high as HBM allows; the fused CE kernel (§4) frees room for a bigger one.

### Size the model to the hardware

TPU MXUs are large systolic arrays (typically 128×128 or 256×256 depending on generation). FLOPs
per matmul scale with `dim²` while the per-program dispatch cost is roughly constant, so a larger
hidden dim does much more useful work per launch.

- Use a **hidden dim that is a multiple of the MXU tile** (256 is a safe multiple across
  generations) — see *dimension alignment*.
- In practice, moving from a small debug dim to a production dim (e.g. 1024 → 2048) can **roughly
  double MFU** purely from better MXU utilization, before any other change.
- Keep scaling depth/dim until you hit the single-chip HBM ceiling, then sync memory relief (§4,
  §5) before scaling further.

---

## 4. Reduce dispatch and materialization  *(general)*

### torch.compile — mandatory for performance

**Compilation is essential; for models running on TPU, it is the critical step that transforms a
slow debug configuration into an optimized, performant execution.**
An uncompiled model is dispatch-bound (see §1); `torch.compile(backend="tpu")` fuses the ops
inside its captured region into one XLA program, collapsing many eager dispatches into few. This is
the single highest-impact change and typically yields a solid double-digit-percent speedup on its
own. Do this first.

**Compile at transformer-block granularity** — compile each block module (TorchTitan's
`apply_compile` does exactly this). This is the robust, low-risk granularity: it fuses the ops
*within* a block while keeping activation checkpointing *between* blocks, so memory stays bounded.

**Always compile with `dynamic=False`.** LLM training runs at a fixed batch × sequence length, so
the shapes are static; `dynamic=False` tells Dynamo to specialize to those static shapes and skip
dynamic-shape guards:

```python
compiled_block = torch.compile(block, backend="tpu", dynamic=False)
```

On a TPU/XLA backend, dynamic shapes are actively harmful — they force padding and recompilation
and defeat fusion. **Do not leave `dynamic` at its default (`None`)**: Dynamo can then silently
switch a region to dynamic shapes after a recompile, quietly tanking performance. If your shapes
genuinely are fixed (they should be for training), `dynamic=False` is always correct.

**Be careful with whole-model compilation.** Capturing the entire stack as one graph is appealing
but has real failure modes on TorchTPU:

- Without activation checkpointing inside the graph, all layers' activations are live at once →
  **compile-time / runtime OOM** at any real depth. (You cannot easily put `checkpoint()` inside a
  single compiled TorchTPU graph — the RNG-state save/restore does not trace cleanly.)
- The token-embedding backward can bind incorrectly when compiled — leave the **embedding eager**.
- Lazily-created buffers (e.g. RoPE cos/sin) must be **materialized before tracing**, or the
  graph's input set shifts and recompilation/errors follow.
- And critically: even when whole-model compile *works*, it collapses only the transformer.
  The optimizer step, the loss, the embedding, and layout ops remain separate eager programs — so
  it does **not** remove the whole-step dispatch idle (see §6). Prefer per-block compile.

### Deferred-and-fused eager mode (a lighter-weight fallback)

TorchTPU also offers a **deferred-and-fused** execution mode that sits between plain op-by-op
execution and full `torch.compile`. Instead of dispatching each op immediately, it **defers** ops
and lets XLA **fuse across op boundaries**, cutting some of the per-op dispatch overhead without
requiring a captured graph.

**How to enable it** — there are 2 ways of doing this:

1. Set the env var **before `torch_tpu` is imported** (it's read at libtpu init):

   ```bash
   export TPU_DEFER_AND_FUSE=1
   ```

   ```python
   # or, in Python, before the first `import torch_tpu`:
   import os
   os.environ["TPU_DEFER_AND_FUSE"] = "1"
   ```

2. Wrap the region you want deferred-and-fused (e.g. the whole training loop) in the execution-mode
   context:

   ```python
   from torch_tpu._internal import execution_mode as em

   with em.set_eager_mode(em.EagerMode.DEFER_AND_FUSE):
       train()
   ```

**It can be combined with `torch.compile`.** The two are not mutually exclusive: keep your
transformer blocks under `torch.compile` and run the training loop inside the deferred-and-fused
context — the compiled regions still execute as compiled programs, while the mode defers and fuses
the *un-compiled* glue around them (the eager ops between compiled regions, and anything Dynamo
broke out of the graph). Use it as a fallback where you can't compile, and as a complement to shave
dispatch off the eager remainder where you can.

When it helps:

- Layered on top of `torch.compile` to reduce dispatch on the eager ops the compiled graph doesn't
  cover.
- Code that **can't be `torch.compile`d cleanly** — heavy graph breaks, data-dependent control flow,
  or Python that Dynamo won't trace — but where you still want dispatch overhead reduced.
- Quick experiments where you want some fusion without wiring up compilation.

Caveats:

- It is **not a substitute for `torch.compile`** for peak performance — compilation fuses more
  aggressively into larger programs. Compile your blocks first; use defer-and-fuse as a fallback
  where you can't compile, or layered on top to cover the eager remainder.
- Some op patterns don't lower cleanly under fusion (e.g. certain unaligned dynamic-update-slice
  fusions can fail at the libtpu level). If you hit a fusion crash, disable it for that model and
  fall back to `torch.compile` or plain execution.
- **Always measure.** On a model already covered by per-block `torch.compile`, defer-and-fuse
  typically adds little, because the compiled path already fuses the compute.

### Fused kernels (TorchTitan + tokamax)

Pull in the TPU Pallas kernels rather than hand-rolling — they are bridged into TorchTPU as
`custom_call`s and both fuse compute and cut dispatches, and both compose with compilation and with
FSDP. TorchTitan's **llama3** TPU config enables both — see
[`llama3/config_registry.py`](llama3/config_registry.py), where the `llama3_8b()` / `llama3_70b()`
configs set `use_splash_attention_kernel` and `use_loss_kernel`.

Both are Pallas kernels whose performance depends on their block/tile sizes: **the defaults are a
starting point, not an optimum — tune them per model shape and TPU generation, gated by the VMEM
budget (below). Sweep and measure.**

#### Fused linear + cross-entropy

tokamax mosaic kernel
([`kernels/linear_softmax_cross_entropy_loss.py`](kernels/linear_softmax_cross_entropy_loss.py)),
wired into training by [`loss.py`](loss.py) (`pallas_cross_entropy_loss` /
`build_cross_entropy_loss`). Fuses the `lm_head` matmul with softmax-cross-entropy so the full
`[batch, seq, vocab]` logits tensor is **never materialized**. Its biggest value is **memory** — at
large vocab/batch it removes a huge activation and unblocks a larger batch (feeding §3); the
throughput effect grows with batch and is neutral when the logits are already small.

**Tile sizes** — three block sizes in
[`LossKernelConfig` (`tpu_job_config.py`)](tpu_job_config.py): `loss_b_block_size` (default
**1024**), `loss_h_block_size` (default **512**), `loss_v_block_size` (default **2048**).
**Constraint:** the flattened token count (`batch × seq`) must be a multiple of `loss_b_block_size`,
or the kernel silently falls back to XLA — keep it divisible by 1024. Tune `loss_v_block_size`
against your vocab size.

#### Splash attention

A flash-style TPU attention kernel
([`kernels/splash_attention.py`](kernels/splash_attention.py)), patched into a model's attention by
[`workarounds.py`](workarounds.py) (`use_splash_attention_patch`). It matters most at long sequence
length and under memory pressure; at short sequence it is often a wash against TorchTPU's built-in
flash SDPA, so measure before adopting.

**Tile sizes** — nine block-size knobs in
[`SplashAttentionKernelConfig` (`tpu_job_config.py`)](tpu_job_config.py), each defaulting to
**512**: `sa_block_q`, `sa_block_kv`, `sa_block_dkv`, `sa_block_kv_compute`, `sa_block_q_dkv`,
`sa_block_kv_dkv`, `sa_block_kv_dkv_compute`, `sa_block_q_dq`, `sa_block_kv_dq`. **Meaningful
default: 512 across the board.** Larger blocks (1024/2048) can help at long sequence *if they fit
VMEM*; smaller blocks (256) cut VMEM but usually underperform and can trip kernel constraints.
Retune `sa_block_q` / `sa_block_kv` first.

#### VMEM budget — set it for v6e

Pallas tile sizes are bounded by the per-core **scoped VMEM limit**, and the default is
conservative, so the kernels above can't use their best tiles until you raise it. On
**v6e (Trillium)** set it via the libtpu init arg:

```bash
LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536"
```

**Meaningful v6e default: `65536` KiB (64 MiB)** — the value TorchTitan's Trillium (v6e) recipes
converged on (older baselines used `131072` / 128 MiB; 64 MiB is the current recommendation). Raise
it if a kernel reports insufficient VMEM for a larger tile; lower it if you hit VMEM-limit
rejections at startup. This is **generation-specific** — other TPU generations cap at different
values, so verify the value is accepted at init rather than assuming it transfers.

---

## 5. Memory and parallelism  *(general)*

### SimpleFSDP for sharding

For models (or batch/sequence lengths) that don't fit one chip, shard with **SimpleFSDP**
(TorchTitan's DTensor-based FSDP: `data_parallel(model, mesh, mode=...)`):

- `fully_shard` (ZeRO-3-like) shards params **and** optimizer state across chips — this is what
  makes large models trainable that plain DDP OOMs on.
- `hybrid_shard` (HSDP, a 2-D mesh: replicate across groups, shard within) is the multi-host
  choice.
- SimpleFSDP composes with `torch.compile`, which lets the compiler overlap the DTensor
  collectives with compute — combining FSDP with per-block compile is meaningfully faster than
  FSDP alone.

**Why SimpleFSDP over FSDP2 here?** SimpleFSDP expresses sharding purely as **DTensor
parametrizations riding DTensor autograd** — the all-gather is a `redistribute` in forward and the
reduce-scatter is implicit in backward, with **no module hooks and no explicit collective calls**.
That makes it **compiler-native**: the collectives live inside the traced graph, so `torch.compile`
can see, schedule, and overlap them with compute. FSDP2 (`fully_shard` module wrapper) drives its
collectives from pre/post-forward **hooks**, which are harder to trace cleanly and cause graph
breaks — the compiler can't overlap them as well. On the compile-first TorchTPU path, SimpleFSDP's
traceable design is the reason FSDP + per-block compile actually overlaps (and is much simpler:
a thin parametrization, not a wrapper with hook/state machinery).

### Use a mixed-precision policy

Run FSDP with a mixed-precision policy — the standard, correct setup is:

```python
MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
```

- `param_dtype=bfloat16` — parameters are all-gathered and used for compute in **bf16** (the
  matmuls run in bf16, which is what your MFU denominator assumes).
- `reduce_dtype=float32` — the gradient reduce-scatter accumulates in **fp32**, avoiding bf16
  cross-shard accumulation error.

This keeps effective master params/grads fp32-accurate while compute is bf16. **Do not** drop
`reduce_dtype` to bf16 to save bandwidth — that is a model-quality regression, not a perf lever.

---

## 6. Precision, layout, and the small stuff  *(general)*

- **Mixed precision — keep norms in fp32.** The bf16-compute / fp32-accurate-params setup itself is
  the FSDP `MixedPrecisionPolicy` (§5); without FSDP, use `torch.autocast`. The §6-specific point is
  a code-level gotcha: if you cast to bf16 *manually* (not via autocast), a bf16 input against an
  fp32 norm weight can raise a dtype-mismatch — compute **layer/RMS norm in fp32** and cast the
  result back. (Autocast keeps norms in fp32 already.)
- **Align dims to the MXU tile** so matmuls tile cleanly and don't pad. The MXU is 128×128 or
  256×256 depending on generation; **multiples of 256** are a safe choice across generations.
- **Avoid host syncs in the step loop.** In eager mode, anything that reads a device tensor on the
  host — `.item()`, `.cpu()`, `print(tensor)`, Python `if tensor > x`, logging the loss every step —
  forces a device→host sync that stalls dispatch and can dominate a small step. Log/inspect every N
  steps, keep metrics on-device, and detach reductions from the hot path.
- **Strip per-op Python overhead in benchmark runs.** If your code runs per-op runtime checks
  (e.g. shape/type validators), disable them for perf runs — they add host overhead in eager mode.
  Compilation bypasses them inside compiled regions anyway.

---

## 7. The dispatch ceiling — maximize compiled coverage  *(general)*

Even after per-block compile and fused kernels, TorchTPU has a residual dispatch floor: the
**optimizer step, the loss, the embedding, and layout ops remain separate programs**, and the
host-side gaps between whatever is *not* inside a compiled region set a structural MFU ceiling. You
can shrink that surface, but you cannot compile it away by only capturing the transformer.

Concretely: collapsing the transformer into a single compiled graph does **not** reduce the idle
*fraction* — the idle isn't located in the transformer, it's in everything *around* it.

**So the way to raise the ceiling is to maximize compiled coverage and minimize the eager
surface:**

- Compile every transformer block (§4) — that's the bulk of the FLOPs.
- Use the fused CE kernel (§4) so the loss is a single `custom_call` instead of a chain of eager
  ops.
- Prefer a fused kernel over an eager op sequence wherever one exists (attention, etc.).
- Keep batch/dim large (§3) so the compiled regions dominate the step and the fixed eager glue is a
  smaller fraction of wall-clock.

Whatever training-step glue cannot be folded into a compiled region is your model's eager ceiling.
On a well-sized dense model with per-block compile + fused CE you will typically land in the
low-to-mid tens of percent MFU; pushing further means shrinking that eager surface.

---

## 8. Mixture-of-Experts (MoE)  *(MoE-specific — TBD)*

> **Status: TBD.** This section is a placeholder for MoE-specific guidance. The general sections
> (§1–§2, §4–§8) apply unchanged to MoE models on TorchTPU; the items below are the *additional*
> levers a sparse/routed model introduces. They are not yet benchmarked on the TorchTPU eager lane
> and will be filled in with measured guidance.

Everything in §1–§8 still holds — MoE is just as dispatch-bound, and per-block compile + fused CE
+ SimpleFSDP + mixed precision all still apply. What changes is the FFN: instead of one dense MLP
per layer, a router sends each token to its top-k of N experts. That adds these concerns:

- **Expert FFN matmul kernel** *(TBD)* — the per-expert FFN is a *ragged/grouped* matmul (variable
  tokens per expert), not a dense GEMM. Use a grouped / segment matmul Pallas kernel (tokamax
  provides these) rather than a Python loop over experts, which would explode the eager dispatch
  count. This is usually the single largest MoE lever.
- **Dispatch / combine** *(TBD)* — the token→expert gather (dispatch) and expert→token scatter
  (combine) around the expert FFN are a major cost and a heavy source of layout/`as_strided`
  traffic. Prefer fused dispatch/combine kernels; profile this bucket specifically.
- **Gate + up fusion** *(TBD)* — fuse the expert FFN's gate and up projections into a single
  matmul + chunk (as in dense SwiGLU) to cut expert-matmul dispatches.
- **Expert parallelism (EP)** *(TBD)* — for large expert counts, shard experts across chips (EP)
  and compose it with FSDP on the non-expert params (a 2-D or 3-D mesh). How EP collectives
  (all-to-all for dispatch/combine) overlap with compute on the TorchTPU lane is the open question.
- **Router precision & load balance** *(TBD, quality-sensitive)* — routing in fp32; capacity
  factor / token-drop tradeoffs affect both throughput **and model quality** — treat any change
  that drops tokens as a correctness question, not a free perf lever.
- **MFU accounting** *(TBD)* — use **active** params (params actually touched per token = shared +
  top-k experts), *not* total params, in `num_flops_per_token`. Reporting total-param MFU for a
  sparse model overstates utilization.

Until this section is filled in with measurements, start MoE work from the general recipe (§1–§8),
add a grouped-matmul expert kernel + fused dispatch/combine first, and profile the dispatch/combine
bucket before anything else.

---

## Quick-start checklist  *(dense)*

Eager (uncompiled) is only for steps 1's profiling and for debugging — every performance number
below assumes a compiled model.

1. Profile a few steady-state steps; read the idle fraction.
2. **Compile — mandatory.** `torch.compile` each transformer block with
   `backend="tpu", dynamic=False` (TorchTitan `apply_compile`). Do this before tuning anything
   else; an uncompiled model is a debug config, not a performance one.
3. Size the model: hidden dim ≥ a large MXU-aligned value; push batch until HBM-bound.
4. Add the fused kernels (tokamax / TorchTitan): the fused CE kernel (frees memory → bigger batch)
   and splash attention (long sequence). **Tune their tile sizes** (§4) — the defaults aren't
   optimal.
5. If it doesn't fit one chip, SimpleFSDP `fully_shard` with
   `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`; use `hybrid_shard` multi-host.
6. bf16 compute / fp32 master; norm in fp32; dims multiple of 256; shape-checks off for benchmarks.
7. Re-profile. If matmuls are near peak and the only idle left is the training-step glue, you're at
   the torch_tpu eager ceiling (§7) — raise it by shrinking the eager surface, not by more tuning.

---

### Operational notes

- **Build TorchTPU wheels with optimizations on** (`-c opt` for a Bazel build). A fastbuild wheel
  can have *dramatically* slower eager dispatch and will silently ruin every measurement here.
- **After any hard crash / killed process, clear the stale libtpu lock** (e.g.
  `rm -f /tmp/libtpu_lockfile`) or the next run may fail to initialize the device.
