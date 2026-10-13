---
name: tpu_model_onboarding
description: Onboard a model to TorchTitan's TPU lane (torchtitan/experiments/tpu): place the code, prove parity with its reference, make it train, then climb the optimization ladder one flag at a time and ship a target-size recipe. Use when a model must train on TPU, whether it exists upstream in torchtitan/models or is ported from HF transformers, Megatron, MaxText or custom code. Not for debugging an existing model's numerics (/numerics_debugging) or for GPU-only work.
---

# Model onboarding (TPU lane)

Onboarding is a ladder. Rung 1 makes the model correct on the plain SDPA path
and trains it end to end; it is a PR of its own, because everything later is
measured against it. Each later rung is one feature behind one flag, with one
parity test and one end-to-end case, as one commit. Muse Glimmer is the worked
example: #521 (rung 1), then #529, #530, #532, #534.

Read before starting: `torchtitan/experiments/tpu/README.md` (environment,
VM and cluster setup, launch commands), `torch-tpu-optimization-guide.md`
(the levers, in order), `.claude/rules/models.md`, `.claude/rules/experiments.md`.

Templates in the tree; read them instead of expecting this skill to restate them:

- wrap an upstream model: `experiments/tpu/muse_glimmer/` (PR #521)
- port a model: PR #491 (`experiments/tpu/examples/gemma4/`, open at the time
  of writing); its `TestGemma4HFReferenceParity` is the parity-test template
- recipe README: `experiments/tpu/recipes/ironwood/qwen3-30b-a3b/tpu7x-4x4x4/README.md`

## Step 0: three questions before any code

1. **Where does it run?**
   - A. this machine is the TPU VM: run directly; check the chips with
     `torch.accelerator.device_count()` (not `torch.tpu.*`, which raises) first.
   - B. a TPU VM over ssh: `gcloud compute tpus tpu-vm ssh <name> --zone=<z>
     --command='...'`; you need name, zone, project and the checkout path, and
     you sync code and fetch `--job.dump_folder` yourself.
   - C. GKE via Cluster Toolkit (`gcluster`): submit per the recipe README;
     you need project, cluster, image registry, whether you may build images,
     and a dispatch budget agreed before the first job.

   Rung 1 and the kernel rungs close on A or B with 1-4 chips; only Step 3
   needs C. Under C, every check that can run on CPU or one chip runs there
   first, and each dispatch passes the gate in Step 3. Never fall back
   silently: no chip, failed ssh or missing image means stop and say so.
2. **What is the reference, and how do the weights arrive?** HF transformers
   (safetensors, reference runs in-process on CPU) is the common case; upstream
   torchtitan GPU checkpoints run in-process too. Megatron/NeMo, MaxText and
   custom code usually cannot run in the test process: a script next to the
   test runs the reference on a pinned tiny config and dumps weights, tokens,
   logits, loss and every gradient with provenance, and the test consumes the
   dump. Either way the weight mapping is a key-and-layout translation under
   `BaseStateDictAdapter` (HF-named; the names are legacy) and the test checks
   keys both ways. If an HF checkpoint already trains through
   `experiments/tpu/transformers_modeling_backend`, the onboarding may be a
   recipe, not a port.
3. **Wrap or port?** Upstream has the model: add only `experiments/tpu/<name>/`
   (registry wrapper, `config_registry.py`, `infra/parallelize.py`, `tests/`).
   Otherwise: `experiments/tpu/examples/<name>/`, built on
   `torchtitan/models/common/` (subclass, do not copy; a copied block needs a
   one-line reason in the PR). `examples/` is a review requirement.

Then read the architecture off the reference and write it into the PR
description (`references/pr_template.md`): layer pattern, norms, RoPE, QK-norm,
softcaps, tying, MoE shapes, and anything the layer forward mutates (it blocks
scan). Every constant gets a source line.

## Step 1 (PR 1): correct on SDPA, trains end to end

- Model code, plain SDPA, no Pallas kernels, no GraphTrainer. `debugmodel`
  flavor plus the real sizes; a `<name>_debugmodel` recipe. Register in
  `experiments/tpu/train.py` and `train_minimal.py`.
- Write it scan-ready even if scan ships later: stock `Decoder.forward` loop
  (`check_layers_are_stackable` passes); no module-attribute or in-place
  writes in the layer forward (MoE counters and routing metrics behind config);
  no complex ops (real-arithmetic RoPE); heterogeneous layers compiled per
  block with `dynamic=False`.
- The reference-parity test, the one test that matters: tiny config covering
  every layer type; unit-initialized scales perturbed (`1 + 0.1 * randn`, or a
  dropped multiply is invisible); geometry that exercises the architecture
  (window shorter than the sequence, more experts than top-k, tying if the
  real model ties); weights through the adapter with no missing or unexpected
  keys; one forward and backward; logits, loss and every parameter gradient
  compared. `BaseAcceleratorDeviceTest`, so CI runs it on CPU and on one chip.
  Start from #491's bounds, tighten if there is margin, record the measured
  error next to the bound. Do not pad the PR with per-component ladders or
  sharding tests.
- State-dict round trip, flavor checks, and a mask-semantics test if the
  architecture has one (sliding window against the reference's definition).
- End-to-end cases in `experiments/tpu/tests/test_train_distributed.py`
  (FSDP2 and SimpleFSDP, 3 steps on `c4_test`, context longer than any window).
- A run on the Step 0 target that meets the bar below.

## The bar for "it trains" (every rung; recorded in the PR with its command)

```
Steps          >= 10; judge from steps past warmup. 2-4 steps is not a verdict.
Loss           decreasing, no NaN/inf, no spikes
Determinism    two runs with --debug.seed=42 --debug.deterministic give
               bit-identical loss and grad_norm, read from the TensorBoard event
               file (tag loss_metrics/global_avg_loss); stdout prints 5 digits
Oracle         record the deterministic step-1 loss and grad_norm; every later
               rung compares against it
Structure      per-layer patch counts == layer count; heads / kv_heads / vocab ==
               the checkpoint; the HLO dump (--extra-libtpu-init-flags=
               xla_dump_to=<dir>) contains the custom_calls of enabled kernels.
               Eager log lines do not fire under compile; trust the HLO.
Modes          eager / per-block compile / GraphTrainer losses within ~1e-4 on
               debugmodel
```

Every change after PR 1 gets a numerics class: **bit-identical** (step-1
oracle equal); **rounding-order** (fp32 round-off from tiles, fusion or
reduction order; fine for speed, but a frontier recipe needs the 200-step
reference trajectory, compared on post-warmup medians); **semantic** (bf16
gradient reduce, dropped tokens, another balancer: a separate labelled recipe,
not a lever).

## Step 2: the ladder, one rung per commit

Order, with the guide section in parentheses:

```
1  per-block compile + SimpleFSDP + mp policy (4, 5)   PR 1
2  GraphTrainer wrapper (8)                            3-step loss vs per-block compile
3  splash / tokamax attention (4)                      vs SDPA on a packed batch, both
                                                       layer types, gradients too
4  fused cross entropy (4)                             vs fp32 F.cross_entropy
5  MoE kernels, if MoE (10)                            grouped mm vs torch._grouped_mm,
                                                       empty groups included
6  batch / context to saturation (3)                   recipe change; re-measure
7  scan + offload, or graph split (9)                  scan vs unrolled 4-layer proxy,
                                                       5 steps; only once it no longer fits
8  libtpu flag bundle (9)                              bit-identical step-1 oracle
9  tile sweeps (4, 10)                                 rounding-order; per generation
```

Each rung: the flag stays off in the recipe until Step 3; a parity test
against the path it replaces plus the control that proves the test can fail
(drop the document mask, turn the cap off, watch the metric move); one
end-to-end case with the flag on; a commit message with the numerics class,
before/after step time and MFU on the same recipe and chip count, and the HLO
evidence (custom_call present, segment ids among its operands). Rung k passes
on the target before rung k+1 starts, and the ladder table in the PR is
updated after every rung. One branch; whether it ships as one PR or stacked
PRs is the human's call at the end.

The ladder is done when the profile shows no exposed gap that a listed lever
addresses, every enabled feature has its numerics class, the trajectory check
passed and the README table is filled. "Out of ideas" is not a stop condition;
an unmeasured lever on the list is not either.

## Step 3: target size

Production recipe plus `recipes/<generation>/<model>/<topology>/README.md`:
launch block, every libtpu flag with a one-line reason, `## Benchmark Results`
from `--tpu_config.enable_mfu_benchmark` over >= 10 post-warmup steps. Only
then flip the kernels on by default. The 200-step reference run is an input
you ask for (pinned optimizer, batch, reduce dtype, seed, data, balancer), not
something the TPU side produces.

Gate before any cluster dispatch, every time: the exact config parses
(`ConfigManager().parse_args([...])`); it runs 3 steps on CPU or one chip; the
image carries the commit you think it does; the optimizer path is exactly one
of eager / `compile_optimizer` / `optimizer_in_graph`;
`num_tokens_per_train_step == ranks x num_tokens_per_microbatch_per_dp_rank`;
command-line flags are in the recipe (left on the command line they get
forgotten). When the budget is spent, write the ladder table and stop.

## Rules

- Run touched test modules the way CI does before pushing: `python -m <module>
  --device=tpu` on a chip, `--device=cpu` for the CPU shard. CI classifies a
  module by its base class: `BaseAcceleratorDeviceTest` -> single chip,
  `BaseDistributedDeviceTest` in a `*_distributed.py` -> multi-chip, anything
  else -> the CPU shard, where a `skipUnless(tpu)` test skips forever, green.
  Take the device from the base class; never detect it yourself.
- No core edits for the model. No Meta copyright headers on new files.
- No performance claim from fewer than 10 steps, from a single run, or before
  the numerics class is known.
- The PR description is the onboarding record and the only state that
  survives between sessions; keep it current on every push. No handover files
  in the repo.
- `references/gotchas.md` before debugging anything that "trains fine, but";
  `/numerics_debugging` for a first-diverging-op search.
