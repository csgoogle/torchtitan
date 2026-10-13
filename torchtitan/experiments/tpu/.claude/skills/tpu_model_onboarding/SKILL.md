---
name: tpu_model_onboarding
description: Onboard a new model to TorchTitan's TPU lane (torchtitan/experiments/tpu). Use when adding a model that must train on TPU, whether it already exists upstream in torchtitan/models or must be ported from a reference implementation (HF transformers, upstream torchtitan on GPU, Megatron/NeMo, MaxText, or your own code). Covers where the model runs, model placement, the reference-parity test, end-to-end training cases, the rung-by-rung optimization ladder that applies torch-tpu-optimization-guide.md feature by feature, and the target-size recipe with benchmark results. Don't use for general training runs, for debugging an existing model's numerics (use /numerics_debugging), or for GPU-only models.
---

# Model onboarding (TPU lane)

Onboarding is a ladder. The first rung makes the model correct on the plain
SDPA path and proves it trains end to end; that rung is a PR of its own,
because it is what every later measurement is compared against. Every
performance feature after it is one rung: one flag, one parity test against
the path it replaces, one end-to-end case, one commit. This is how Muse
Glimmer landed (#521 then #529, #530, #532, #534, two days) and the opposite
of how Gemma 4 (#491: 14 files, 16 review threads on unrelated layers, open
for weeks) went.

Read first: `torchtitan/experiments/tpu/README.md` (environment, TPU VM and
cluster setup), `torchtitan/experiments/tpu/torch-tpu-optimization-guide.md`
(what to apply, in order), `torchtitan/models/README.md` (upstream model
anatomy), `.claude/rules/models.md` and `.claude/rules/experiments.md`.

References: [run.md](references/run.md) (where it runs, how to launch, how
to read a run, what to do when it fails), [model.md](references/model.md)
(where code goes, registration, scan readiness), [tests.md](references/tests.md)
(the mandatory tests and their templates), [train.md](references/train.md)
(acceptance criteria and numerics classes), [optimize.md](references/optimize.md)
(the ladder, rung by rung), [pr_template.md](references/pr_template.md) (the
onboarding record), [gotchas.md](references/gotchas.md).

## Step 0: scope. Ask these before touching code.

1. **Where does this run?** Three answers, and the rest of the run depends on
   it (`run.md` has the table):
   - **A. this machine is the TPU VM** - commands run directly; verify the chips
     with `torch.accelerator` before anything else.
   - **B. a TPU VM you can ssh into** - commands run through
     `gcloud compute tpus tpu-vm ssh`; you need its name, zone, project and the
     checkout path, and you sync code and fetch `dump_folder` yourself.
   - **C. GKE through Cluster Toolkit (`gcluster`)** - jobs are submitted from
     the recipe's launch block; you need project, cluster, blueprint, image
     registry, whether you may build and push images, and a dispatch budget.
   Rung 1 and the kernel rungs close on A or B with 1-4 chips; only the
   target-size recipe (Step 3) needs C. If the answer is C alone, every
   CPU-closable check still runs locally first, and a cluster dispatch is
   expensive: smoke the exact config on CPU or one chip before each one. Never
   fall back silently: if A reports no chip or B's ssh fails, stop and say so.
2. **What is the reference, and where do the weights come from?** HF
   transformers is the worked example throughout this skill; it is not the
   only case. Pick the row and it decides the shape of the parity test
   (`tests.md`):

   | Reference | Weights arrive as | Reference forward runs | Parity test shape |
   |---|---|---|---|
   | HF transformers | safetensors, HF key names | in-process on CPU | in-process: load through the adapter, run both, compare |
   | upstream torchtitan (GPU) | DCP checkpoint, torchtitan names | in-process on CPU (same code) | in-process |
   | Megatron-LM / NeMo | TP/PP-sharded checkpoint, fused QKV / gate-up interleaving, own naming | usually not runnable here | golden tensors: the reference side dumps logits, loss and every gradient for a pinned tiny config; the test loads the dump |
   | MaxText / JAX | Orbax pytree | a different stack | golden tensors (in-process only if `JAX_PLATFORMS=cpu` is acceptable) |
   | custom / research code | whatever it is | depends | golden tensors by default |

   Whatever the source, the weight mapping is a key-and-layout translation
   (`model.md`), and the test counts keys both ways.
   If the source is an HF checkpoint, check first whether the Transformers
   modeling backend already trains it (`experiments/tpu/transformers_modeling_backend`,
   `--hf-model=<id>`): if it does and the goal is "train this model", the
   onboarding may be a recipe, not a port.
3. **Pick the style.**
   - **A. Wrap upstream**: the model exists in `torchtitan/models/<name>/`.
     Add only `torchtitan/experiments/tpu/<name>/` (registry wrapper,
     `config_registry.py`, `infra/parallelize.py`, `tests/`). Template:
     `experiments/tpu/muse_glimmer/` (PR #521).
   - **B. Port**: the model is not upstream. Add it under
     `torchtitan/experiments/tpu/examples/<name>/` with `model.py`,
     `state_dict_adapter.py`, `sharding.py`, `infra/parallelize.py`,
     `config_registry.py`, `tests/`, built on `torchtitan/models/common/`.
     Template: PR #491 (`examples/gemma4`). The `examples/` location is a
     review requirement: it keeps later upstreaming easy.
4. **Read the architecture and write it down** in the onboarding record
   (`pr_template.md`): layer pattern (homogeneous / sliding-global cycle /
   dense-then-MoE), norms and eps, RoPE theta and partial factor, QK-norm,
   softcaps, tied embeddings, MoE router and expert shapes, anything that
   mutates module state in forward (breaks scan). Every constant you type by
   hand gets a source line in the reference.

## Step 1 (PR 1): correct on SDPA, trains end to end

Deliverables, all in one PR:
- Model code per `model.md`, plain SDPA attention, no Pallas kernels, no
  GraphTrainer. `debugmodel` (and `testmodel`) flavors plus the real sizes in
  the model registry; a `<name>_debugmodel` recipe in `config_registry.py`.
- Registration in `experiments/tpu/train.py` (import for side effects; the
  `model_spec.name` list that routes `tpu_compile_mode`).
- The **reference-parity test** (`tests.md`): tiny config covering every
  layer type, reference weights through your adapter with no missing or
  unexpected keys, one forward and backward, logits + loss + per-parameter
  gradients compared. This is the test that matters; do not pad the PR with
  per-component or sharding tests.
- State-dict round trip, config/flavor checks, and any mask-semantics test
  the architecture needs (sliding window vs the reference's definition).
- End-to-end cases in `experiments/tpu/tests/test_train_distributed.py`
  (FSDP2 and SimpleFSDP, 3 steps of `debugmodel` on `c4_test`) with a context
  length that exercises the architecture (longer than any sliding window).
- A run per `run.md` on the target from Step 0: 10+ steps, loss falls, no
  NaN, bitwise repeat with `--debug.seed=42 --debug.deterministic`, and the
  step-1 loss / grad_norm written down as the oracle (`train.md`).

## Step 2: the ladder, one rung per commit

Walk `optimize.md`. Each rung is one commit on one branch, with: one feature
behind a config flag (kernels stay opt-in in the recipe until Step 3), a
parity test against the SDPA or fp32 path on inputs that exercise the feature,
a check that the test can fail (disable the feature's mask or cap and watch
the metric move), an end-to-end case, and the numerics class in the commit
message. Typical order: GraphTrainer wrapper, splash / tokamax attention,
fused cross entropy, MoE kernels, then scan + offload or graph split for sizes
that do not fit.

Do not start rung k+1 until rung k's "Done when" checks pass on the target.
Keep the `## Ladder` table in the onboarding record current after every rung
(`pr_template.md`); it is how a later session, or a reviewer, picks up where
you stopped. Whether the stack ships as one PR or as stacked PRs is the
human's call when the stack is done, not yours during the run; the PR
boundary that matters is the one after Step 1, because correctness and
kernels have different reviewers.

## Step 3: target size

Add the production recipe(s) and `recipes/<generation>/<model>/<topology>/
README.md` with the launch block, every libtpu flag with a one-line reason,
and a `## Benchmark Results` table. Confirm training at the target size per
`train.md` acceptance criteria, record the numerics class of each enabled
feature (bit-identical / rounding-order / semantic), and only then flip the
kernels on by default. The 200-step reference trajectory is an input you ask
for (a run of the reference implementation with the pinned settings in
`train.md`), not something you produce on the TPU side.

## Rules

- Before every push: the touched test modules CI-style
  (`python -m <module> --device=tpu` on a chip, `--device=cpu` or
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for the CPU shard). There is no lint job
  and no pre-commit config on `main`; CI is the test shards in
  `.github/workflows/tpu-test.yml`.
- No core edits for the model (`.claude/rules/experiments.md`). A copy of a
  `models/common/` block needs a one-line reason in the PR (Gemma 4: common
  `GroupedExperts` hardcodes `silu`).
- No Meta copyright headers on new files; no `dtensor` suffixes in test or
  function names.
- A parity test with no assertion that fails when the feature is off is not
  a test.
- No performance claim from fewer than 10 steps, none from a single run, none
  before the numerics class of the change is known.
- No multi-host dispatch without the pre-dispatch gate in `run.md`; stop and
  report at the budget rather than iterating on the cluster.
- The PR description is the onboarding record. Keep it current on every
  push (`pr_template.md`); nothing else (no `*_handover.md`) goes in the repo.
