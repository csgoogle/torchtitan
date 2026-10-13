---
name: tpu_model_onboarding
description: Onboard a new model to TorchTitan's TPU lane (torchtitan/experiments/tpu). Use when adding a model that must train on TPU, whether it already exists upstream in torchtitan/models or must be ported from HuggingFace or another reference. Covers model placement, the HF reference-parity test, end-to-end training cases, the sliced PR ladder that applies torch-tpu-optimization-guide.md feature by feature, and the target-size recipe with benchmark results. Don't use for general training runs, for debugging an existing model's numerics (use /numerics_debugging), or for GPU-only models.
---

# Model onboarding (TPU lane)

Onboarding is a ladder of small PRs. The first PR makes the model correct on
the plain SDPA path and proves it trains end to end; every performance feature
after that is its own PR with its own parity test and its own end-to-end case.
This is how Muse Glimmer landed (PRs #521, #529, #530, #532, #534, two days)
and what the Gemma 4 review (#491) asked for.

Read first: `torchtitan/experiments/tpu/README.md` (how to run),
`torchtitan/experiments/tpu/torch-tpu-optimization-guide.md` (what to apply,
in order), `torchtitan/models/README.md` (upstream model anatomy),
`.claude/rules/models.md` and `.claude/rules/experiments.md`.

References: [model.md](references/model.md) (where code goes, registration,
scan readiness), [tests.md](references/tests.md) (the mandatory tests and
their templates), [train.md](references/train.md) (running and acceptance
criteria), [optimize.md](references/optimize.md) (the PR ladder from the
guide), [pr_template.md](references/pr_template.md) (PR description as the
onboarding record), [gotchas.md](references/gotchas.md).

## Step 0: scope

1. **Pick the style.**
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
2. **Pick the reference.** HuggingFace `transformers` is the default; the
   parity test loads HF weights through your `StateDictAdapter`. Use another
   reference (MaxText, upstream torchtitan on CPU) only if no HF port exists,
   and say so in the PR.
3. **Read the architecture and write it down** in the PR description
   (`pr_template.md`): layer pattern (homogeneous / sliding-global cycle /
   dense-then-MoE), norms and eps, RoPE theta and partial factor, QK-norm,
   softcaps, tied embeddings, MoE router and expert shapes, anything that
   mutates module state in forward (breaks scan). Every constant you type by
   hand gets a source line.

## Step 1 (PR 1): correct on SDPA, trains end to end

Deliverables, all in one PR:
- Model code per `model.md`, plain SDPA attention, no Pallas kernels, no
  GraphTrainer. `debugmodel` (and `testmodel`) flavors plus the real sizes in
  the model registry; a `<name>_debugmodel` recipe in `config_registry.py`.
- Registration in `experiments/tpu/train.py` (import for side effects; the
  `model_spec.name` list that routes `tpu_compile_mode`).
- The **HF reference-parity test** (`tests.md`): tiny config covering every
  layer type, HF weights through `StateDictAdapter.from_hf` with no missing
  or unexpected keys, one forward and backward, logits + loss + per-parameter
  gradients compared. This is the test that matters; do not pad the PR with
  per-component or sharding tests.
- State-dict round trip, config/flavor checks, and any mask-semantics test
  the architecture needs (sliding window vs HF's definition).
- End-to-end cases in `experiments/tpu/tests/test_train_distributed.py`
  (FSDP2 and SimpleFSDP, 3 steps of `debugmodel` on `c4_test`) with a context
  length that exercises the architecture (longer than any sliding window).
- A run on a TPU VM per `train.md`: 10+ steps, loss falls, no NaN, bitwise
  repeat with `--debug.seed=42 --debug.deterministic`.

## Step 2 (PRs 2..n): one feature per PR, from the guide

Walk `optimize.md`. Each PR adds one feature behind a config flag (kernels
stay opt-in in the recipe until Step 3), a parity test against the SDPA or
fp32 path on inputs that exercise the feature, a check that the test can fail
(disable the feature's mask or cap and watch the metric move), and an
end-to-end case. Typical order: GraphTrainer wrapper, splash / tokamax
attention, fused cross entropy, MoE kernels, then scan + offload or graph
split for sizes that do not fit.

## Step 3: target size

Add the production recipe(s) and `recipes/<generation>/<model>/<topology>/
README.md` with the launch block, every libtpu flag with a one-line reason,
and a `## Benchmark Results` table. Confirm training at the target size per
`train.md` acceptance criteria, record the numerics class of each enabled
feature (bit-identical / rounding-order / semantic), and only then flip the
kernels on by default.

## Rules

- `pre-commit run --all-files` before every push; CPU tests green locally
  (`TORCH_DEVICE_BACKEND_AUTOLOAD=0`), TPU tests green on a v6e-4.
- No core edits for the model (`.claude/rules/experiments.md`). A copy of a
  `models/common/` block needs a one-line reason in the PR (Gemma 4: common
  `GroupedExperts` hardcodes `silu`).
- No Meta copyright headers on new files; no `dtensor` suffixes in test or
  function names.
- A parity test with no assertion that fails when the feature is off is not
  a test.
- No performance claim from fewer than 10 steps, none from a single run, none
  before the numerics class of the change is known.
- The PR description is the onboarding record. Keep it current on every
  push (`pr_template.md`); nothing else (no `*_handover.md`) goes in the repo.
