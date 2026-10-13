# PR description as the onboarding record

The merged Muse Glimmer PRs (#521, #529, #530, #532) are the standard: the
description states what changed and why, which tests prove it, the measured
bounds next to the asserted ones, and what is deliberately left for later.
No handover file is committed; the description is updated on every push and
the wiki (if the project keeps one) links to the PR.

## PR 1 template

```
<Model> on the TorchTitan TPU lane (<style A: wraps torchtitan/models/<name> |
style B: ported from <HF id / reference>, under experiments/tpu/examples/<name>>).

Architecture (source: <HF config.json / modeling_<x>.py / paper>):
- layers: <N>, pattern <homogeneous | k sliding + 1 global | ...>
- attention: <GQA heads/kv>, head_dim <d> (<d_global> on global layers), QK-norm <y/n>,
  sliding window <W>, softcap <attn/final>
- norms: <RMSNorm, eps 1e-6, (1+w) | w>
- RoPE: theta <t>, partial factor <f>, <complex | real>
- FFN / MoE: <gated act>, experts <E> top-k <k>, router <details>, balancer <details>
- embeddings: tied <y/n>, scaled by sqrt(dim) <y/n>
- forward side effects (scan blockers): <none | counters X, metrics Y behind flags>
Every constant above is taken from <file:line>.

Code:
- <file>: <one line each>
- Deviations from torchtitan/models/common and why: <...>
- Registration: train.py import + model_spec.name route; train_minimal.py

Tests:
- test_<name>.py::Test<Model>HFReferenceParity: HF <class> (CPU, fp32) vs
  <Model>Model (<device>), tiny config <dims>, <variants>; logits / loss / all
  grads. Bounds <...>; measured <...> on v6e, <...> on v7x.
- state-dict round trip; flavor checks; <mask semantics test>
- test_train_distributed: <name>_fsdp, <name>_simplefsdp (3 steps, context <L>
  because <reason>)

Runs (TPU VM <type>, image <tag>, torch_tpu <sha>, libtpu <ver>):
- <name>_debugmodel, 10 steps: loss <l0> -> <l10>, grad_norm <...>
- determinism: two runs --debug.seed=42 --debug.deterministic bit-identical
- step-1 oracle: loss <x>, grad_norm <y>
- eager / per-block / (GraphTrainer): <losses agree to ...>

Not in this PR (follow-ups): GraphTrainer wrapper, splash/tokamax, fused CE,
<MoE kernels>, <size> recipe.
```

## Feature PR template (Step 2)

```
<Feature> for <Model> on TPU.

What: <one paragraph: what it replaces, how it is enabled, default off>.
Why: <profile evidence: bucket, exposed time>.

Numerics class: <bit-identical | rounding-order | semantic (then: why this is
a separate recipe)>. Step-1 oracle: <same | delta>.

Tests:
- <parity test>: <inputs geometry>; bounds <...>, measured <...>
- control: <what was disabled and how much the metric moved>
- test_train_distributed: <case name> (<flags>)

Measurement (<recipe>, <chips>, >= 10 steps past warmup):
  before <step time / TPS / MFU / HBM>, after <...>.
HLO evidence: <custom_call name present N times; segment ids in operands>.
```

## Step 3 template

```
<Model> <size> recipe on <generation> <topology>.

Recipe: <fn name>; flags with reasons in the README.
Features on by default: <list, each with numerics class and the PR that added it>.
Trajectory check: <reference>, 200 steps, post-warmup median |diff| <x> nats,
offset <y> (<accepted because ... | open>).
Benchmark Results table: in recipes/<...>/README.md.
Known gaps: <EP, TP, PP, scan for heterogeneous layers, ...>.
```
