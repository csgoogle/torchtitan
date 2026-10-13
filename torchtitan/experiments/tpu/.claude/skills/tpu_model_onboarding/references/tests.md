# Tests

What to write, what not to write, and the one template that matters.

## Mandatory in PR 1

```
Test                                  | File                                           | Template
--------------------------------------|------------------------------------------------|-----------------------------------------------
HF reference parity (single chip)     | tpu/<path>/tests/test_<name>.py                | PR #491 TestGemma4HFReferenceParity
State-dict round trip                 | same file                                      | PR #491 TestGemma4StateDictAdapter
Flavor / config checks                | same file                                      | PR #491 TestGemma4Flavors, muse test_muse_glimmer_config
Architecture-specific mask semantics  | same file, only if the arch has one            | PR #491 TestGemma4SdpaSlidingWindow
End-to-end training cases             | tpu/tests/test_train_distributed.py            | muse_glimmer_fsdp / muse_glimmer_simplefsdp
                                      | tpu/tests/test_train_minimal_distributed.py    | qwen3_fsdp / qwen3_simplefsdp
FLOP accounting (if a Config subclass)| tpu/tests/test_mfu_flops.py                    | MaxTextQwen3FlopsTest
```

Do not add: per-component CPU-vs-device ladders (embedding, attention, norm,
FFN one by one), `test_dtensor_parallelize_distributed.py`,
`test_simple_fsdp_sharding.py`. Those were written for the first models when
DTensor itself was unproven; they are no longer required (PR #491 review). Add a
component test only when it guards a specific thing the parity test cannot see.

## The HF reference-parity test

One test class, `BaseAcceleratorDeviceTest` from `tpu/base_device_test.py`, so
CI runs it on CPU and on a single TPU chip. Shape, from PR #491:

1. **Reference**: build `transformers.<Model>ForCausalLM` from a tiny
   `<Model>TextConfig` in fp32, `.eval()`. Tiny means hidden 128, 2 layers that
   cover every layer type (one sliding + one global; one dense + one MoE), 8
   heads, head dim 16-32, vocab 2048, seq 16, batch 2. Skip cleanly if the
   installed `transformers` lacks the class (`self.skipTest(...)`).
2. **Perturb unit-initialized parameters**: norms, learned scales,
   `layer_scalar`-style multipliers get `1.0 + 0.1 * randn`. With the defaults
   a missing scale multiplies by 1 and the test cannot see it.
3. **Geometry exercises the architecture**: sliding window smaller than the
   sequence so the window masks; MoE with more experts than top-k; tied
   embeddings if the real model ties them.
4. **Weights through your adapter**: `adapter.from_hf(hf_model.state_dict())`
   into `tt_model.load_state_dict(..., strict=False)` and assert both `missing`
   and `unexpected` are empty. This tests the adapter and the model together.
5. **One forward and backward on the same tokens**, model on
   `self.accelerator_device`, reference on CPU. Use `set_unflatten_seq_len` and
   the `[T]` token layout the trainer uses, with explicit `positions`.
6. **Compare three things** with `test_utils.check_equivalence`:
   logits, loss, and every parameter gradient. Map HF gradients through
   `adapter.from_hf` too; drop the duplicate tied-head gradient; assert the key
   sets are equal before comparing values.
7. **Parameterize** over flavors and converters: dense, MoE, MoE with
   `TpuGroupedExpertsConverter` (Pallas off so it runs on CPU).

Tolerances that passed on v6e and v7x for Gemma 4 (fp32 reference, bf16-free
model, TPU matmul precision default):

```
Quantity   | dense            | MoE
-----------|------------------|-----------------
logits     | atol 5e-2 rtol 1e-2 | atol 1e-1 rtol 2e-2
loss       | atol 5e-3 rtol 1e-3 | atol 5e-3 rtol 2e-3
gradients  | atol 8e-2 rtol 2e-2 | atol 8e-2 rtol 2e-2
```

Start there; tighten if the model passes with margin, and record the measured
error in the PR description next to the bound.

## Feature parity tests (Step 2 PRs)

Each kernel or optimization PR adds a test against the path it replaces, on
the same packed batch, comparing outputs and gradients:

- splash / tokamax attention vs SDPA (PR #530): context 4x the sliding
  window, three documents packed in one row with boundaries inside the window,
  both layer types; relative L2 <= 1e-2 forward, <= 2.5e-2 gradients, measured
  margin in the PR.
- fused cross entropy vs fp32 `F.cross_entropy` (PR #532): bit-identical to the
  stock tokamax kernel when the model-specific option (soft-cap) is off.
- scan vs unrolled on a 4-layer proxy, 5 steps, loss and grad_norm within bf16
  noise.
- GraphTrainer vs per-layer compile: 3-step loss agreement on `debugmodel`.

**The test must be able to fail.** Add the control: remove the document mask
and assert the SDPA output moves by more than 10%; turn the cap off and assert
capped and uncapped differ. A kernel that silently ignores the feature then
fails the test instead of passing it.

## End-to-end cases

`tpu/tests/test_train_distributed.py` runs `--module/--config` in a subprocess
for 3 steps on `c4_test`. Add one case per parallelism the recipe supports
(FSDP2 via `--parallelism.no-use_simple_fsdp`, SimpleFSDP, TP if the model
allows it) and later one per feature flag. Set `max_context_length` so the
architecture does its work (muse: 256 > the 128-token window; 512 with
splash so one block spans the window) and keep the token budget consistent
(`num_tokens_per_microbatch_per_dp_rank`). Use `skip_devices` for TPU-only
kernels. Mirror the FSDP cases in `test_train_minimal_distributed.py`.

## Multi-step CPU-vs-device parity

If you keep a multi-step CPU-vs-device test, bound step 0 tightly and the
post-optimizer steps loosely. After two Adam steps at lr 1e-2 the TPU-vs-CPU
loss gap on the muse debugmodel ranged 0.004 to 0.37 across seeds (PR #534):
bf16 SDPA error is amplified by the update, not a bug. Make each check
independent so a failed loss check does not skip the gradient checks.

## CI discovery rules (`.ci/discover_tpu_tests.py`)

- Any `test_*.py` is collected. Hardware comes from the base class:
  `BaseAcceleratorDeviceTest` runs on CPU and single TPU; distributed tests
  subclass `base_distributed_device_test` and the file name must end in
  `_distributed.py`.
- Files with an `absltest.main()` block run as `python -m <module> --device=...`
  (900 s timeout); files without one run under pytest.
- CPU runs set `TORCH_DEVICE_BACKEND_AUTOLOAD=0`; a test that imports
  `torch_tpu` unconditionally breaks the CPU shard.
- Use `@parameterized.named_parameters`; `testcase_name` is what CI reports.

## Reference checks outside CI

- `scripts/checkpoint_conversion/numerical_tests_*.py` compares logits by KL
  against an HF model on real weights; run it on the full checkpoint before
  Step 3 and quote the KL in the PR.
- `/numerics_debugging` when a parity test fails and the first diverging op
  is not obvious.
