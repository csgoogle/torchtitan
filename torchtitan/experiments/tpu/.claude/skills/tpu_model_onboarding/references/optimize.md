# The optimization ladder

Source of truth for each lever: `torchtitan/experiments/tpu/torch-tpu-optimization-guide.md`
(section numbers below refer to it). This file says in what order to climb,
what each rung must contain, and when to stop.

Rung 1 is a PR of its own (it is the oracle everything else is compared
against, and it has different reviewers than kernels). Rungs 2 and up are one
commit each on one branch; the stack ships as one PR or as stacked PRs at the
human's discretion when it is done. An agent does not open a PR per rung: it
cannot wait for a review between rungs, and five PRs against the same base
conflict with each other.

## Per-rung contract

Every rung (one commit) has:

1. One feature behind a config flag; the recipe leaves it off until Step 3.
2. A parity test against the path it replaces, on a batch that exercises the
   feature (`tests.md`), plus the control that proves the test can fail.
3. One end-to-end case in `test_train_distributed.py` with the flag on.
4. In the commit message, and mirrored into the `## Ladder` table of the
   onboarding record: the numerics class (`train.md`), step time / MFU before
   and after on the same recipe and chip count (>= 10 steps, measured past
   warmup), and the HLO evidence that the feature fired (custom_call present,
   segment ids reaching the kernel).
5. Guide-ordered: do not skip ahead to kernels while the model is still
   dispatch-bound on the compile path.
6. Rung k's "Done when" (the four items above, passing on the Step 0 target)
   before rung k+1 starts. That is the part of "merge before the next one"
   that can be honoured inside one session.

## Order

```
Rung | Lever                         | Guide | Parity test                          | Notes
-----|-------------------------------|-------|--------------------------------------|------------------------------------------
1    | per-block torch.compile +     | 4, 5  | reference parity; e2e cases          | PR 1 of the onboarding. dynamic=False if
     | SimpleFSDP + mp policy        |       |                                      | layer types share a block class
2    | GraphTrainer wrapper          | 8     | 3-step loss vs per-block compile     | tpu/graph_trainer/<name>.py + train.py
     |                               |       | (test_tpu_graph_trainer_distributed) | route; TP/PP rejected if unsupported
3    | splash / tokamax attention    | 4     | kernel vs SDPA on packed batch,      | needs positions -> segment ids at the
     |                               |       | both layer types, grads too          | call site; off-by-one window check
4    | fused cross entropy           | 4     | vs fp32 F.cross_entropy; bit-ident.  | plain Linear head only unless you extend
     |                               |       | to stock kernel when option off      | the kernel (soft-cap: PR #532)
5    | MoE kernels (if MoE)          | 10    | grouped mm vs torch._grouped_mm incl.| TpuGroupedExpertsConverter + dispatcher
     |                               |       | empty groups; paired vs unpaired     | converter; EP stays 1
6    | batch / context to saturation | 3     | none (recipe change)                 | re-measure after 3 and 4 freed memory
7    | scan + offload, or graph split| 9     | scan vs unrolled 4-layer proxy, 5    | only when 2 OOMs at compile or the step
     |                               |       | steps, loss + grad_norm              | no longer fits; pick one, not both
8    | libtpu flag bundle            | 9     | bit-identical step-1 loss            | tpu_flags.py bundles with a reason each
9    | tile sweeps (splash, CE, MoE) | 4, 10 | rounding-order; step-1 oracle +      | per generation; v6e tiles do not carry
     |                               |       | trajectory check for the frontier    | to v7x
```

Levers that have been measured as neutral or negative on this lane and need a
fresh measurement before they go in a recipe: op-level selective AC, splash
block 1024 vs 2048 at 16k, MaxText's scheduler flag set wholesale,
DEFER_AND_FUSE on top of per-block compile, unrolled GraphTrainer at 30B+,
fused AdamW when the eager optimizer is not on the critical path.

## Structural checks per lever

- **Attention kernels**: the model's attention must forward `positions` to the
  inner attention; splash builds its document mask from them. A packed batch
  with no segment ids trains fine and attends across documents (muse, hf_backend).
  Confirm in HLO that the custom_call operands include the segment ids.
- **Fused CE**: `workarounds.use_output_projection_patch` rejects non-plain
  heads. If the head scales or caps logits, extend the kernel (PR #532 is the
  pattern) rather than silently dropping the cap.
- **MoE**: `use_pallas_paired_grouped_mm` requires `use_pallas_grouped_mm`,
  full N tile in the weight-gradient tiling, and EP = 1. The routed row count
  must be a multiple of `tile_m`. Weight-gradient K tiles larger than the axis
  are not clamped and exhaust VMEM.
- **Scan**: `check_scan_was_applied` runs at startup; `use_scan_layers` without
  GraphTrainer is rejected. MoE counters and routing metrics off.
- **Graph split**: `graph_trainer_fsdp_schedule="none"`, no `optimizer_in_graph`.
- **Optimizer path**: exactly one of eager, `compile_optimizer` (separate
  program) or `optimizer_in_graph` (GraphTrainer only, AdamW only, not with
  `compile_grad_norm`, not with `use_scan_layers`); the config rejects the
  invalid pairs.
- **Flags**: every flag in the recipe carries a comment with its reason and
  the measured effect; flags left on the command line get forgotten.

## Recipe README (Step 3)

`tpu/recipes/<generation>/<model>/<topology>/README.md`, following
`recipes/ironwood/qwen3-30b-a3b/tpu7x-4x4x4/README.md`:

- one paragraph: what the recipe is, link to the TPU README for cluster setup;
- the export block and launch command, flags commented;
- `## Benchmark Results`: run date, chips, sequence, batch, step time, TPS,
  TFLOPS, MFU, peak HBM, and the image / libtpu / torch_tpu versions. Numbers
  from `--tpu_config.enable_mfu_benchmark` over >= 10 post-warmup steps.

## Stop rule

The ladder is done for a target size when: the profile shows no exposed
dispatch gaps or exposed collectives that a listed lever addresses, every
enabled feature has its numerics class recorded, the trajectory check passed,
and the README benchmark table is filled. "Out of ideas" is not a stop
condition; an unmeasured lever on the list is.
