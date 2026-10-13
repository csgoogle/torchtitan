# Model code: where it goes and what it must satisfy

Paths are relative to the repo root. `tpu/` means `torchtitan/experiments/tpu/`.

## Style A: wrap an upstream model

Template: `tpu/muse_glimmer/` (PR #521).

```
tpu/<name>/
  __init__.py            model_registry(flavor, *, seq_len=None, converters=None) -> ModelSpec
  config_registry.py     <name>_debugmodel() -> TPUTrainerConfig  (+ real sizes later)
  infra/parallelize.py   parallelize_<name>(...)  (upstream parallelize with skip_dp, then SimpleFSDP)
  attention.py           only if the TPU attention path needs a model-specific inner attention
  tests/
```

`model_registry` calls the upstream registry (`torchtitan.models.<name>.model_registry`)
with `attn_backend="sdpa"` and the caller's `converters`, then swaps in TPU pieces:
replaces `parallelize_fn` with yours, sets `pipelining_fn=None` if PP is unsupported,
and applies any attention swap. Keep a `testmodel` flavor (2 layers, dim 128, one of
each layer type) for unit tests next to upstream's `debugmodel`.

`infra/parallelize.py` follows `tpu/llama3/infra/parallelize.py`: when
`parallelism.use_simple_fsdp`, call upstream's parallelize with `skip_dp=True`
(TP, AC, compile) and then `simple_fsdp_data_parallel(model, dp_mesh, mode=
"fully_shard"|"hybrid_shard", mp_policy=...)` from
`torchtitan.experiments.graph_trainer.simple_fsdp`; otherwise delegate fully.
`compile_mode in ("fullgraph", "aot_fx_trace")` disables per-block compile.

## Style B: port a model that is not upstream

Template: PR #491, `tpu/examples/gemma4/`. Follow `torchtitan/models/README.md`:

```
tpu/examples/<name>/
  __init__.py            flavors dict {str: Model.Config} + model_registry(flavor, ...) -> ModelSpec
  model.py               Model(Decoder) + nested Model.Config; get_nparams_and_flops()
  state_dict_adapter.py  BaseStateDictAdapter subclass: to_hf / from_hf (HF-named protocol; see
                         "The weight mapping" below for other sources), ignore non-text towers
  sharding.py            set_<name>_sharding_config(...) using decoder_sharding helpers
  infra/parallelize.py   model.parallelize(parallel_dims) -> MoE fsdp -> AC -> compile -> FSDP
  config_registry.py
  tests/
```

- Build from `torchtitan/models/common/`: `Decoder`, `TransformerBlock`,
  `AttentionBase` / `SDPAInnerAttention`, `FeedForward`, `MoE` / `GroupedExperts` /
  `TokenChoiceRouter`, `RotaryEmbedding`, `RMSNorm`. Subclass for a deviation;
  do not copy. If a common block hardcodes something (`GroupedExperts` hardcodes
  `silu`), put the change in the TPU subclass via `derive` and say why in the PR.
- Hyperparameters live on `Model.Config`; init via `param_init` on each
  sub-config; RoPE and MoE buffers in `_init_self_buffers()`.
- Keep model files architecture-only: no training infrastructure, no device
  branches. TPU-only behaviour goes through converters, overrides or
  `workarounds.py`, never `if tpu:` in the model.
- Control flow stays in `forward`.
- Weight tying is a config field, handled in one place.
- `state_dict_adapter.py` keeps the canonical checkpoint keys; if kernels want
  another layout, translate in load/save hooks (see `TpuGroupedExperts`). Note
  those hooks cover the model state dict only; optimizer state is keyed by
  live parameter names and shapes, so a layout switch is not
  checkpoint-compatible for a resumed optimizer.

## The weight mapping (any reference)

`BaseStateDictAdapter` (`torchtitan/protocols/state_dict_adapter.py`) is
HF-named (`to_hf`, `from_hf`, `get_hf_storage_reader`) and the conversion
scripts under `scripts/checkpoint_conversion/` are HF-only, but what the
adapter does is a key-and-layout translation, and that is the same job for
every source. Implement it under the same protocol (the names are legacy) or
as a one-off converter script, and have the parity test count keys both ways.
What it has to handle, by source:

- **HF transformers**: key renames, fused vs separate QKV, tied `lm_head`,
  per-model RoPE conventions (`_validate_hf_rope_config`), non-text towers
  dropped.
- **Megatron-LM / NeMo**: TP shard concatenation order per tensor (column vs
  row parallel), PP stage offsets into layer indices, fused QKV interleave
  (per-head-grouped vs contiguous), fused gate/up, norm naming and bias
  presence, the master-weight dtype. Gather to one unsharded state dict on
  the reference side first (`tests.md`, golden-tensor shape); the adapter
  never deals with sharding.
- **MaxText / JAX**: pytree paths to module names, transposed kernel layouts
  (`[in, out]` vs `[out, in]`), scanned-layer stacks split per layer.
- **custom code**: write the table down in the PR before writing the adapter.

No Megatron or MaxText adapter exists in this tree yet; the first one is the
template for the next.

## Config and registration

- Recipe functions return `TPUTrainerConfig`; derive variants from a parent
  recipe (`cfg = <name>_debugmodel(); cfg.x = ...`) rather than restating it.
- Token budgets: `training.num_tokens_per_train_step` must equal
  ranks x `num_tokens_per_microbatch_per_dp_rank` or config validation fails.
- Datasets: `c4_test_dataloader()` for debug recipes; the production dataset
  only in production recipes.
- Register in `tpu/train.py`: import the package for side effects
  (`import torchtitan.experiments.tpu.<name>`), add `model_spec.name` to the
  tuple that pops `tpu_compile_mode` under GraphTrainer, and, when the
  GraphTrainer wrapper lands, add the `elif model_name == "<name>"` route to its
  registry. `train_minimal.py` needs the import too.
- FLOP accounting: `Model.Config.get_nparams_and_flops()` must count what the
  model does (active experts for MoE, GQA head dims, tied embeddings). For
  MaxText-comparable numbers subclass the config like
  `tpu/mfu_flops.py:MaxTextQwen3Config` and cover it in
  `tpu/tests/test_mfu_flops.py`. Without this the recipe reports wrong MFU.

## TPU extension points (use these, not model edits)

- `ModelConfigConverter` subclasses passed as `converters=[...]` to the
  registry: `TpuPackedQKVConverter` (`tpu/qkv.py`), `TpuTokenDispatcherConverter`
  (`tpu/token_dispatcher.py`), `TpuGroupedExpertsConverter`
  (`tpu/grouped_experts.py`). They rewrite the config tree before the model is
  built; `derive(config, TpuCls.Config)` copies fields into the TPU subclass.
- `override.imports`: import-path `@override` factories under `tpu/overrides/`
  (`real_rope`, `fused_loss`, `bf16_rope`, `discarded_token_counter`) that
  replace a module class at construction. Use for a model-specific swap that
  must happen before FSDP wraps the module.
- `workarounds.apply_patches(model, job_config)`: post-build patches keyed on
  config flags (splash / tokamax attention, fused-loss output projection).
  Check `_check_lm_head_is_plain_projection` style guards if the model's head
  is not a plain `Linear` (soft-capped heads needed PR #532).
- `bld_layout.py` is on by default for dense Decoder models: `[T, D]` inputs are
  viewed as `[B, L, D]` inside blocks. If the model's forward is not stock
  `Decoder.forward`, check `tpu_model_class` raises rather than silently
  skipping (`tpu/layer_loop.py`).

## Scan and GraphTrainer readiness (decide in Step 0, enforce in Step 1)

Even if scan ships later, write the model so it can be scanned:

- Stock `Decoder.forward` loop over `model.layers`; `check_layers_are_stackable`
  must pass (`tpu/graph_trainer/layer_stack.py`).
- The layer forward must not write module attributes or captured buffers, and
  must not mutate its inputs in place. Grep layer code for `self\.\w+ = `,
  `.add_(`, `.copy_(`. MoE load-balancing counters and routing metrics are the
  usual offenders; keep them behind config so a scan recipe can turn them off.
- No complex-number ops in the layer (`view_as_complex` does not trace inside
  `scan`); use real-arithmetic RoPE (`tpu/overrides/real_rope.py`).
- Heterogeneous layers (sliding / global with different head dims) share one
  block class: compile per block with `dynamic=False`, or Dynamo marks the
  differing dims symbolic on the second layer type and the TPU backend rejects
  the graph (Gemma 4, PR #491 commit 8f6e12ffa). Scan over such layers is not
  supported; graph split is the large-size path for them.
- Anything keyed off per-layer parameter names (allreduce-slice patterns,
  SAC name matchers) stops matching under scan; do not depend on it.
