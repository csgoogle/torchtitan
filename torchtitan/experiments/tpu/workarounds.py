import logging
from typing import Any, Optional
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torchtitan.tools.logging import logger


_SPLASH_KEYS = {
    "block_q", "block_kv", "block_dkv", "block_kv_compute",
    "block_q_dkv", "block_kv_dkv", "block_kv_dkv_compute",
    "block_q_dq", "block_kv_dq", "use_fused_bwd_kernel",
    "use_vmap_bwd", "use_document_aware_block_skipping",
    "q_layout", "k_layout", "v_layout"
}


def _extract_splash_kwargs(sa_config: Any) -> dict[str, Any]:
  """Safe prefix-mapping helper to map sa_config attributes to standard names."""
  if sa_config is None:
    return {}
  kwargs = {}
  for k in _SPLASH_KEYS:
    val = getattr(sa_config, f"sa_{k}", getattr(sa_config, k, None))
    if val is not None:
      kwargs[k] = val
  return kwargs


def _hf_attn_implementation(model: nn.Module) -> str | None:
  """The HF ``_attn_implementation`` of a wrapped transformers model, else None."""
  hf_config = getattr(getattr(model, "model", None), "config", None)
  return getattr(hf_config, "_attn_implementation", None)


def _splash_attention_modules(model: nn.Module) -> list[nn.Module]:
  """The first-class ``SplashAttention`` inner attentions of ``model``."""
  from torchtitan.experiments.tpu.attention import SplashAttention  # pylint: disable=g-import-not-at-top

  return [m for m in model.modules() if isinstance(m, SplashAttention)]


def use_splash_attention_patch(model: nn.Module, sa_config: Any = None) -> None:
  """Patches splash attention into the model.

  Language models built through ``SplashAttentionConverter`` already carry
  ``SplashAttention`` inner attentions; for those this only pushes the recipe /
  CLI kernel tuning (``sa_*`` block sizes, layouts, ...) into the modules. The
  module-replacement path below remains for models that still build a
  ``ScaledDotProductAttention`` / ``FlexAttention`` (Flux, [redacted] wrappers).
  """
  from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa as standard_splash_sdpa  # pylint: disable=g-import-not-at-top
  import types

  # Extract parameters internally using our safe prefix mapper
  sa_kwargs = _extract_splash_kwargs(sa_config)

  converted = _splash_attention_modules(model)
  if converted:
    windowed = 0
    for module in converted:
      module.configure_kernel(use_tokamax=False, **sa_kwargs)
      if module.sliding_window_size is not None:
        windowed += 1
    logger.info(
        "Splash attention ENABLED: configured %d SplashAttention modules "
        "(causal=%d, sliding-window=%d) with %s",
        len(converted),
        len(converted) - windowed,
        windowed,
        sa_kwargs,
    )
    return

  def _splash_sdpa_redacted(self, *, query, key, value, scale=None, **kwargs):
    """Replaces [redacted]'s native SDPA with splash_sdpa.

    This function replaces ScaledDotProductAttention._sdpa_native_torch with
    `splash_sdpa`. Like TorchTitan's wrapper, [redacted] operates on inputs of shape
    (B, S, H, D) and requires transposing to (B, H, S, D) for splash_sdpa.
    However, [redacted] requires special handling because it handles GQA by tiling
    k/v heads before invoking SDPA. This wrapper aligns with [redacted]'s GQA
    mechanics, transposes inputs for splash_sdpa, and transposes the output
    back to (B, S, H, D).

    For mixed local/global attention models (e.g. REDACTED PT MoE with
    ``attention_layer_pattern=[local_rope, local_rope, local_rope, global_nope]``
    and ``local_attention_window_size=N``), reads
    ``self.sliding_window_size`` per-layer:
      * None  -> full causal (CausalMask) -- same as before, identical to the
                old redacted path where every layer is global causal.
      * int N -> causal sliding-window (LocalMask with window_size=(N, 0)).
                The splash kernel's per-window cache means each unique window
                size compiles its own kernel; mixing windowed and non-windowed
                layers in one model is supported.
    """
    query, key, value = self._maybe_tile_qkv_for_gqa(query, key, value)
    # Transpose from (B, S, H, D) to (B, H, S, D).
    query = query.transpose(-3, -2)
    key = key.transpose(-3, -2)
    value = value.transpose(-3, -2)

    # Prevent torch.compile failures by ensuring tensors are contiguous for the
    # custom Pallas kernel, which doesn't handle arbitrary strides. This is a
    # no-op in eager mode if the tensors are already contiguous.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # Per-layer mask selection. [redacted]'s ScaledDotProductAttention sets
    # `sliding_window_size` from the layer config (None for global, int for
    # local). Models without local layers don't set the attribute at all;
    # default to None => same behavior as before this change.
    local_window_size = getattr(self, "sliding_window_size", None)

    # splash_sdpa expects enable_gqa=False as heads are already tiled.
    out = standard_splash_sdpa(
        query,
        key,
        value,
        scale=scale,
        is_causal=True,
        local_window_size=local_window_size,
        enable_gqa=False,
        **sa_kwargs,
    )
    # Transpose back to match [redacted]'s expected layout
    return out.transpose(-3, -2).contiguous()

  # Patch all attention modules in the model. For [redacted] modules, also note
  # whether each layer has a local sliding-window -- useful for mixed
  # local/global models (REDACTED PT MoE) so the kernel cache shows the expected
  # mix of CausalMask and LocalMask compiled ops.
  from torchtitan.experiments.tpu.attention import SplashAttention  # pylint: disable=g-import-not-at-top

  replacements = []
  _patched = 0
  _windowed = 0
  _causal = 0

  # Identify attention modules to replace
  # (ScaledDotProductAttention & FlexAttention).
  for parent_module in model.modules():
    for child_name, child_module in parent_module.named_children():
      cls_name = type(child_module).__name__

      if "ScaledDotProductAttention" in cls_name:
        if "[redacted]" in type(child_module).__module__:
          # [redacted] patching (in-place method replacement)
          child_module._sdpa_native_torch = types.MethodType(
              _splash_sdpa_redacted, child_module
          )
          _patched += 1
          if getattr(child_module, "sliding_window_size", None) is not None:
            _windowed += 1
          else:
            _causal += 1
        else:
          replacements.append((parent_module, child_name, child_module))
      elif "FlexAttention" in cls_name:
        if "[redacted]" not in type(child_module).__module__:
          replacements.append((parent_module, child_name, child_module))

  # Apply replacements
  for parent_module, child_name, child_module in replacements:
    # SplashAttention needs the window size to pick its mask: LocalMask for a
    # sliding-window layer, CausalMask otherwise. The child is the inner
    # attention being replaced (ScaledDotProductAttention / FlexAttention), the
    # parent the attention module that owns it (e.g. GQAttention). Models set
    # sliding_window_size on either one, so check the child then the parent.
    local_window_size = getattr(child_module, "sliding_window_size", None)
    if local_window_size is None:
      local_window_size = getattr(parent_module, "sliding_window_size", None)

    splash_attn = SplashAttention.Config(
        is_causal=True,  # Default for LMs
        sliding_window_size=local_window_size,
        **sa_kwargs,
    ).build()
    splash_attn.original_module = child_module

    setattr(parent_module, child_name, splash_attn)
    _patched += 1
    if local_window_size is not None:
      _windowed += 1
    else:
      _causal += 1

  if _patched == 0:
    all_classes = sorted(list({type(m).__name__ for _, m in model.named_modules()}))
    raise ValueError(
        "Splash attention patch was requested but NO matching attention"
        " modules (SplashAttention, ScaledDotProductAttention, FlexAttention)"
        f" were found in the model. Found module types: {all_classes}"
    )
  else:
    logger.info(
        "Splash attention ENABLED: replaced %d attention modules "
        "(causal=%d, sliding-window=%d)",
        _patched,
        _causal,
        _windowed,
    )


def use_tokamax_splash_attention_patch(model: nn.Module, sa_config: Any) -> None:
  """Patches zero-overhead Tokamax layout-fused splash attention into the model."""
  from torchtitan.experiments.tpu.kernels.tokamax_splash_attention import TokamaxConfig  # pylint: disable=g-import-not-at-top
  from torchtitan.experiments.tpu.kernels.tokamax_splash_attention import splash_sdpa as tokamax_splash_sdpa  # pylint: disable=g-import-not-at-top
  import types

  # Update core JAX trace compilation configs internally from config registry
  TokamaxConfig.update(
      block_q=getattr(sa_config, "block_q", None),
      block_kv=getattr(sa_config, "block_kv", None),
      block_kv_compute=getattr(sa_config, "block_kv_compute", None),
      block_q_dkv=getattr(sa_config, "block_q_dkv", None),
      block_kv_dkv=getattr(sa_config, "block_kv_dkv", None),
      block_kv_dkv_compute=getattr(sa_config, "block_kv_dkv_compute", None),
      block_q_dq=getattr(sa_config, "block_q_dq", None),
      block_kv_dq=getattr(sa_config, "block_kv_dq", None),
      use_fused_bwd_kernel=getattr(sa_config, "use_fused_bwd_kernel", None),
      q_layout=getattr(sa_config, "q_layout", None),
      k_layout=getattr(sa_config, "k_layout", None),
      v_layout=getattr(sa_config, "v_layout", None),
  )

  def _get_kwargs(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}

  def _filter_splash_kwargs(kwargs_dict):
    return {k: v for k, v in kwargs_dict.items() if k in _SPLASH_KEYS}

  # Extract parameters dynamically using our shared prefix mapper helper!
  sa_kwargs = _extract_splash_kwargs(sa_config)

  converted = _splash_attention_modules(model)
  if converted:
    for module in converted:
      module.configure_kernel(use_tokamax=True, **sa_kwargs)
    logger.info(
        "Tokamax splash attention ENABLED: configured %d SplashAttention modules",
        len(converted),
    )
    return

  def _splash_forward(
      self,
      q,
      k,
      v,
      *,
      scale=None,
      enable_gqa=False,
      is_causal=True,
      **extra_args,
  ):
    if extra_args.get("attention_masks") is not None:
      raise ValueError(
          "TPU SplashAttention Pallas kernel does not support runtime"
          " attention_masks."
      )

    local_window_size = getattr(self, "sliding_window_size", None)

    # Tokamax JAX-side transpositions path (Direct pass-through of contiguous layouts)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    combined_args = {**extra_args, **sa_kwargs}
    forward_kwargs = _filter_splash_kwargs(combined_args)
    out = tokamax_splash_sdpa(
        q,
        k,
        v,
        scale=scale,
        is_causal=is_causal,
        local_window_size=local_window_size,
        enable_gqa=enable_gqa,
        **forward_kwargs,
    )
    return out

  def _splash_sdpa_redacted(self, *, query, key, value, scale=None, **kwargs):
    query, key, value = self._maybe_tile_qkv_for_gqa(query, key, value)
    local_window_size = getattr(self, "sliding_window_size", None)

    # Tokamax JAX-side transpositions path (Direct pass-through of GQA tiled buffers)
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    combined_args = {**kwargs, **sa_kwargs}
    forward_kwargs = _filter_splash_kwargs(combined_args)
    out = tokamax_splash_sdpa(
        query,
        key,
        value,
        scale=scale,
        is_causal=True,
        local_window_size=local_window_size,
        enable_gqa=False,
        **forward_kwargs,
    )
    return out

  _patched = 0
  _windowed = 0
  _causal = 0
  for _, module in model.named_modules():
    cls_name = type(module).__name__
    if "ScaledDotProductAttention" in cls_name:
      if "[redacted]" in type(module).__module__:
        module._sdpa_native_torch = types.MethodType(_splash_sdpa_redacted, module)
      else:
        module.forward = types.MethodType(_splash_forward, module)
      _patched += 1
      if getattr(module, "sliding_window_size", None) is not None:
        _windowed += 1
      else:
        _causal += 1

  if _patched == 0:
    all_classes = sorted(list({type(m).__name__ for _, m in model.named_modules()}))
    raise ValueError(
        "Tokamax splash attention patch was requested but NO matching "
        f"attention modules (ScaledDotProductAttention) were found in the model. Found module types: {all_classes}"
    )
  else:
    logger.info(
        "Tokamax splash attention ENABLED: patched %d attention modules "
        "(causal=%d, sliding-window=%d)",
        _patched, _causal, _windowed,
    )


class PallasLossLinear(nn.Module):
  """Wrapper for model.output to return (x, weight) for Pallas loss.

  Holding the Linear module as a child names the parameter
  ``lm_head.original_linear.weight`` which is not mapped by model state dict
  adapters and breaks loading from checkpoint with a KeyError. We add hooks to
  fix at state-dict layer to strip the ``original_linear`` infix on save and
  restore on load, making the wrapper invisible to checkpointing paths but
  preserving module tree hierarchy.
  """

  _INFIX = "original_linear."

  def __init__(self, original_linear: nn.Linear):
    super().__init__()
    self.original_linear = original_linear
    self._register_state_dict_hook(PallasLossLinear._strip_infix)
    self._register_load_state_dict_pre_hook(self._restore_infix)

  @staticmethod
  def _strip_infix(module, state_dict, prefix, local_metadata):
    """Rename ``<prefix>original_linear.X`` -> ``<prefix>X`` on the way out."""
    for key in [k for k in state_dict if k.startswith(prefix + module._INFIX)]:
      state_dict[prefix + key[len(prefix + module._INFIX):]] = state_dict.pop(key)
    return state_dict

  def _restore_infix(self, state_dict, prefix, *args, **kwargs):
    """Rename ``<prefix>X`` -> ``<prefix>original_linear.X`` on the way in."""
    for key in [
        k
        for k in state_dict
        if k.startswith(prefix) and not k.startswith(prefix + self._INFIX)
    ]:
      state_dict[prefix + self._INFIX + key[len(prefix):]] = state_dict.pop(key)

  @property
  def weight(self) -> torch.Tensor:
    # Dynamically retrieve the weight to avoid caching a stale, unsharded
    # reference before FSDP sharding.
    return self.original_linear.weight

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x, self.weight.t()


def use_output_projection_patch(
    model: nn.Module,
) -> None:
  """Replaces model.output with an identity-like module that returns (x, weight)."""
  if hasattr(model, "output") and isinstance(model.output, nn.Linear):
    model.output = PallasLossLinear(model.output)
    logger.info("Patched model.output to return (x, weight) for Pallas loss")
  elif hasattr(model, "output_transform") and isinstance(model.output_transform, nn.Linear):
    # redacted uses output_transform as output projection layer
    model.output_transform = PallasLossLinear(model.output_transform)
    logger.info(
        "Patched model.output_transform to return (x, weight) for Pallas loss"
    )
  elif hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
    # Original torchtitan models (e.g. Llama) use lm_head for output projection
    model.lm_head = PallasLossLinear(model.lm_head)
    logger.info("Patched model.lm_head to return (x, weight) for Pallas loss")
  else:
    logger.warning("Failed to patch output projection layer. Model output projection layer not found (output, output_transform, or lm_head) or is not nn.Linear.")


def use_gmm_kernel_patch(
    model: nn.Module,
) -> None:
  """Enables the GMM kernel for GroupedExperts modules in the model.

  The pre-merge implementation monkey-patched
  ``torchtitan.models.common.moe._run_experts_grouped_mm`` and
  ``indices_padding_wrapper``. Upstream's ``GroupedExperts`` now runs
  ``torch._grouped_mm`` from its own ``_grouped_mm`` method and neither hook
  exists any more, so the patch has no seam to attach to. The supported TPU
  grouped-matmul path is ``TpuGroupedExperts`` (``grouped_experts.py``,
  selected with ``TpuGroupedExpertsConverter`` +
  ``use_pallas_grouped_mm=True``).
  """
  del model
  raise NotImplementedError(
      "--qwen3.use_gmm_kernel: the module-level grouped_mm patch has no hook"
      " after the upstream merge (GroupedExperts._grouped_mm is a method now)."
      " Use TpuGroupedExpertsConverter with use_pallas_grouped_mm=True instead."
  )


def use_fill_indices_patch(
    model: nn.Module,
) -> None:
  """Enables the fill indices kernel for MoE modules in the model.

  This workload-specific monkey patch replaces the default implementations in
  torchtitan.models.moe.kernels with our TPU-specific JAX implementation.
  """
  # from torchtitan.experiments.tpu.kernels import fill_indices as tpu_fill_indices
  # from torchtitan.models.moe import kernels as moe_kernels

  # # Patch both because generate_permute_indices defaults to use_cpu=True
  # moe_kernels.fill_indices_wrapper = tpu_fill_indices.fill_indices
  # moe_kernels.fill_indices_cpu = tpu_fill_indices.fill_indices
  # logger.info("Patched MoE kernels to use TPU fill_indices")
  # TODO Evaluate if we can remove this since fill_indices no longer exists.
  pass


def use_cpu_safe_histc_patch() -> None:
  """Monkey patches torch.histc to support int dtypes on CPU."""
  original_histc = torch.histc

  def _cpu_safe_histc(input, bins=100, min=0, max=0, *, out=None):
    if input.device.type == "cpu":
      return original_histc(input.float(), bins, min, max, out=out).to(input.dtype)
    return original_histc(input, bins, min, max, out=out)

  torch.histc = _cpu_safe_histc
  logger.info("Monkey-patched torch.histc to support int dtypes on CPU")


def use_rope_tpu_optimization_patch() -> None:
  """No-op since the token-major merge; kept for callers and tests.

  The patch replaced ``torch.gather`` over a 4-D expanded index with a 2-D
  ``rope_cache[positions]`` lookup for ``[B, S]`` positions. Upstream's
  ``_reshape_for_broadcast`` now takes ``[T]`` positions and already does
  ``rope_cache[positions].view(T, 1, width)`` -- the fast path this used to
  install -- so there is nothing left to patch.
  """
  logger.info(
      "RoPE TPU optimization patch is a no-op: upstream's token-major "
      "_reshape_for_broadcast already indexes the cache with 1-D positions"
  )


def use_real_rope_arithmetic_patch() -> None:
  """Monkey patches ComplexRoPE.apply_rotary_emb to avoid a complex view.

  The complex form calls `view_as_complex` on `query`, which requires grad.
  A higher-order op such as `scan` re-traces its body through
  functionalization, which rewrites that to `view_as_complex_copy` -- and only
  the non-copy variant has a derivative:

    NotImplementedError: view_as_complex_copy does not support automatic
    differentiation for outputs with complex dtype

  Writing the multiply out as (a + bi)(c + di) = (ac - bd) + (ad + bc)i is the
  same expression with no differentiable complex view. Same pairing, same
  interleaved output, bit-identical in fp32. The cache is a buffer, never
  requires grad, and stays complex.

  Remove this once `view_as_complex_copy` has an autograd formula upstream.
  """
  try:
    from torchtitan.models.common.rope import ComplexRoPE  # pylint: disable=g-import-not-at-top
  except ImportError:
    return

  def _apply_rotary_emb_real(
      query: torch.Tensor,
      key: torch.Tensor | None,
      rope_cache: torch.Tensor,
      *,
      inverse: bool = False,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # Same contract as upstream's complex form: ``query`` / ``key`` are
    # ``[T, H, K]`` (any leading shape), ``rope_cache`` broadcasts against the
    # ``[..., K // 2]`` complex view, ``inverse`` conjugates the cache.
    if inverse:
      rope_cache = rope_cache.conj()
    cache = torch.view_as_real(rope_cache)
    cos, sin = cache[..., 0], cache[..., 1]

    def rotate(x: torch.Tensor) -> torch.Tensor:
      pairs = x.float().reshape(*x.shape[:-1], -1, 2)
      real, imag = pairs[..., 0], pairs[..., 1]
      rotated = torch.stack(
          [real * cos - imag * sin, real * sin + imag * cos], dim=-1
      )
      return rotated.flatten(-2).type_as(x)

    query_out = rotate(query)
    if key is None:
      return query_out
    return query_out, rotate(key)

  # `staticmethod`, matching what it replaces: the model calls this on an
  # instance but passes query itself as the first argument.
  ComplexRoPE.apply_rotary_emb = staticmethod(_apply_rotary_emb_real)
  logger.info(
      "Monkey-patched ComplexRoPE.apply_rotary_emb with real arithmetic, so "
      "that it can be traced inside a higher-order op"
  )


def apply_patches(model: nn.Module, job_config: Any) -> None:
  """Applies all TPU-specific patches and workarounds to the model."""
  # Enable CPU safe histc workaround.
  use_cpu_safe_histc_patch()

  # scan's autograd backward reverses its stacks with torch.flip, which is the
  # 70B OOM. Only installed when something is actually going to scan.
  if getattr(
      getattr(job_config, "tpu_config", None), "use_scan_layers", False
  ):
    from torchtitan.experiments.tpu.scan_backward_patch import (  # pylint: disable=g-import-not-at-top
        apply_scan_reverse_backward_patch,
    )

    # Fatal, not a warning. The run asked for scan, and the failure the patch
    # prevents is a large-model OOM at some later step -- a much worse way to
    # learn that a torch bump moved the copied source out from under us.
    if not apply_scan_reverse_backward_patch():
      raise RuntimeError(
          "--tpu-config.use-scan-layers needs the scan backward patch and it"
          " did not install; the warning above says which case this is."
      )

  # Splash Attention Patch
  if hasattr(job_config, "splash_attention_kernel"):
    sa_config = job_config.splash_attention_kernel
    if sa_config.use_tokamax_splash_attention_kernel:
      use_tokamax_splash_attention_patch(model, sa_config)
    elif sa_config.use_splash_attention_kernel:
      use_splash_attention_patch(model, sa_config)
    # The HF transformers backend selects splash by attention implementation
    # name rather than by module replacement; its adapter is a function, so
    # the same ``sa_*`` tuning is pushed to it here.
    if _hf_attn_implementation(model) == "splash_tpu":
      from torchtitan.experiments.tpu.transformers_modeling_backend import attention as hf_attention  # pylint: disable=g-import-not-at-top

      hf_attention.configure_kernel(**_extract_splash_kwargs(sa_config))

  # Pallas Loss Patch (Output Projection)
  if (
      hasattr(job_config, "loss_kernel")
      and job_config.loss_kernel.use_loss_kernel
  ):
    use_output_projection_patch(model)

  # Qwen3 GMM MoE Patches
  if hasattr(job_config, "qwen3") and job_config.qwen3.use_gmm_kernel:
    use_gmm_kernel_patch(model)
  if hasattr(job_config, "qwen3") and job_config.qwen3.use_fill_indices_kernel:
    use_fill_indices_patch(model)

  # Apply RoPE TPU Optimization Patch
  use_rope_tpu_optimization_patch()
  use_real_rope_arithmetic_patch()


def raise_dynamo_recompile_limits(model_config) -> None:
  """Size Dynamo's recompile limits for per-layer compile under SimpleFSDP.

  Parametrization / AC wrapping gives every decoder layer its own module
  class, so Dynamo recompiles the shared checkpoint-wrapper ``forward`` once
  per layer (a type guard miss, not a real recompile). torch's default
  recompile_limit of 8 exactly fits the 8-layer debug models and fails hard
  (fullgraph) at 52/80 layers. Called by both TPU trainers before the first
  compiled forward.
  """
  import torch._dynamo.config as dynamo_config  # pylint: disable=g-import-not-at-top

  n_layers = len(getattr(model_config, "layers", ()) or ())
  wanted = max(64, 2 * n_layers + 16)
  dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, wanted)
  dynamo_config.accumulated_recompile_limit = max(
      dynamo_config.accumulated_recompile_limit, 8 * wanted
  )
