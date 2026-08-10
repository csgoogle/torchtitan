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
    "use_vmap_bwd", "q_layout", "k_layout", "v_layout"
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


def use_splash_attention_patch(model: nn.Module, sa_config: Any = None) -> None:
  """Patches splash attention into the model."""
  from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa as standard_splash_sdpa  # pylint: disable=g-import-not-at-top
  import types

  # Extract parameters internally using our safe prefix mapper
  sa_kwargs = _extract_splash_kwargs(sa_config)

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
    """Replace ScaledDotProductAttention.forward with splash_sdpa.

    Reads ``sliding_window_size`` from the patched module if set; otherwise
    behaves as before (causal or full mask depending on ``is_causal``). Most
    torchtitan wrappers don't expose this attribute → defaults to None.
    """
    if extra_args.get("attention_masks") is not None:
      raise ValueError(
          "TPU SplashAttention Pallas kernel does not support runtime"
          " attention_masks."
      )

    # TorchTitan passes tensors as (B, S, H, D). However, the splash_sdpa Pallas kernel
    # expects (B, H, S, D). We transpose them here to match the kernel's expected layout.
    # We also prevent torch.compile failures by ensuring tensors are contiguous for the
    # custom Pallas kernel, which doesn't handle arbitrary strides. This is a
    # no-op in eager mode if the tensors are already contiguous.
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    local_window_size = getattr(self, "sliding_window_size", None)


    out = standard_splash_sdpa(
        q,
        k,
        v,
        scale=scale,
        is_causal=is_causal,
        local_window_size=local_window_size,
        enable_gqa=enable_gqa,
        **sa_kwargs,
    )
    # Transpose back to (B, S, H, D) so the rest of the TorchTitan model receives its expected format
    return out.transpose(1, 2)

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
      * None  → full causal (CausalMask) — same as before, identical to the
                old redacted path where every layer is global causal.
      * int N → causal sliding-window (LocalMask with window_size=(N, 0)).
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
    # default to None ⇒ same behavior as before this change.
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
  # whether each layer has a local sliding-window — useful for mixed
  # local/global models (REDACTED PT MoE) so the kernel cache shows the expected
  # mix of CausalMask and LocalMask compiled ops.
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
        "Splash attention patch was requested but NO matching attention "
        f"modules (ScaledDotProductAttention) were found in the model. Found module types: {all_classes}"
    )
  else:
    logger.info(
        "Splash attention ENABLED: patched %d attention modules "
        "(causal=%d, sliding-window=%d)",
        _patched, _causal, _windowed,
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
  """Wrapper for model.output to return (x, weight) for Pallas loss."""

  def __init__(self, original_linear: nn.Linear):
    super().__init__()
    self.original_linear = original_linear

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

  This workload-specific monkey patch directly intercepts and overrides the
  `_run_experts_grouped_mm` runner inside Torchtitan's MoE module. This allows
  us to synchronously route execution to our locally compiled JAX/Pallas
  kernel (`gmm.grouped_matrix_multiply`).
  """
  logger.info("Applying GMM Kernel patch to Torchtitan's MoE module.")
  from torchtitan.experiments.tpu.kernels import gmm  # pylint: disable=g-import-not-at-top
  import torchtitan.models.common.moe as torchtitan_moe  # pylint: disable=g-import-not-at-top

  # Explicitly intercept and rewrite the Torchtitan inner executor
  # for `_run_experts_grouped_mm`. This is significantly more robust than
  # hijacking PyTorch's ATen dispatch because it guarantees we call the custom
  # Pallas kernel synchronously without torch._grouped_mm interfering.
  def _custom_run_experts_grouped_mm(
      w1: torch.Tensor,
      w2: torch.Tensor,
      w3: torch.Tensor,
      x: torch.Tensor,
      num_tokens_per_expert: torch.Tensor,
  ) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    h = F.silu(
        gmm.grouped_matrix_multiply(
            x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets
        )
    )
    h = h * gmm.grouped_matrix_multiply(
        x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )
    out = gmm.grouped_matrix_multiply(
        h, w2.bfloat16().transpose(-2, -1), offs=offsets
    ).type_as(x)
    return out

  # Bypass the Triton-based padding dependency that Torchtitan injects for
  # non-EP setups. Pallas naturally handles the unpadded jagged offsets.
  def bypassed_padding_wrapper(fn):
    logger.info(
        "indices_padding_wrapper bypassed since Pallas kernel handles padding."
    )
    return fn

  torchtitan_moe.indices_padding_wrapper = bypassed_padding_wrapper  # pyrefly: ignore[missing-attribute]
  torchtitan_moe._run_experts_grouped_mm = _custom_run_experts_grouped_mm


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
  """Monkey patches torchtitan.models.common.rope._reshape_for_broadcast with a TPU-optimized path.

  Why upstream torch.gather is slow on TPU:
  1. Index Tensor Bloat: Upstream expands the 2D `positions` tensor into a
     massive 4D index tensor, multiplying its memory size by `cache_width`
     (e.g., 33.5 MB vs 262 KB).
  2. Forced Scalar Memory Lookups: The mathematical API of `torch.gather`
     requires an independent scalar index for every single element in the
     output tensor. Because XLA cannot assume the indices are identical at
     runtime, it must generate machine code that computes a dynamic memory
     address and fetches exactly 1 scalar number at a time (millions of
     scattered scalar fetches), destroying memory bandwidth.

  Why direct 2D advanced indexing is fast on TPU:
  1. 128x Smaller Index: Uses the 2D `positions` tensor directly without
     expanding it.
  2. Fast Contiguous Vector Loads: By writing `rope_cache[positions]`, we
     semantically request entire rows of the matrix. Because XLA knows the
     elements of a row are stored contiguously in physical memory
     (Row-Major order), it compiles a single Vector Load instruction
     (HLO Gather with slice_sizes=[1, cache_width]). The TPU calculates just
     one dynamic address per row and scoops up the entire vector block in a
     single fast hardware cycle.
  3. O(1) Metadata Unsqueeze: We reshape the 3D result to 4D via
     `.unsqueeze(2)`, which only modifies shape metadata and does zero
     memory work.

  Equivalence: Both methods mathematically compute
  `rope_cache[positions[b, s], k]`, providing exact bitwise parity while
  eliminating the TPU scalar lookup bottleneck.
  """
  try:
    import torchtitan.models.common.rope as rope
    import spmd_types as spmd
    from torch.fx.experimental.symbolic_shapes import guard_or_false  # pylint: disable=g-import-not-at-top,g-importing-member
  except ImportError:
    return

  @spmd.local_map(out_types={"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.R})
  def _reshape_for_broadcast_patched(
      rope_cache: torch.Tensor,
      query_shape: torch.Size | tuple[int, ...],
      positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    ndim = len(query_shape)
    assert ndim > 1
    bsz, seqlen = query_shape[:2]
    cache_width = rope_cache.shape[-1]

    # Case 1: No positions (sequential). Unchanged from original.
    if positions is None:
      rope_cache = rope_cache[0:seqlen]
      shape = [
          d if i == 1 else cache_width if i == ndim - 1 else 1
          for i, d in enumerate(query_shape)
      ]
      return rope_cache.view(*shape)

    # Case 2: Singleton positions (bsz=1). Unchanged from original.
    if guard_or_false(positions.size(0) == 1):
      rope_cache = rope_cache[positions.squeeze(0)]
      shape = [
          d if i == 1 else cache_width if i == ndim - 1 else 1
          for i, d in enumerate(query_shape)
      ]
      return rope_cache.view(*shape)
    # Case 3: Per-batch positions (bsz > 1).
    # Replaces slow 4D torch.gather (33.5 MB index, scalar HLO lookups)
    # with fast 2D indexing rope_cache[positions].unsqueeze(2) (262 KB index,
    # native TPU 2D embedding lookup copying contiguous vector rows).
    else:
      # Index 2D cache [max_seqlen, cache_width] with 2D positions [bsz, seqlen]:
      # yields [bsz, seqlen, cache_width]. unsqueeze(2) inserts head
      # dimension, returning shape [bsz, seqlen, 1, cache_width].
      return rope_cache[positions].unsqueeze(2)

  rope._reshape_for_broadcast = _reshape_for_broadcast_patched
  logger.info(
      "Monkey-patched "
      "torchtitan.models.common.rope._reshape_for_broadcast "
      "with TPU optimized path"
  )


def apply_patches(model: nn.Module, job_config: Any) -> None:
  """Applies all TPU-specific patches and workarounds to the model."""
  # Enable CPU safe histc workaround.
  use_cpu_safe_histc_patch()

  # Splash Attention Patch
  if hasattr(job_config, "splash_attention_kernel"):
    sa_config = job_config.splash_attention_kernel
    if sa_config.use_tokamax_splash_attention_kernel:
      use_tokamax_splash_attention_patch(model, sa_config)
    elif sa_config.use_splash_attention_kernel:
      use_splash_attention_patch(model, sa_config)

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
