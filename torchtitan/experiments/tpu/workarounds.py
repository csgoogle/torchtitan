import logging
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode
from torchtitan.tools.logging import logger
from torch import nn

def attention_sdpa_forward_dtensor_workaround(sdpa_backends, q, k, v, scale=None, is_causal=True, enable_gqa=False):
    """
    Unwraps DTensor to prevent SDPA sharding propogation errors (b/482035664)
    Used in torchtitan.models.attention.ScaledDotProductAttentionWrapper
    """
    # get device metadata
    mesh = q.device_mesh
    placements = q.placements

    # only unwrap if not already local from previous ops
    q_in = q.to_local()
    k_in = k.to_local() if isinstance(k, torch.distributed.tensor.DTensor) else k
    v_in = v.to_local() if isinstance(v, torch.distributed.tensor.DTensor) else v

    with sdpa_kernel(sdpa_backends, set_priority=True):
        output_local = F.scaled_dot_product_attention(
            q_in, k_in, v_in, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa
        )

    # rewrap to original sharding
    return torch.distributed.tensor.DTensor.from_local(output_local, mesh, placements)


def use_splash_attention_patch(
    model: nn.Module,
    block_q: int | None = None,
    block_kv: int | None = None,
    block_dkv: int | None = None,
    block_kv_compute: int | None = None,
    block_q_dkv: int | None = None,
    block_kv_dkv: int | None = None,
    block_kv_dkv_compute: int | None = None,
    block_q_dq: int | None = None,
    block_kv_dq: int | None = None,
    use_fused_bwd_kernel: bool | None = None,
    q_layout: str | None = None,
    k_layout: str | None = None,
    v_layout: str | None = None,
) -> None:
  """Patches splash attention into the model."""
  from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa  # pylint: disable=g-import-not-at-top
  import types

  # Helper to filter out None values so we don't override defaults in splash_sdpa
  def _get_kwargs(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}

  def _splash_forward(
      self, q, k, v, *, scale=None, enable_gqa=False, is_causal=True):
    """Replace ScaledDotProductAttentionWrapper.forward with splash_sdpa."""
    kwargs = _get_kwargs(
        block_q=block_q,
        block_kv=block_kv,
        block_dkv=block_dkv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_kv_dkv_compute=block_kv_dkv_compute,
        block_q_dq=block_q_dq,
        block_kv_dq=block_kv_dq,
        use_fused_bwd_kernel=use_fused_bwd_kernel,
        q_layout=q_layout,
        k_layout=k_layout,
        v_layout=v_layout,
    )
    return splash_sdpa(
        q,
        k,
        v,
        scale=scale,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
        **kwargs,
    )

  def _splash_sdpa_redacted(self, *, query, key, value, scale=None, **kwargs):
    """Replaces [redacted]'s native SDPA with splash_sdpa.

    This function replaces ScaledDotProductAttention._sdpa_native_torch with
    `splash_sdpa`. [redacted]'s attention layer requires special handling because
    unlike torchtitan's wrapper which works with shape (B, H, S, D),
    it expects inputs of shape (B, S, H, D) and handles GQA by tiling
    k/v heads before SDPA. This wrapper mimics [redacted]'s GQA handling,
    transposes inputs to (B, H, S, D) for splash_sdpa, and transposes
    the output back to (B, S, H, D).
    """
    query, key, value = self._maybe_tile_qkv_for_gqa(query, key, value)
    # Transpose from (B, S, H, D) to (B, H, S, D).
    query = query.transpose(-3, -2)
    key = key.transpose(-3, -2)
    value = value.transpose(-3, -2)

    call_kwargs = _get_kwargs(
        block_q=block_q,
        block_kv=block_kv,
        block_dkv=block_dkv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_kv_dkv_compute=block_kv_dkv_compute,
        block_q_dq=block_q_dq,
        block_kv_dq=block_kv_dq,
        use_fused_bwd_kernel=use_fused_bwd_kernel,
        q_layout=q_layout,
        k_layout=k_layout,
        v_layout=v_layout,
    )
    # splash_sdpa expects enable_gqa=False as heads are already tiled.
    out = splash_sdpa(
        query,
        key,
        value,
        scale=scale,
        is_causal=True,
        enable_gqa=False,
        **call_kwargs,
    )
    # Transpose back to match [redacted]'s expected layout
    return out.transpose(-3, -2)

  # Patch all attention modules in the model
  _patched = 0
  for _, module in model.named_modules():
    cls_name = type(module).__name__
    if "ScaledDotProductAttentionWrapper" in cls_name:
      module.forward = types.MethodType(_splash_forward, module)
      _patched += 1
    elif cls_name == "ScaledDotProductAttention":
      module._sdpa_native_torch = types.MethodType(_splash_sdpa_redacted, module)
      _patched += 1

  logger.info(
      "Splash attention ENABLED: patched %d attention modules",
      _patched)


class PallasLossLinear(nn.Module):
  """Wrapper for model.output to return (x, weight) for Pallas loss."""

  def __init__(self, original_linear: nn.Linear):
    super().__init__()
    self.original_linear = original_linear
    self.weight = original_linear.weight

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x, self.weight.t()


def use_output_projection_patch(
    model: nn.Module,
) -> None:
  """Replaces model.output with an identity-like module that returns (x, weight)."""
  if hasattr(model, "output") or hasattr(model, "output_transform"):
    if isinstance(model.output, nn.Linear):
      model.output = PallasLossLinear(model.output)
      logger.info("Patched model.output to return (x, weight) for Pallas loss")
    elif model.output is None:
      logger.warning("model.output is None, cannot wrap for Pallas loss")
    else:
      logger.warning(
          "model.output is of type %s, not nn.Linear, "
          "cannot wrap for Pallas loss",
          type(model.output),
      )
  else:
    logger.warning("model has no output attribute, cannot wrap for Pallas loss")


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
  import torch.nn.functional as F  # pytype: disable=import-error
  import torchtitan.models.moe.moe as torchtitan_moe  # pytype: disable=import-error

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

  torchtitan_moe.indices_padding_wrapper = bypassed_padding_wrapper
  torchtitan_moe._run_experts_grouped_mm = _custom_run_experts_grouped_mm


def use_fill_indices_patch(
    model: nn.Module,
) -> None:
  """Enables the fill indices kernel for MoE modules in the model.

  This workload-specific monkey patch replaces the default implementations in
  torchtitan.models.moe.kernels with our TPU-specific JAX implementation.
  """
  from torchtitan.experiments.tpu.kernels import fill_indices as tpu_fill_indices
  from torchtitan.models.moe import kernels as moe_kernels

  # Patch both because generate_permute_indices defaults to use_cpu=True
  moe_kernels.fill_indices_wrapper = tpu_fill_indices.fill_indices
  moe_kernels.fill_indices_cpu = tpu_fill_indices.fill_indices
  logger.info("Patched MoE kernels to use TPU fill_indices")


def use_cpu_safe_histc_patch() -> None:
  """Monkey patches torch.histc to support int dtypes on CPU."""
  original_histc = torch.histc

  def _cpu_safe_histc(input, bins=100, min=0, max=0, *, out=None):
    if input.device.type == "cpu":
      return original_histc(input.float(), bins, min, max, out=out).to(input.dtype)
    return original_histc(input, bins, min, max, out=out)

  torch.histc = _cpu_safe_histc
  logger.info("Monkey-patched torch.histc to support int dtypes on CPU")
