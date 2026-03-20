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
) -> None:
  """Patches splash attention into the model."""
  from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa  # pylint: disable=g-import-not-at-top

  def _splash_forward(
      self, q, k, v, *, scale=None, enable_gqa=False, is_causal=True):
    """Replace ScaledDotProductAttentionWrapper.forward with splash_sdpa."""
    return splash_sdpa(
          q, k, v,
          scale=scale,
          is_causal=is_causal,
          enable_gqa=enable_gqa,
      )

  # Patch all attention modules in the model
  _patched = 0
  for _, module in model.named_modules():
    cls_name = type(module).__name__
    if "ScaledDotProductAttentionWrapper" in cls_name:
      import types
      module.forward = types.MethodType(_splash_forward, module)
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
  if hasattr(model, "output"):
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


class CPUSafeHistcMode(TorchDispatchMode):
  """A global dispatcher that intercepts and casts calls to torch.histc.

  It implements the following workaround, because PyTorch's torch.histc
  implementation on CPU doesn't support int dtypes.
  This is in contrast to the GPU and TPU implementations, which support both.

  Pipeline:
    original_dtype -> float32 -> torch.histc -> original_dtype
  """

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    kwargs = kwargs or {}

    if func.overloadpacket == torch.ops.aten.histc:
      # Unpack arguments.
      input_tensor, bins, min_val, max_val = args

      if input_tensor.device.type == 'cpu':
        input_fp32 = input_tensor.to(torch.float32)

        new_args = (input_fp32, bins, min_val, max_val)
        result = func(*new_args, **kwargs)

        return result.to(input_tensor.dtype)

    return func(*args, **kwargs)


_global_histc_mode = CPUSafeHistcMode()
_global_histc_mode.__enter__()

logging.info('[Workaround] Globally enabled torch.histc workaround for CPU.')
