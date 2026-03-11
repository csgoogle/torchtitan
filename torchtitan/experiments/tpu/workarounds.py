import logging
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode


def attention_sdpa_forward_dtensor_workaround(sdpa_backends, q, k, v, scale=None):
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
            q_in, k_in, v_in, scale=scale, is_causal=True
        )

    # rewrap to original sharding
    return torch.distributed.tensor.DTensor.from_local(output_local, mesh, placements)




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