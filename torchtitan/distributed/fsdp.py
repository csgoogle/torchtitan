# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule


def disable_fsdp_gradient_division(model: nn.Module) -> None:
  """
    Disable FSDP's automatic gradient division for all FSDP modules.

    Set gradient_divide_factor=1.0 to disable FSDP's automatic gradient division.
    We handle gradient scaling ourselves in the training loop with global token count.

    Note: This also works for ReplicateModule since it inherits from FSDPModule.

    Args:
        model: The model containing FSDP-wrapped or Replicate-wrapped modules
    """
  # The `premul_sum` reduction type is not supported on non-CUDA devices
  # (e.g., TPU), but it may be used by FSDP on CUDA devices.
  # To avoid runtime errors on non-CUDA devices, we explicitly disable
  # `premul_sum` and force post-multiply mode via
  # `set_force_sum_reduction_for_comms(True)`.
  # Since `gradient_divide_factor` is set to 1.0 in the code below,
  # this results in a numerically equivalent outcome between CUDA and
  # non-CUDA devices (i.e., A/1+B/1+C/1 == (A+B+C)/1).
  device = next(model.parameters()).device
  for module in model.modules():
    if isinstance(module, FSDPModule):
      if device.type != "cuda":
        module.set_force_sum_reduction_for_comms(True)
      module.set_gradient_divide_factor(1.0)


def get_fsdp_reshard_after_forward_policy(
    reshard_after_forward_policy: str, pp_enabled: bool
) -> bool:
    """Resolve fsdp_reshard_after_forward policy string to a boolean.

    Args:
        reshard_after_forward_policy: One of "always", "never", or "default".
        pp_enabled: Whether pipeline parallelism is enabled.

    Returns:
        Boolean indicating whether to reshard after forward.
    """
    match reshard_after_forward_policy:
        case "always":
            return True
        case "never":
            return False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            return not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )
