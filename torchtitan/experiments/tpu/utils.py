# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.algorithms._checkpoint import checkpoint_wrapper


__all__ = ["ptd_checkpoint_wrapper_with_early_stop"]


def ptd_checkpoint_wrapper_with_early_stop(
    module: torch.nn.Module,
    checkpoint_impl: checkpoint_wrapper.CheckpointImpl = checkpoint_wrapper.CheckpointImpl.NO_REENTRANT,
    checkpoint_fn=None,
    early_stop: bool = False,
    **checkpoint_fn_kwargs,
) -> torch.nn.Module:
    """Wraps a module with PyTorch Distributed's checkpoint_wrapper.

    This function is a wrapper around
    `torch.distributed.algorithms._checkpoint.checkpoint_wrapper`
    with an added `early_stop` parameter, although early stopping is not
    currently supported in g3.

    Args:
        module: The module to wrap.
        checkpoint_impl: The checkpoint implementation to use.
        checkpoint_fn: Optional function to use for checkpointing.
        early_stop: Whether to enable early stopping.
            Currently raises NotImplementedError if True.
        **checkpoint_fn_kwargs: Additional keyword arguments to pass to
            `checkpoint_fn`.

    Returns:
        The wrapped module.

    Raises:
        NotImplementedError: If `early_stop` is set to True.
    """
    if early_stop:
      raise NotImplementedError(
          "early_stop is not supported by ptd_checkpoint_wrapper."
      )

    checkpoint_fn_kwargs["early_stop"] = early_stop

    return checkpoint_wrapper.checkpoint_wrapper(
        module,
        checkpoint_impl=checkpoint_impl,
        checkpoint_fn=checkpoint_fn,
        **checkpoint_fn_kwargs,
    )
