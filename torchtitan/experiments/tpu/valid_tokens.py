# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import torch
from torch.distributed import _functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d


def loss_with_global_valid_token_sum(
    loss_fn,
    batch_mesh,
    *,
    all_reduce: Callable[..., torch.Tensor] = funcol.all_reduce,
):
    """Wrap a loss so its token-count SUM is part of the traced loss graph."""

    def loss_with_global_valid_tokens(pred, labels, *, global_valid_tokens):
        if batch_mesh is not None:
            global_valid_tokens = all_reduce(
                global_valid_tokens,
                reduceOp=c10d.ReduceOp.SUM.name,
                group=batch_mesh,
            )
        return loss_fn(
            pred,
            labels,
            global_valid_tokens=global_valid_tokens,
        )

    return loss_with_global_valid_tokens
