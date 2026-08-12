# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from torchtitan.models.common.config_utils import update_moe_aux_loss_configs
from torchtitan.models.common.moe import (
    _AuxLossAutograd,
    _batch_wise_load_balance_loss,
    _sequence_wise_load_balance_loss,
    MoELoadBalanceAuxLoss,
)


@pytest.mark.parametrize(
    "loss_fn",
    [_sequence_wise_load_balance_loss, _batch_wise_load_balance_loss],
)
def test_aux_loss_injection_matches_explicit_loss(loss_fn):
    """
    Verify aux-loss injection matches adding the aux loss to the main loss.

    _AuxLossAutograd forwards top-k scores unchanged, then injects a unit
    gradient for the precomputed aux loss during backward. This should produce
    the same score gradients as the ordinary autograd expression
    (topk_scores.sum() + aux_loss).backward().
    """
    B, L, E, K = 3, 5, 7, 2
    coeff = 0.125
    torch.manual_seed(0)

    scores_BLE = torch.rand(B, L, E, dtype=torch.double, requires_grad=True)
    topk_expert_ids_BLK = torch.topk(scores_BLE.detach(), k=K, dim=-1).indices

    injected_scores_BLE = scores_BLE.clone().detach().requires_grad_(True)
    injected_topk_scores_BLK = injected_scores_BLE.gather(
        dim=-1, index=topk_expert_ids_BLK
    )
    injected_aux_loss = loss_fn(
        injected_scores_BLE, topk_expert_ids_BLK, K, coeff
    )
    injected_output = _AuxLossAutograd.apply(
        injected_topk_scores_BLK, injected_aux_loss
    )
    injected_output.sum().backward()

    explicit_scores_BLE = scores_BLE.clone().detach().requires_grad_(True)
    explicit_topk_scores_BLK = explicit_scores_BLE.gather(
        dim=-1, index=topk_expert_ids_BLK
    )
    explicit_aux_loss = loss_fn(explicit_scores_BLE, topk_expert_ids_BLK, K, coeff)
    (explicit_topk_scores_BLK.sum() + explicit_aux_loss).backward()

    torch.testing.assert_close(
        injected_scores_BLE.grad,
        explicit_scores_BLE.grad,
        rtol=1e-12,
        atol=1e-12,
    )


def test_batch_wise_aux_loss_matches_qwen_definition():
    scores_BLE = torch.tensor(
        [
            [[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]],
            [[0.3, 0.2, 0.5], [0.6, 0.1, 0.3]],
        ]
    )
    topk_expert_ids_BLK = torch.tensor(
        [
            [[0, 1], [1, 2]],
            [[2, 0], [0, 1]],
        ]
    )
    coeff = 1e-3

    actual = _batch_wise_load_balance_loss(
        scores_BLE,
        topk_expert_ids_BLK,
        top_k=2,
        aux_loss_coeff=coeff,
    )

    counts_E = torch.tensor([3.0, 3.0, 2.0])
    num_tokens = counts_E.sum() / 2
    token_fraction_E = counts_E * (3 / (2 * num_tokens))
    probability_fraction_E = scores_BLE.sum(dim=(0, 1)) / num_tokens
    expected = coeff * torch.sum(token_fraction_E * probability_fraction_E)
    torch.testing.assert_close(actual, expected)


def test_batch_wise_aux_loss_rejects_gradient_accumulation():
    aux_loss = MoELoadBalanceAuxLoss.Config(
        type="batch_wise",
        coeff=1e-3,
        top_k=2,
    )
    layer = SimpleNamespace(moe=SimpleNamespace(aux_loss=aux_loss))

    update_moe_aux_loss_configs(
        [layer],
        pp_enabled=False,
        local_global_batch_size=16,
        global_batch_size=16,
    )
    assert aux_loss.global_batch_size == 16

    with pytest.raises(ValueError, match="gradient accumulation"):
        update_moe_aux_loss_configs(
            [layer],
            pp_enabled=False,
            local_global_batch_size=16,
            global_batch_size=32,
        )
