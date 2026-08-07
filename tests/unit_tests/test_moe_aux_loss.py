# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from types import SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from torch import nn

from torchtitan.components.metrics import TensorBoardLogger
from torchtitan.components.optimizer import get_moe_load_balancing_metrics
from torchtitan.models.common.config_utils import update_moe_aux_loss_configs
from torchtitan.models.common.moe import (
    _AuxLossAutograd,
    _batch_wise_load_balance_loss,
    _record_moe_aux_loss_telemetry,
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


def test_aux_loss_metrics_include_route_histogram():
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
    router_logits_BLE = torch.log(scores_BLE)

    aux_loss = MoELoadBalanceAuxLoss.Config(
        type="batch_wise",
        coeff=1e-3,
        top_k=2,
    ).build()
    _record_moe_aux_loss_telemetry(
        aux_loss,
        scores_BLE,
        topk_expert_ids_BLK,
        router_logits_BLE=router_logits_BLE,
        expert_bias_E=None,
    )

    class FakeMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.aux_loss = aux_loss

    class FakeBlock(nn.Module):
        moe_enabled = True

        def __init__(self):
            super().__init__()
            self.moe = FakeMoE()

    class FakeModelPart(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict({"0": FakeBlock()})

    metrics = get_moe_load_balancing_metrics([FakeModelPart()])

    assert json.loads(metrics["moe/route_counts_json"]) == [[3, 3, 2]]
    assert not any(tag.startswith("moe/route_count/layer_") for tag in metrics)
    torch.testing.assert_close(
        metrics["moe/route_assignments/layer_000"],
        torch.tensor(8.0),
    )
    assert "moe/global_aux_loss_normalized" in metrics
    assert "moe/router_logits_abs_mean" in metrics


def test_route_histogram_round_trips_as_one_tensorboard_event(tmp_path):
    logger = TensorBoardLogger(str(tmp_path))
    logger.log({"moe/route_counts_json": "[[3,3,2]]"}, step=1)
    logger.close()

    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    tag = "moe/route_counts_json/text_summary"
    events = accumulator.Tensors(tag)

    assert len(events) == 1
    assert events[0].step == 1
    assert json.loads(events[0].tensor_proto.string_val[0].decode()) == [[3, 3, 2]]


def test_tensorboard_scalar_metrics_share_one_event(tmp_path):
    logger = TensorBoardLogger(str(tmp_path))
    logger.log(
        {
            "loss_metrics/global_avg_loss": 1.25,
            "moe/global_aux_loss_normalized": torch.tensor(1.01),
            "moe/route_counts_json": "[[3,3,2]]",
        },
        step=7,
    )
    logger.close()

    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    assert accumulator.Scalars("loss_metrics/global_avg_loss")[0].value == 1.25
    assert accumulator.Scalars("moe/global_aux_loss_normalized")[
        0
    ].value == pytest.approx(1.01)
    assert json.loads(
        accumulator.Tensors("moe/route_counts_json/text_summary")[
            0
        ].tensor_proto.string_val[0].decode()
    ) == [[3, 3, 2]]

    event_files = list(tmp_path.glob("events.out.tfevents.*"))
    assert len(event_files) == 1
    expected_tags = {
        "loss_metrics/global_avg_loss",
        "moe/global_aux_loss_normalized",
        "moe/route_counts_json/text_summary",
    }
    scalar_events = [
        event
        for event in EventFileLoader(str(event_files[0])).Load()
        if expected_tags.intersection(value.tag for value in event.summary.value)
    ]
    assert len(scalar_events) == 1
    assert {value.tag for value in scalar_events[0].summary.value} == expected_tags
