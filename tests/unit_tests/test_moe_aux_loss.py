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
from torch import nn
from torch.nn import functional as F

from torchtitan.components.metrics import TensorBoardLogger
from torchtitan.components.optimizer.optimizer import (
    collect_moe_routing_metric_tensors,
    configure_moe_routing_metrics,
    get_moe_load_balancing_metrics,
    set_moe_routing_metric_tensors,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step
from torchtitan.models.common.config_utils import update_moe_aux_loss_configs
from torchtitan.models.common.moe import (
    _AuxLossAutograd,
    _batch_wise_load_balance_loss,
    _sequence_wise_load_balance_loss,
    _tokens_per_expert,
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


def test_tokens_per_expert_matches_dense_routing_map():
    expert_ids = torch.tensor(
        [
            [[0, 2], [3, 1], [2, 3]],
            [[1, 3], [0, 2], [3, 0]],
        ]
    )
    expected = F.one_hot(expert_ids, num_classes=4).sum(dim=(0, 1, 2))

    actual = _tokens_per_expert(expert_ids, num_experts=4)

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.int64


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


def test_aux_loss_metric_capture_does_not_change_output_or_gradient():
    torch.manual_seed(0)
    scores = torch.rand(2, 4, 5, dtype=torch.double)
    expert_ids = scores.topk(k=2, dim=-1).indices
    outputs = []
    gradients = []

    for enabled in (False, True):
        current_scores = scores.clone().requires_grad_(True)
        topk_scores = current_scores.gather(-1, expert_ids)
        aux_loss = MoELoadBalanceAuxLoss.Config(
            type="batch_wise",
            coeff=1e-3,
            top_k=2,
        ).build()
        aux_loss.collect_routing_metrics = enabled
        output = aux_loss(topk_scores, current_scores, expert_ids)
        output.sum().backward()
        outputs.append(output.detach())
        gradients.append(current_scores.grad)

    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(gradients[0], gradients[1])


def test_routing_metrics_ignore_model_parts_without_top_level_layers():
    class FakeModelWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(2, 2))

    model_parts = [FakeModelWrapper()]

    assert not configure_moe_routing_metrics(model_parts, enabled=True)
    assert collect_moe_routing_metric_tensors(model_parts) == []
    assert get_moe_load_balancing_metrics(model_parts) == {}


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
    aux_loss = MoELoadBalanceAuxLoss.Config(
        type="batch_wise",
        coeff=1e-3,
        top_k=2,
    ).build()
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

    model_parts = [FakeModelPart()]
    assert configure_moe_routing_metrics(model_parts, enabled=True)

    topk_scores_BLK = scores_BLE.gather(-1, topk_expert_ids_BLK)
    aux_loss(topk_scores_BLK, scores_BLE, topk_expert_ids_BLK)
    normalized_aux_loss = aux_loss._last_local_normalized_aux_loss
    set_moe_routing_metric_tensors(
        model_parts,
        [torch.tensor([[3.0, 3.0, 2.0]]), normalized_aux_loss.unsqueeze(0)],
    )
    captured_counts, captured_aux_loss = collect_moe_routing_metric_tensors(
        model_parts
    )
    torch.testing.assert_close(captured_counts, torch.tensor([[3.0, 3.0, 2.0]]))
    torch.testing.assert_close(captured_aux_loss, normalized_aux_loss.unsqueeze(0))

    metrics = get_moe_load_balancing_metrics(model_parts)
    expected_normalized_aux_loss = _batch_wise_load_balance_loss(
        scores_BLE,
        topk_expert_ids_BLK,
        top_k=2,
        aux_loss_coeff=1.0,
    )

    assert json.loads(metrics["moe/route_counts_json"]) == [[3, 3, 2]]
    assert not any(tag.startswith("moe/route_count/layer_") for tag in metrics)
    torch.testing.assert_close(
        metrics["moe/route_assignments/layer_000"],
        torch.tensor(8.0),
    )
    torch.testing.assert_close(
        metrics["moe/global_aux_loss_normalized"], expected_normalized_aux_loss
    )
    assert "moe/router_logits_abs_mean" not in metrics
    assert "moe/dropped_routes_total" not in metrics


def test_graph_trainer_returns_routing_metrics_as_graph_outputs():
    class FakeMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.aux_loss = MoELoadBalanceAuxLoss.Config(
                type="batch_wise",
                coeff=1e-3,
                top_k=2,
            ).build()

    class FakeBlock(nn.Module):
        moe_enabled = True

        def __init__(self):
            super().__init__()
            self.moe = FakeMoE()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3, 3))
            self.layers = nn.ModuleDict({"0": FakeBlock()})

        def forward(self, x, **_kwargs):
            scores = (x @ self.weight).softmax(dim=-1)
            expert_ids = scores.topk(k=2, dim=-1).indices
            topk_scores = scores.gather(-1, expert_ids)
            aux_loss = self.layers["0"].moe.aux_loss
            topk_scores = aux_loss(topk_scores, scores, expert_ids)
            aux_loss._last_local_tokens_per_expert_E = (
                F.one_hot(expert_ids, num_classes=3)
                .sum(dim=(0, 1, 2))
                .detach()
                .float()
            )
            return topk_scores

    model = FakeModel()
    assert configure_moe_routing_metrics([model], enabled=True)

    def loss_fn(pred, labels, *, global_valid_tokens):
        return ((pred - labels) ** 2).sum() / global_valid_tokens

    fwd_bwd_step = make_fwd_bwd_step(
        model,
        loss_fn,
        collect_moe_routing_metrics=True,
    )
    inputs = torch.randn(2, 4, 3)
    labels = torch.randn(2, 4, 2)
    eager_outputs = fwd_bwd_step(inputs, labels, 8.0, {})
    traced_step = minimal_fx_tracer(fwd_bwd_step, module=model)(
        inputs, labels, 8.0, {}
    )
    traced_outputs = run_traced(traced_step, module=model)(
        inputs, labels, 8.0, {}
    )

    assert len(traced_outputs) == 4
    assert traced_outputs[-2].shape == (1, 3)
    assert traced_outputs[-1].shape == (1,)
    for eager, traced in zip(eager_outputs, traced_outputs, strict=True):
        torch.testing.assert_close(eager, traced)


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


def test_tensorboard_logs_scalar_and_text_metrics(tmp_path):
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
