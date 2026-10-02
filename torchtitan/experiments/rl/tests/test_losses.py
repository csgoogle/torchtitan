# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the RL policy losses."""

from __future__ import annotations

import math

import pytest
import torch

from torchtitan.components.loss import compute_logprobs
from torchtitan.experiments.rl.losses import DAPOLoss, GRPOLoss


def _inputs(ratios: list[float], advantages: list[float]):
    """Packed [T, V] logits / [T] labels plus generator logprobs that give each
    token the requested importance ratio pi_theta / pi_old under the loss."""
    torch.manual_seed(0)
    num_tokens = len(ratios)
    logits = torch.randn(num_tokens, 11)
    labels = torch.randint(0, 11, (num_tokens,))
    trainer_logprobs = compute_logprobs(logits, labels)
    generator_logprobs = trainer_logprobs - torch.tensor([math.log(r) for r in ratios])
    return dict(
        logits=logits,
        labels=labels,
        generator_logprobs=generator_logprobs,
        advantages=torch.tensor(advantages),
        loss_mask=torch.ones(num_tokens, dtype=torch.bool),
    )


def _clipped_frac(loss_fn, inputs) -> float:
    _, metrics = loss_fn(global_valid_tokens=inputs["labels"].numel(), **inputs)
    return metrics["loss/ratio_clipped_frac"].item()


def test_clipped_frac_counts_only_the_clipped_branch() -> None:
    loss_fn = DAPOLoss(DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28))
    # ratio, advantage -> which term min() selects
    cases = [
        (1.5, +1.0),  # above band, A > 0: clipped (zero gradient)
        (0.5, -1.0),  # below band, A < 0: clipped (zero gradient)
        (1.5, -1.0),  # above band, A < 0: unclipped term, still trains
        (0.5, +1.0),  # below band, A > 0: unclipped term, still trains
        (1.5, 0.0),  # above band, A == 0: no clipping
        (1.1, +1.0),  # inside band
        (0.9, -1.0),  # inside band
        (1.25, +1.0),  # inside the clip-higher band, above 1 + ratio_clip_low
    ]
    inputs = _inputs([r for r, _ in cases], [a for _, a in cases])
    assert _clipped_frac(loss_fn, inputs) == pytest.approx(2 / len(cases))


def test_clipped_frac_matches_tokens_with_zero_surrogate_gradient() -> None:
    # The clipped term carries no gradient w.r.t. the logits, so the metric
    # must equal the fraction of nonzero-advantage tokens whose gradient is 0.
    loss_fn = GRPOLoss(GRPOLoss.Config(clip_eps=0.2))
    ratios = [0.3, 0.7, 0.85, 1.0, 1.15, 1.3, 2.0, 0.6, 1.4, 1.05]
    advantages = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0]
    inputs = _inputs(ratios, advantages)
    inputs["logits"].requires_grad_(True)
    loss, metrics = loss_fn(global_valid_tokens=len(ratios), **inputs)
    loss.backward()
    zero_grad = (inputs["logits"].grad.abs().sum(dim=-1) == 0).float()
    assert metrics["loss/ratio_clipped_frac"].item() == pytest.approx(
        zero_grad.mean().item()
    )
    # 0.3 and 0.6 (A<0), 1.3 and 1.4 (A>0) are clipped; 0.7 (A>0) and 2.0 (A<0)
    # are outside the band but keep the unclipped term.
    assert metrics["loss/ratio_clipped_frac"].item() == pytest.approx(4 / len(ratios))
