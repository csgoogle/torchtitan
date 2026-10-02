# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""bf16 master weights drop small updates; RL configs should keep fp32 masters."""

from __future__ import annotations

import logging

import pytest
import torch

from torchtitan.config import TrainingConfig


def _skip_without(*names: str) -> None:
    for name in names:
        pytest.importorskip(name)


def _warn_on_bf16_master_weights(training: TrainingConfig) -> None:
    # The trainer actor module imports monarch and torchstore.
    _skip_without("monarch", "torchstore")
    from torchtitan.experiments.rl.actors import trainer

    trainer._warn_on_bf16_master_weights(training)


def test_bf16_master_weights_round_away_rl_sized_updates() -> None:
    # The arithmetic behind the warning: a step of order lr = 2.5e-6 is below
    # half a bf16 ulp for weights inside a binade at |w| >= 2**-10, so
    # round-to-nearest returns the same bf16 weight, while fp32 keeps it.
    w = torch.tensor([0.0015, -0.02, 0.5], dtype=torch.float32)
    for step in (2.5e-6, -2.5e-6):
        assert torch.equal(w.bfloat16() + step, w.bfloat16())
        assert not torch.equal(w + step, w)


def test_bf16_master_weights_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _warn_on_bf16_master_weights(TrainingConfig(dtype="bfloat16"))
    assert "rounded away" in caplog.text


def test_fp32_master_weights_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _warn_on_bf16_master_weights(TrainingConfig())
    assert caplog.text == ""


@pytest.mark.parametrize(
    "name",
    [
        "rl_grpo_qwen3_0_6b_flex",
        "rl_grpo_qwen3_0_6b_flex_batch_invariant",
        "rl_grpo_qwen3_0_6b_varlen",
        "rl_grpo_qwen3_0_6b_varlen_batch_invariant",
    ],
)
def test_qwen3_0_6b_recipes_keep_fp32_master_weights(name: str) -> None:
    # The example configs import the controller, generator and renderer.
    _skip_without("vllm", "monarch", "torchstore", "renderers")
    from torchtitan.experiments.rl.examples.alphabet_sort import config_registry

    training = getattr(config_registry, name)().trainer.training
    assert training.dtype == "float32"
    assert training.mixed_precision_param == "bfloat16"
