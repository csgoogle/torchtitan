# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the `examples.gsm8k` task and its TPU recipe config.

The dataset tests need only torchtitan and `datasets`. The env and reward tests
also need `renderers`, and the recipe config tests need `vllm`, `monarch`,
`torchstore` and `renderers`. Those are optional dependencies, so each test that
needs one skips on its own when it is missing (the TPU CI image installs none).
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

import pytest
import torch
from datasets import Dataset

from torchtitan.experiments.rl.examples.gsm8k import (
    data as gsm8k_data,
    GSM8KDataset,
    GSM8KSample,
)
from torchtitan.experiments.rl.models.cast_linear import CastLinear
from torchtitan.models.common.linear import Linear

if TYPE_CHECKING:
    from torchtitan.experiments.rl.rollout import Rollout

_ROWS = {
    "question": [f"q{i}" for i in range(6)],
    "answer": [f"step {i}\n#### {1000 * i:,}" for i in range(6)],
}


def _skip_without(*names: str) -> None:
    for name in names:
        pytest.importorskip(name)


def _recipe_config():
    # The recipe config imports the controller, the vLLM generator and the
    # renderer.
    _skip_without("vllm", "monarch", "torchstore", "renderers")
    from torchtitan.experiments.rl.examples.gsm8k.config_registry import (
        rl_grpo_qwen3_0_6b_gsm8k_tpu,
    )

    return rl_grpo_qwen3_0_6b_gsm8k_tpu()


def _patch_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gsm8k_data, "load_dataset", lambda *a, **k: Dataset.from_dict(_ROWS)
    )


def _rollout(content: str) -> Rollout:
    # rollout.types imports renderers.
    _skip_without("renderers")
    from torchtitan.experiments.rl.rollout import Rollout, RolloutStatus, RolloutTurn
    from torchtitan.experiments.rl.types import RolloutTurnID

    turn = RolloutTurn(
        rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=0),
        prompt_token_ids=[],
        completion_token_ids=[],
        completion_logprobs=[],
        min_policy_version=0,
        completion_message={"role": "assistant", "content": content},
    )
    return Rollout(
        group_id=0, rollout_id=0, status=RolloutStatus.COMPLETED, turns=[turn]
    )


def test_gold_answer_strips_thousands_separators() -> None:
    assert gsm8k_data.gold_answer("48 + 24 = 72\n#### 72") == "72"
    assert gsm8k_data.gold_answer("... #### 1,600") == "1600"
    with pytest.raises(ValueError):
        gsm8k_data.gold_answer("no marker")


def test_eval_subset_is_the_measured_512_rows() -> None:
    rows = gsm8k_data.eval_subset(1319, 512)
    assert len(rows) == len(set(rows)) == 512
    assert rows[:3] == [0, 2, 5] and rows[-1] == (511 * 1319) // 512
    assert rows == sorted(rows)


def test_dataset_replays_the_selected_rows_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hub(monkeypatch)
    dataset = GSM8KDataset(
        GSM8KDataset.Config(split="test", row_indices=[1, 4], shuffle=False)
    )
    expected = [
        GSM8KSample(question="q1", answer="1000"),
        GSM8KSample(question="q4", answer="4000"),
    ]
    assert [next(dataset) for _ in range(4)] == expected * 2


def test_dataset_shuffle_is_seeded_and_resumable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hub(monkeypatch)
    a = GSM8KDataset(GSM8KDataset.Config(seed=7))
    b = GSM8KDataset(GSM8KDataset.Config(seed=7))
    assert [next(a) for _ in range(9)] == [next(b) for _ in range(9)]
    state = a.state_dict()
    expected = [next(a) for _ in range(5)]
    b.load_state_dict(state)
    assert [next(b) for _ in range(5)] == expected


def test_env_is_single_turn() -> None:
    # The env imports renderers (also through rl.environment).
    _skip_without("renderers")
    from torchtitan.experiments.rl.examples.gsm8k import GSM8KEnv
    from torchtitan.experiments.rl.examples.gsm8k.env import INSTRUCTION

    env = GSM8KEnv(GSM8KEnv.Config(), env_input=GSM8KSample(question="Q?", answer="1"))
    init = asyncio.run(env.init())
    assert init.init_prompt_messages == [
        {"role": "user", "content": "Q?" + INSTRUCTION}
    ]
    step = asyncio.run(env.step({"role": "assistant", "content": "\\boxed{1}"}))
    assert step.done


@pytest.mark.parametrize(
    "content, reward",
    [
        ("so the answer is \\boxed{1600}", 1.0),
        ("\\boxed{1,600}", 1.0),
        ("\\boxed{\\$1600.00}", 1.0),
        ("\\boxed{\\text{1600}}", 1.0),
        ("first \\boxed{12}, then \\boxed{1600}", 1.0),
        ("first \\boxed{1600}, then \\boxed{12}", 0.0),
        ("the answer is 1600", 0.0),
        ("\\boxed{1601}", 0.0),
    ],
)
def test_reward_reads_the_last_boxed_answer(content: str, reward: float) -> None:
    # The rubric imports renderers (through rl.rollout and rl.rubrics).
    _skip_without("renderers")
    from torchtitan.experiments.rl.examples.gsm8k import RewardBoxedAnswer

    reward_fn = RewardBoxedAnswer(RewardBoxedAnswer.Config())
    sample = GSM8KSample(question="q", answer="1600")
    assert asyncio.run(reward_fn(_rollout(content), sample)) == reward


def _measured_lr_factor(step: int) -> float:
    """The schedule of the measured runs, for 1-based optimizer step ``step``."""
    s = step - 1
    if s < 35:
        return s / 35.0
    return 0.1 + 0.45 * (1.0 + math.cos(math.pi * (s - 35) / 1135.0))


def test_recipe_keeps_fp32_masters_and_the_measured_lr_schedule() -> None:
    config = _recipe_config()
    trainer = config.trainer
    assert trainer.training.dtype == "float32"
    assert trainer.training.mixed_precision_param == "bfloat16"
    kwargs = trainer.optimizer.param_groups[0].optimizer_kwargs
    assert kwargs["lr"] == 2.5e-6 and kwargs["weight_decay"] == 0.0

    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    schedulers = trainer.lr_scheduler.build(
        optimizers=[optimizer], training_steps=trainer.training.steps
    )
    for step in range(1, config.async_loop.num_training_steps + 1):
        factor = schedulers.schedulers[0].get_last_lr()[0]
        if step <= 35:
            # torchtitan's warmup runs one step ahead of the measured one.
            assert factor == pytest.approx(_measured_lr_factor(step + 1))
        else:
            assert factor == pytest.approx(_measured_lr_factor(step))
        optimizer.step()
        schedulers.step()


def test_recipe_resolves_from_the_command_line() -> None:
    # `--module gsm8k`, as in the README's launch command.
    config = _recipe_config()
    from torchtitan.config import ConfigManager

    parsed = ConfigManager().parse_args(
        ["--module", "gsm8k", "--config", "rl_grpo_qwen3_0_6b_gsm8k_tpu"]
    )
    assert type(parsed) is type(config)
    assert parsed.async_loop.num_training_steps == config.async_loop.num_training_steps


def test_recipe_gives_the_generator_fp32_logits() -> None:
    # The trainer and the generator share model_spec. Its fp32 lm_head makes
    # the generator's logits fp32, so the vLLM TPU backend's stock Gumbel
    # uniform, rand_like(logits), is fp32 too. A bf16 uniform collapsed every run.
    model_config = _recipe_config().model_spec.model
    heads = [
        cfg
        for fqn, cfg, _, _ in model_config.traverse(Linear.Config)
        if fqn.rsplit(".", 1)[-1] == "lm_head"
    ]
    assert len(heads) == 1
    assert isinstance(heads[0], CastLinear.Config)
    assert heads[0].compute_dtype == "float32"
