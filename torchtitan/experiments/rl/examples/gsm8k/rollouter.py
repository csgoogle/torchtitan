# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.gsm8k.data import eval_subset, GSM8KDataset
from torchtitan.experiments.rl.examples.gsm8k.env import GSM8KEnv
from torchtitan.experiments.rl.examples.gsm8k.rubric import RewardBoxedAnswer
from torchtitan.experiments.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.experiments.rl.rubrics import Rubric


class GSM8KWorker(RolloutWorker):
    """GSM8K's single-turn env and ``\\boxed{}`` reward. Pure config."""

    @dataclass(kw_only=True, slots=True)
    class Config(RolloutWorker.Config):
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardBoxedAnswer.Config(weight=1.0)],
                # A truncated reply has no final answer.
                truncation_reward=0.0,
            )
        )
        message_env: GSM8KEnv.Config = field(default_factory=GSM8KEnv.Config)
        token_env: TokenEnv.Config = field(
            default_factory=lambda: TokenEnv.Config(max_num_turns=1)
        )


class GSM8KRollouter(Rollouter):
    """GSM8K: the train split for rollouts and a fixed 512-row test subset for
    validation; the env and reward live in `GSM8KWorker`. Pure config."""

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: GSM8KDataset.Config = field(
            default_factory=lambda: GSM8KDataset.Config(split="train", seed=42)
        )
        validation_dataset: GSM8KDataset.Config = field(
            default_factory=lambda: GSM8KDataset.Config(
                split="test", row_indices=eval_subset(1319, 512), shuffle=False
            )
        )
        worker: GSM8KWorker.Config = field(default_factory=GSM8KWorker.Config)
