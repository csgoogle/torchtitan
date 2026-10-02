# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.experiments.rl.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)
from torchtitan.experiments.rl.examples.gsm8k.data import GSM8KSample

# Appended to every question, as in the measured runs.
INSTRUCTION = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)


class GSM8KEnv(MessageEnv):
    """Single-turn GSM8K: one user message with the question; the reply ends the rollout."""

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, env_input: GSM8KSample) -> None:
        self._question = env_input.question

    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(
            init_prompt_messages=[
                {"role": "user", "content": self._question + INSTRUCTION}
            ]
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        del completion_message  # single turn: the first reply is the answer
        return MessageEnvStepOutput(done=True)
