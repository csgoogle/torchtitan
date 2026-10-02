# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""0/1 GSM8K reward: the last ``\\boxed{...}`` in the reply equals the gold number."""

from __future__ import annotations

import re
from dataclasses import dataclass

from torchtitan.experiments.rl.examples.gsm8k.data import GSM8KSample
from torchtitan.experiments.rl.rollout import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn


def last_boxed(text: str) -> str | None:
    """Payload of the last brace-balanced ``\\boxed{...}`` in ``text``, or None."""
    best = None
    start = 0
    while (i := text.find("\\boxed{", start)) >= 0:
        j, depth = i + len("\\boxed{"), 1
        while j < len(text) and depth:
            depth += (text[j] == "{") - (text[j] == "}")
            j += 1
        if depth == 0:
            best = text[i + len("\\boxed{") : j - 1]
        start = i + len("\\boxed{")
    return best


def _normalize(answer: str) -> str:
    s = answer.strip()
    s = re.sub(r"\\(?:text|mathrm|mbox)\s*\{([^{}]*)\}", r"\1", s)
    for token in ("\\$", "$", "\\%", "%", "\\!", "\\,"):
        s = s.replace(token, "")
    s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)  # thousands separators
    return re.sub(r"\s+", "", s).rstrip(".")


def answers_match(prediction: str, gold: str) -> bool:
    """Numeric match (relative tolerance 1e-6), else exact match after normalization."""
    p, g = _normalize(prediction), _normalize(gold)
    try:
        return abs(float(p) - float(g)) <= 1e-6 * max(1.0, abs(float(g)))
    except ValueError:
        return p == g


class RewardBoxedAnswer(RewardFn):
    """1.0 if the final reply's last ``\\boxed{}`` matches the gold answer, else 0.0."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: GSM8KSample) -> float:
        if not rollout.turns:
            return 0.0
        message = rollout.turns[-1].completion_message or {}
        content = message.get("content")
        boxed = last_boxed(content) if isinstance(content, str) else None
        if boxed is None:
            return 0.0
        return 1.0 if answers_match(boxed, env_input.answer) else 0.0


__all__ = ["RewardBoxedAnswer"]
