# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from datasets import load_dataset

from torchtitan.config import Configurable

# openai/gsm8k ("main" config) at the revision the measured runs used.
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def gold_answer(solution: str) -> str:
    """The numeric answer after ``####`` in a GSM8K solution, without thousands commas.

    Example::

        gold_answer("... so 48 + 24 = 72 clips.\\n#### 72")  # -> "72"
        gold_answer("... #### 1,600")  # -> "1600"
    """
    if solution.count("####") != 1:
        raise ValueError(f"expected exactly one '####' in {solution!r}")
    return solution.split("####")[1].strip().replace(",", "")


def eval_subset(num_rows: int, size: int) -> list[int]:
    """A fixed, evenly spread subset of ``size`` row indices out of ``num_rows``.

    The measured runs scored held-out pass@1 on ``eval_subset(1319, 512)`` of the
    GSM8K test split: rows ``(i * 1319) // 512`` for ``i < 512``.
    """
    if not 0 < size <= num_rows:
        raise ValueError(f"need 0 < size <= num_rows, got {size=} {num_rows=}")
    return [(i * num_rows) // size for i in range(size)]


@dataclass(frozen=True, kw_only=True, slots=True)
class GSM8KSample:
    """One GSM8K problem: the question and its numeric gold answer."""

    question: str
    answer: str


class GSM8KDataset(Configurable):
    """Endless, seeded stream of GSM8K problems from the HF Hub ``openai/gsm8k``.

    Row order is shuffled with ``seed`` and reshuffled on each wrap. With
    ``shuffle=False`` the rows (or ``row_indices``) are replayed in order, so every
    validation pass of ``len(row_indices)`` samples sees the same problems.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        split: str = "train"
        """``train`` (7,473 rows) or ``test`` (1,319 rows)."""

        repo_id: str = "openai/gsm8k"
        revision: str | None = GSM8K_REVISION
        """HF Hub revision to load; None loads the latest."""

        data_path: str | None = None
        """Local parquet file with ``question`` / ``answer`` columns; overrides the
        HF download when set (offline use)."""

        row_indices: list[int] | None = None
        """Keep only these rows (e.g. ``eval_subset(1319, 512)``). None keeps all."""

        seed: int = 42
        shuffle: bool = True

    def __init__(self, config: Config) -> None:
        if config.data_path:
            dataset = load_dataset(
                "parquet", data_files=config.data_path, split="train"
            )
        else:
            dataset = load_dataset(
                config.repo_id,
                "main",
                split=config.split,
                revision=config.revision,
            )
        questions = [str(q) for q in dataset["question"]]
        answers = [gold_answer(str(a)) for a in dataset["answer"]]
        indices = (
            list(config.row_indices)
            if config.row_indices is not None
            else list(range(len(questions)))
        )
        if not indices:
            raise ValueError("GSM8KDataset has no rows")
        self._samples = [
            GSM8KSample(question=questions[i], answer=answers[i]) for i in indices
        ]
        self._rng = random.Random(config.seed)
        self._shuffle = config.shuffle
        self._order = list(range(len(self._samples)))
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._pos = 0

    def __iter__(self) -> Iterator[GSM8KSample]:
        return self

    def __next__(self) -> GSM8KSample:
        if self._pos >= len(self._order):
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._pos = 0
        sample = self._samples[self._order[self._pos]]
        self._pos += 1
        return sample

    def state_dict(self) -> dict:
        """Snapshot the RNG + position so a run can resume mid-stream."""
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "pos": self._pos,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._pos = state_dict["pos"]
