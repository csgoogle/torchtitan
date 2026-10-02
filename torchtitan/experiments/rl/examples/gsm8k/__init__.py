# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

from torchtitan.experiments.rl.examples.gsm8k.data import GSM8KDataset, GSM8KSample

# The env and rubric import renderers, and the rollouter also imports monarch.
# Both are optional dependencies, so these names are loaded on first access.
# Importing the dataset (`gsm8k.data`) then works without them, e.g. on the
# TPU CI image, which installs neither.
_LAZY_EXPORTS = {
    "GSM8KEnv": "env",
    "GSM8KRollouter": "rollouter",
    "GSM8KWorker": "rollouter",
    "RewardBoxedAnswer": "rubric",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(f"{__name__}.{_LAZY_EXPORTS[name]}")
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "GSM8KDataset",
    "GSM8KEnv",
    "GSM8KRollouter",
    "GSM8KSample",
    "GSM8KWorker",
    "RewardBoxedAnswer",
]
