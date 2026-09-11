# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib.metadata import version

# Import torch before anything else in the package. On TPU hosts torch's
# device-backend autoload imports torch_tpu, whose loader pulls in dynamo and
# FSDP, and FSDP now imports spmd_types. Many modules in this tree import
# spmd_types before torch (isort order); if spmd_types is the first thing that
# imports torch, its own `import torch` re-enters spmd_types while it is only
# partially initialised and torch fails with "Failed to load the backend
# extension: torch_tpu". Importing torch here makes the order irrelevant.
import torch  # noqa: F401  # isort: skip

try:
    __version__ = version("torchtitan")
except Exception:
    __version__ = "0.0.0+unknown"
