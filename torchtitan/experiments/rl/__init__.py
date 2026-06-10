# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified approach for running TorchTitan models with vLLM inference."""

# Eager imports of model wrappers and registry functions are removed from this
# package-level init to prevent import-time dependency crashes.
# Importing these wrappers directly pulls in vllm/CUDA libraries, which fail 
# on accelerator nodes that only run training (like CPU/TPU Trainer processes
# which do not have vllm or CUDA libraries installed).