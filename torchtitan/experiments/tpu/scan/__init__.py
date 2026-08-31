# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scan over transformer decoder layers on TPU.

Replaces a model's per-layer Python loop with a single `scan` higher-order op,
so N identical blocks lower to one `stablehlo.while` instead of N unrolled
copies: compile time and program size stop growing with depth.
"""
