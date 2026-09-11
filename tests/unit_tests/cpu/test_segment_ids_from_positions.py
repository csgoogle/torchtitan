# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fork: the blocked prefix sum in ``segment_ids_from_positions`` must equal the
plain cumsum bit for bit (it exists only because torch_tpu's eager cumsum over
a long packed sequence is two orders of magnitude slower)."""

import torch

from torchtitan.models.common.attention import segment_ids_from_positions


def _reference(positions: torch.Tensor) -> torch.Tensor:
    starts = (positions == 0).to(torch.int32)
    return (torch.cumsum(starts, dim=-1) - 1).int()


def _random_packed_positions(total: int, gen: torch.Generator) -> torch.Tensor:
    pieces, left = [], total
    while left > 0:
        n = int(torch.randint(1, 700, (1,), generator=gen))
        n = min(n, left)
        pieces.append(torch.arange(n))
        left -= n
    return torch.cat(pieces)


def test_matches_plain_cumsum_on_many_packings():
    gen = torch.Generator().manual_seed(0)
    for total in (1, 127, 128, 129, 256, 2048, 4096 + 1, 16384):
        for _ in range(5):
            positions = _random_packed_positions(total, gen)
            got = segment_ids_from_positions(positions)
            want = _reference(positions)
            assert got.dtype == want.dtype
            assert torch.equal(got, want), f"total={total}"


def test_batched_positions():
    gen = torch.Generator().manual_seed(1)
    rows = [_random_packed_positions(2048, gen) for _ in range(3)]
    positions = torch.stack(rows)
    assert torch.equal(segment_ids_from_positions(positions), _reference(positions))


def test_single_document_is_all_zero():
    positions = torch.arange(4096)
    assert segment_ids_from_positions(positions).eq(0).all()
