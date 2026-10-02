# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.models.common.attention import create_sdpa_document_mask
from torchtitan.models.qwen3 import qwen3_configs


def _sdpa_debugmodel():
    build_config, max_context_length = qwen3_configs["debugmodel"]
    return build_config(attn_backend="sdpa", seq_len=max_context_length)


def test_mask_is_same_document_and_causal() -> None:
    # The folded [T] token stream: documents of lengths [3, 2, 4, 5, 4].
    positions = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3])
    mask = create_sdpa_document_mask(positions)
    num_tokens = positions.numel()
    assert mask.shape == (1, 1, num_tokens, num_tokens)
    assert mask.dtype == torch.bool

    doc = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4])
    for q in range(num_tokens):
        for k in range(num_tokens):
            expected = bool(doc[q] == doc[k]) and k <= q
            assert bool(mask[0, 0, q, k]) == expected, (q, k)

    # A legacy [B, L] batch gets one [L, L] mask per row.
    rows = positions.view(2, 9)
    row_mask = create_sdpa_document_mask(rows)
    assert row_mask.shape == (2, 1, 9, 9)
    assert torch.equal(row_mask[0, 0], mask[0, 0, :9, :9])


def test_packed_document_matches_the_document_alone() -> None:
    torch.manual_seed(0)
    config = _sdpa_debugmodel()
    model = config.build()
    model.init_states()
    model.eval()

    first = torch.randint(0, config.vocab_size, (6,))
    second = torch.randint(0, config.vocab_size, (10,))
    tokens = torch.cat([first, second])
    positions = torch.cat([torch.arange(6), torch.arange(10)])
    alone_positions = torch.arange(10)

    with torch.no_grad():
        packed = model(
            tokens,
            positions=positions,
            attention_masks=model.get_attention_masks(positions),
        )
        alone = model(
            second,
            positions=alone_positions,
            attention_masks=model.get_attention_masks(alone_positions),
        )
        alone_causal = model(second, positions=alone_positions)

    # A single document: the dense mask equals plain causal attention.
    torch.testing.assert_close(alone, alone_causal, atol=1e-5, rtol=1e-5)
    # The packed second document must not see the first one.
    torch.testing.assert_close(packed[6:], alone, atol=1e-5, rtol=1e-5)


def test_preprocess_inputs_keeps_sdpa_maskless() -> None:
    # The pretraining Trainer goes through Decoder.preprocess_inputs, which
    # builds masks only for Flex and Varlen. With SDPA it must keep plain
    # causal attention: no O(T^2) dense mask, nothing for CP to shard.
    from types import SimpleNamespace

    from torchtitan.config import ParallelismConfig

    model = _sdpa_debugmodel().build()
    positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
    tokens = torch.zeros_like(positions)
    _, _, batch = model.preprocess_inputs(
        {"input": tokens, "labels": tokens, "positions": positions},
        parallel_dims=SimpleNamespace(cp_enabled=False),
        parallelism=ParallelismConfig(),
    )
    assert "attention_masks" not in batch
    assert batch["positions"] is positions
