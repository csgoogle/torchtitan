# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PolicyTrainer.push_model_state_dict publishes a tied weight once.

The trainer actor module imports monarch and torchstore. Both are optional
dependencies, so each test skips on its own when either is missing (the TPU CI
image installs neither).
"""

import asyncio
import inspect
from unittest.mock import AsyncMock, patch

import pytest
import torch

from torchtitan.models.qwen3 import qwen3_configs


def _trainer_module():
    # The trainer actor module imports these optional packages.
    for name in ("monarch", "torchstore"):
        pytest.importorskip(name)
    from torchtitan.experiments.rl.actors import trainer

    return trainer


class _TinyDecoder(torch.nn.Module):
    def __init__(self, *, tie: bool) -> None:
        super().__init__()
        self.tok_embeddings = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        if tie:
            self.tok_embeddings.weight = self.lm_head.weight


def _published_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Run push_model_state_dict and return the state dict handed to TorchStore."""

    policy_trainer = _trainer_module().PolicyTrainer
    # The undecorated coroutine behind the Monarch endpoint (unwrapping the
    # concurrent-endpoint and trace-span wrappers).
    push_model_state_dict = inspect.unwrap(
        policy_trainer.__dict__["push_model_state_dict"]._method
    )

    async def run() -> dict[str, torch.Tensor]:
        trainer = object.__new__(policy_trainer)
        trainer.model = model
        trainer._transfer_dtype = torch.bfloat16
        with patch(
            "torchtitan.experiments.rl.actors.trainer.ts.put_state_dict",
            new=AsyncMock(),
        ) as put_state_dict:
            await push_model_state_dict(trainer)
        put_state_dict.assert_awaited_once()
        return put_state_dict.await_args.args[0]

    return asyncio.run(run())


def test_tied_alias_fqns_lists_only_the_repeated_fqn() -> None:
    tied_alias_fqns = _trainer_module()._tied_alias_fqns
    tied = _TinyDecoder(tie=True)
    # state_dict() lists the tied weight under both FQNs; only the second is an alias.
    assert set(tied.state_dict()) == {"tok_embeddings.weight", "lm_head.weight"}
    assert tied_alias_fqns(tied) == {"lm_head.weight"}
    assert tied_alias_fqns(_TinyDecoder(tie=False)) == set()


def test_tied_alias_fqns_on_a_tied_qwen3_decoder() -> None:
    # The RL recipes' Qwen3 0.6B/1.7B/4B flavors tie the embedding and lm_head,
    # as the debugmodel flavor does. The embedding is registered first, so the
    # lm_head FQN is the alias that is not published.
    tied_alias_fqns = _trainer_module()._tied_alias_fqns
    build_config, max_context_length = qwen3_configs["debugmodel"]
    model = build_config(attn_backend="sdpa", seq_len=max_context_length).build()
    assert model.tok_embeddings.weight is model.lm_head.weight
    assert tied_alias_fqns(model) == {"lm_head.weight"}


def test_push_model_state_dict_publishes_a_tied_weight_once() -> None:
    trainer_model = _TinyDecoder(tie=True)
    published = _published_state_dict(trainer_model)

    assert set(published) == {"tok_embeddings.weight"}
    expected = trainer_model.lm_head.weight.detach().to(torch.bfloat16)
    torch.testing.assert_close(published["tok_embeddings.weight"], expected)

    # Receiver side, as the generator does it: fill only the published keys of a
    # freshly built (tied) model's state dict in place, then load it non-strictly.
    generator_model = _TinyDecoder(tie=True).to(torch.bfloat16)
    generator_sd = generator_model.state_dict()
    for name, tensor in published.items():
        generator_sd[name].copy_(tensor)
    generator_model.load_state_dict(generator_sd, strict=False)
    torch.testing.assert_close(generator_model.tok_embeddings.weight.detach(), expected)
    torch.testing.assert_close(generator_model.lm_head.weight.detach(), expected)


def test_push_model_state_dict_publishes_every_untied_weight() -> None:
    published = _published_state_dict(_TinyDecoder(tie=False))
    assert set(published) == {"tok_embeddings.weight", "lm_head.weight"}
