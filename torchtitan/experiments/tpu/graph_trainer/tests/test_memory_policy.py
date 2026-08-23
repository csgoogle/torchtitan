# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
from types import SimpleNamespace
import unittest

import torch
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.graph_trainer.registry import MEMORY_POLICY_REGISTRY
from torchtitan.experiments.tpu.graph_trainer.memory_policy import (
    _is_tpu_splash_attention_forward,
    _select_tpu_splash_attention_save_nodes,
    _tpu_splash_memory_policy_pass,
)


@torch.library.custom_op(
    "pallas::splash_attn_fwd_9876543210abcdef", mutates_args=()
)
def _fake_splash_attn_fwd_test(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return x, x[..., 0].float()


_FAKE_SPLASH_ATTN_FWD = (
    torch.ops.pallas.splash_attn_fwd_9876543210abcdef.default
)


def _fake_splash_attn_bwd_test(
    attention_output: torch.Tensor, logsumexp: torch.Tensor
) -> torch.Tensor:
    del logsumexp
    return attention_output


class TestTpuSplashMemoryPolicy(unittest.TestCase):
    def test_policy_is_registered(self):
        self.assertIs(
            MEMORY_POLICY_REGISTRY["tpu_splash"],
            _tpu_splash_memory_policy_pass,
        )

    def test_saves_attention_outputs_within_budget(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        splash0 = graph.call_function(_FAKE_SPLASH_ATTN_FWD, args=(x,))
        splash0.meta["val"] = (
            torch.empty(2, 4, 8, 16, dtype=torch.bfloat16, device="meta"),
            torch.empty(2, 4, 8, dtype=torch.float32, device="meta"),
        )
        output0 = graph.call_function(operator.getitem, args=(splash0, 0))
        output0.meta["val"] = splash0.meta["val"][0]
        logsumexp0 = graph.call_function(operator.getitem, args=(splash0, 1))
        logsumexp0.meta["val"] = splash0.meta["val"][1]
        splash1 = graph.call_function(_FAKE_SPLASH_ATTN_FWD, args=(output0,))
        splash1.meta["val"] = splash0.meta["val"]
        output1 = graph.call_function(operator.getitem, args=(splash1, 0))
        output1.meta["val"] = splash1.meta["val"][0]
        logsumexp1 = graph.call_function(operator.getitem, args=(splash1, 1))
        logsumexp1.meta["val"] = splash1.meta["val"][1]
        backward1 = graph.call_function(
            _fake_splash_attn_bwd_test, args=(output1, logsumexp1)
        )
        backward1.meta["val"] = output1.meta["val"]
        backward1.meta["autograd_backward"] = True
        backward0 = graph.call_function(
            _fake_splash_attn_bwd_test, args=(output0, logsumexp0)
        )
        backward0.meta["val"] = output0.meta["val"]
        backward0.meta["autograd_backward"] = True
        graph.output((backward0, backward1))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        tuple_bytes = sum(
            int(output.numel()) * output.element_size()
            for output in splash0.meta["val"]
        )
        splash_save_nodes, splash_bytes, candidate_count = (
            _select_tpu_splash_attention_save_nodes(
                gm,
                budget_bytes=tuple_bytes,
            )
        )
        self.assertEqual(candidate_count, 2)
        self.assertEqual(splash_bytes, tuple_bytes)
        self.assertEqual(splash_save_nodes, {splash1})

        with self.assertRaisesRegex(ValueError, "No Splash output tuple fits"):
            _tpu_splash_memory_policy_pass(
                gm,
                config=SimpleNamespace(
                    tpu_config=SimpleNamespace(
                        graph_trainer_memory_policy_save_budget_gb=0.0
                    )
                ),
            )
        _tpu_splash_memory_policy_pass(
            gm,
            config=SimpleNamespace(
                tpu_config=SimpleNamespace(
                    graph_trainer_memory_policy_save_budget_gb=(
                        tuple_bytes + 1
                    )
                    / (1024**3)
                )
            ),
        )
        self.assertIs(splash0.meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE)
        self.assertIs(splash1.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertIs(output1.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertIs(logsumexp1.meta["recompute"], CheckpointPolicy.MUST_SAVE)

        gm = selective_activation_remat_pass(gm)
        recomputed_names = {
            node.name
            for node in gm.graph.nodes
            if node.name.endswith("_recomputed")
        }
        self.assertIn(f"{splash0.name}_recomputed", recomputed_names)
        self.assertNotIn(f"{splash1.name}_recomputed", recomputed_names)
        remat_backward1 = next(
            node for node in gm.graph.nodes if node.name == backward1.name
        )
        self.assertIs(remat_backward1.args[0], output1)
        self.assertIs(remat_backward1.args[1], logsumexp1)

        non_splash = SimpleNamespace(
            target="other.splash_attn_fwd_9876543210abcdef.default",
            meta={"val": splash0.meta["val"]},
        )
        self.assertFalse(_is_tpu_splash_attention_forward(non_splash))

    def test_requires_splash_forward(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(2, 4, device="meta")
        graph.output(x)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        with self.assertRaisesRegex(ValueError, "found no Splash attention"):
            _tpu_splash_memory_policy_pass(
                gm,
                config=SimpleNamespace(
                    tpu_config=SimpleNamespace(
                        graph_trainer_memory_policy_save_budget_gb=1.0
                    )
                ),
            )


if __name__ == "__main__":
    unittest.main()
