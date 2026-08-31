# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest

import torch
from torch import nn

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
    annotate_module_fqns,
    compute_parameter_gradients,
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _make_default_memory_policy,
    _make_full_memory_policy,
    tag_sac_policy,
)
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    remove_parameter_gradient_markers_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    _optimization_barrier,
    apply_tpu_remat_schedule,
)


class _Projection(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.to(torch.bfloat16)


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.moe = nn.Module()
        self.moe.experts = _Projection(dim)
        self.attention = _Projection(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.moe.experts(x)) + self.attention(x)


class _Epilogue(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.square()


class _ToyModel(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(dim), _Block(dim)])
        self.norm = _Epilogue()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def _nearest_parameter_fqns(node: torch.fx.Node) -> tuple[str, ...]:
    frontier = {node}
    seen: set[torch.fx.Node] = set()
    while frontier:
        fqns = {
            fqn
            for candidate in frontier
            for fqn in candidate.meta.get("custom", {}).get(
                PARAMETER_GRADIENT_FQNS_META, ()
            )
        }
        if fqns:
            return tuple(sorted(fqns))
        seen.update(frontier)
        frontier = {
            user
            for candidate in frontier
            for user in candidate.users
            if user not in seen
        }
    return ()


def _describe_barriers(gm: torch.fx.GraphModule) -> dict[int, dict[str, tuple]]:
    description = {}
    for barrier in gm.graph.nodes:
        if barrier.op != "call_function" or barrier.target is not _optimization_barrier:
            continue

        redirected_edges = []
        ordering_edges = []
        redirected_regions = set()
        outputs = {
            user.args[1]: user
            for user in barrier.users
            if user.op == "call_function" and user.target is operator.getitem
        }
        for index, value in enumerate(barrier.args):
            destinations = {
                (
                    _get_layer_id(user),
                    (
                        "recompute"
                        if user.name.endswith("_recomputed")
                        else "backward"
                        if _is_backward_node(user)
                        else "forward"
                    ),
                )
                for user in outputs[index].users
            }
            dtype = str(value.meta["val"].dtype)
            if destinations:
                redirected_regions.update(region_id for region_id, _ in destinations)
                redirected_edges.append((tuple(sorted(destinations)), dtype))
            else:
                ordering_edges.append((_nearest_parameter_fqns(value), dtype))

        assert len(redirected_regions) == 1
        description[redirected_regions.pop()] = {
            "redirected": tuple(sorted(redirected_edges)),
            "ordering": tuple(sorted(ordering_edges)),
        }
    return description


class TestTpuRematSchedule(unittest.TestCase):
    def test_traced_two_layer_schedule(self) -> None:
        for policy_name, policy_fn in (
            ("full", _make_full_memory_policy()),
            ("sac", _make_default_memory_policy(set())),
        ):
            with self.subTest(policy=policy_name):
                torch.manual_seed(0)
                model = _ToyModel()
                annotate_module_fqns(model)

                def step(x: torch.Tensor) -> list[torch.Tensor]:
                    loss = model(x).float().sum()
                    named_params = [
                        (name, parameter)
                        for name, parameter in model.named_parameters(
                            remove_duplicate=False
                        )
                        if parameter.requires_grad
                    ]
                    grads = compute_parameter_gradients(loss, named_params)
                    updated_params = [
                        parameter - 0.01 * grad
                        for (_, parameter), grad in zip(
                            named_params, grads, strict=True
                        )
                    ]
                    return [loss, *updated_params]

                x = torch.randn(2, 4, dtype=torch.bfloat16)
                traced = minimal_fx_tracer(step, module=model)(x)
                expected_outputs = [
                    output.detach().clone()
                    for output in run_traced(traced, module=model)(x)
                ]
                remove_parameter_gradient_markers_pass(
                    traced.gm, traced.example_inputs
                )
                tag_sac_policy(traced.gm, policy_fn=policy_fn)
                selective_activation_remat_pass(traced.gm)
                apply_tpu_remat_schedule(
                    traced.gm,
                    traced.example_inputs,
                    num_static_inputs=traced.num_static_inputs,
                )

                # This single topology comparison proves that every layer entry
                # redirects its saved activation and incoming cotangent through
                # one barrier; layer 0 additionally waits for both layer 1 local
                # dWeights even though an optimizer-like update consumes the
                # tagged gradients instead of returning them. The non-expert
                # path preserves the generic FSDP ordering point at the FP32
                # collective input. The validated expert specialization instead
                # orders on BF16 local dWeight, leaving its cast and collective
                # outside the barrier and available to overlap. Generalizing
                # that ordering point to all eligible gradients is a follow-up.
                self.assertEqual(
                    _describe_barriers(traced.gm),
                    {
                        0: {
                            "redirected": (
                                (((0, "backward"),), "torch.bfloat16"),
                                (((0, "recompute"),), "torch.bfloat16"),
                            ),
                            "ordering": (
                                (("layers.1.attention.weight",), "torch.float32"),
                                (("layers.1.moe.experts.weight",), "torch.bfloat16"),
                            ),
                        },
                        1: {
                            "redirected": (
                                (((1, "backward"),), "torch.bfloat16"),
                                (((1, "recompute"),), "torch.bfloat16"),
                            ),
                            "ordering": (),
                        },
                    },
                )

                actual_outputs = run_traced(traced, module=model)(x)
                for actual, expected in zip(
                    actual_outputs, expected_outputs, strict=True
                ):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
