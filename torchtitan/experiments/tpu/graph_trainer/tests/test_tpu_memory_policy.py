# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the TPU selective memory policies on traced toy graphs."""

from types import SimpleNamespace
import unittest

from absl.testing import absltest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    annotate_module_fqns,
    compute_parameter_gradients,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _find_fsdp_unshard_save_nodes,
)
from torchtitan.experiments.graph_trainer.registry import MEMORY_POLICY_REGISTRY
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    remove_parameter_gradient_markers_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    _optimization_barrier,
    apply_tpu_remat_schedule,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_memory_policy import (
    _saved_bytes,
    full_remat_with_saves,
    MOE_ROUTE_PERMUTATION,
    REMAT_REGION,
    resolve_save_regions,
    ROUTE_PERMUTATION_CLASS,
    save_budget_bytes,
    select_route_permutations,
    select_saves,
)
from torchtitan.experiments.tpu.token_dispatcher import (
    local_reorder,
    stable_grouped_reduce,
)

_TOKENS = 8
_DIM = 8
_TOP_K = 2


def _parallelism(reshard_after_forward: str) -> SimpleNamespace:
    return SimpleNamespace(
        fsdp_reshard_after_forward=reshard_after_forward,
        pipeline_parallel_degree=1,
    )


class _DenseBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.weight.to(x.dtype))


class _DenseToy(nn.Module):
    def __init__(self, dim: int = 8, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_DenseBlock(dim) for _ in range(n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _Attention(nn.Module):
    """A projection, optionally followed by an attention-kernel stand-in."""

    def __init__(self, dim: int, core=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x @ self.weight.to(x.dtype)
        return q if self.core is None else self.core(q)


class _MoE(nn.Module):
    """Top-k routing through the real TPU local reorder and grouped reduce."""

    def __init__(self, dim: int, n_experts: int = 4) -> None:
        super().__init__()
        self.router = nn.Parameter(torch.randn(dim, n_experts))
        self.expert = nn.Parameter(torch.randn(dim, dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.softmax(x.float() @ self.router, dim=-1)
        top_scores, expert_ids = torch.topk(scores, _TOP_K, dim=-1)
        routed, token_indices, routed_scores = local_reorder(
            x, top_scores, expert_ids, _TOP_K
        )
        expert_out = (routed @ self.expert.to(x.dtype)).float()
        weighted = (expert_out * routed_scores.reshape(-1, 1)).to(x.dtype)
        return stable_grouped_reduce(weighted, token_indices, _TOP_K)


class _Block(nn.Module):
    def __init__(self, dim: int, core=None) -> None:
        super().__init__()
        self.attention = _Attention(dim, core)
        self.moe = _MoE(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        return x + self.moe(x)


class _MoEToy(nn.Module):
    def __init__(self, cores=(None, None)) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Block(_DIM, core) for core in cores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _trace_toy(model: nn.Module):
    """Trace one training step of ``model``; returns (traced, input)."""
    annotate_module_fqns(model)

    def step(x: torch.Tensor) -> list[torch.Tensor]:
        loss = model(x).float().sum()
        named_params = [
            (name, parameter)
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        ]
        grads = compute_parameter_gradients(loss, named_params)
        updated = [
            parameter - 0.01 * grad
            for (_, parameter), grad in zip(named_params, grads, strict=True)
        ]
        return [loss, *updated]

    x = torch.randn(_TOKENS, _DIM, dtype=torch.bfloat16)
    return minimal_fx_tracer(step, module=model)(x), x


def _trace_simple_fsdp_graph() -> torch.fx.GraphModule:
    """Trace a two-rank SimpleFSDP toy on a fake process group."""
    assert not dist.is_initialized()
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
    try:
        mesh = DeviceMesh("cpu", [0, 1])
        with torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True):
            model = _DenseToy()
            annotate_module_fqns(model)
            model = data_parallel(
                model,
                device_mesh=mesh,
                mode="fully_shard",
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
            )
            inputs = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)

            def step(inputs: torch.Tensor) -> list[torch.Tensor]:
                loss = model(inputs).float().sum()
                named_params = list(model.named_parameters())
                return [loss, *compute_parameter_gradients(loss, named_params)]

            traced = minimal_fx_tracer(step, module=model)(inputs)
    finally:
        dist.destroy_process_group()
    return traced.gm


class TestFullRematWithSaves(unittest.TestCase):
    def test_zero2_keeps_all_gather_outputs_saved(self) -> None:
        for reshard_after_forward, expected in (
            ("never", CheckpointPolicy.MUST_SAVE),
            ("always", CheckpointPolicy.MUST_RECOMPUTE),
        ):
            with self.subTest(reshard_after_forward=reshard_after_forward):
                gm = _trace_simple_fsdp_graph()
                unshards = _find_fsdp_unshard_save_nodes(gm)
                self.assertTrue(unshards)

                # No policy saves of its own: only the FSDP contract applies.
                full_remat_with_saves(
                    gm,
                    SimpleNamespace(parallelism=_parallelism(reshard_after_forward)),
                    save_nodes=(),
                )

                self.assertEqual(
                    {node.meta["recompute"] for node in unshards}, {expected}
                )


class TestRoutePermutationClass(unittest.TestCase):
    def test_selects_one_permutation_per_layer(self) -> None:
        torch.manual_seed(0)
        traced, _ = _trace_toy(_MoEToy())

        routes = select_route_permutations(traced.gm)

        # The hint is merged into the module FQN, not a replacement for it, so
        # each permutation still knows its layer.
        self.assertEqual([_get_layer_id(node) for node in routes], [0, 1])
        for node in routes:
            self.assertEqual(node.meta["val"].shape, (_TOKENS * _TOP_K,))
            self.assertEqual(
                node.meta["custom"][REMAT_REGION], MOE_ROUTE_PERMUTATION
            )


def _selective_config(
    save_regions: tuple[str, ...], budget_gb: float | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        tpu_config=SimpleNamespace(
            graph_trainer_save_regions=save_regions,
            graph_trainer_memory_policy_save_budget_gb=budget_gb,
        ),
        parallelism=_parallelism("always"),
    )


class TestSelectSaves(unittest.TestCase):
    def test_required_class_without_candidates_raises(self) -> None:
        traced, _ = _trace_toy(_DenseToy())

        with self.assertRaisesRegex(ValueError, "'moe.route_permutation' is required"):
            select_saves(traced.gm, [ROUTE_PERMUTATION_CLASS], budget_bytes=None)

    def test_always_saved_classes_must_fit_the_budget(self) -> None:
        torch.manual_seed(0)
        traced, _ = _trace_toy(_MoEToy())
        routes = select_route_permutations(traced.gm)
        route_bytes = sum(_saved_bytes(node) for node in routes)

        for budget_bytes in (None, route_bytes):
            self.assertEqual(
                select_saves(traced.gm, [ROUTE_PERMUTATION_CLASS], budget_bytes),
                set(routes),
            )
        with self.assertRaisesRegex(ValueError, "save budget"):
            select_saves(traced.gm, [ROUTE_PERMUTATION_CLASS], route_bytes - 1)

    def test_config_values_are_checked(self) -> None:
        self.assertEqual(
            resolve_save_regions(("moe.route_permutation",)),
            [ROUTE_PERMUTATION_CLASS],
        )
        for names, message in (
            ((), "needs graph_trainer_save_regions"),
            (("moe.nope",), "Unknown save regions"),
            (("moe.route_permutation",) * 2, "Duplicate"),
        ):
            with self.subTest(names=names):
                with self.assertRaisesRegex(ValueError, message):
                    resolve_save_regions(names)

        self.assertIsNone(save_budget_bytes(None))
        self.assertEqual(save_budget_bytes(0.5), 512 * 1024**2)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            save_budget_bytes(-1.0)


class TestTpuSelectivePolicy(unittest.TestCase):
    def _run(
        self,
        model: nn.Module,
        save_regions: tuple[str, ...],
        budget_gb: float | None = None,
    ) -> set[torch.fx.Node]:
        """Apply tpu_selective, remat and the TPU remat schedule to ``model``.

        Checks that the chosen values are saved rather than recomputed, that
        every layer still gets its entry barrier, and that the transformed
        graph computes exactly what the traced graph did. Returns the chosen
        nodes.
        """
        n_layers = len(model.layers)
        traced, x = _trace_toy(model)
        expected = [
            output.detach().clone()
            for output in run_traced(traced, module=model)(x)
        ]
        remove_parameter_gradient_markers_pass(traced.gm, traced.example_inputs)
        chosen = select_saves(
            traced.gm,
            resolve_save_regions(save_regions),
            save_budget_bytes(budget_gb),
        )

        MEMORY_POLICY_REGISTRY["tpu_selective"](
            traced.gm, config=_selective_config(save_regions, budget_gb)
        )
        for node in chosen:
            self.assertIs(node.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        selective_activation_remat_pass(traced.gm)
        apply_tpu_remat_schedule(
            traced.gm,
            traced.example_inputs,
            num_static_inputs=traced.num_static_inputs,
        )

        names = {node.name for node in traced.gm.graph.nodes}
        self.assertTrue(any(name.endswith("_recomputed") for name in names))
        self.assertFalse({f"{node.name}_recomputed" for node in chosen} & names)
        barriers = [
            node
            for node in traced.gm.graph.nodes
            if node.op == "call_function" and node.target is _optimization_barrier
        ]
        self.assertEqual(len(barriers), n_layers)
        actual = run_traced(traced, module=model)(x)
        for actual_output, expected_output in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_output, expected_output, rtol=0, atol=0
            )
        return chosen

    def test_saves_route_permutations(self) -> None:
        torch.manual_seed(0)
        chosen = self._run(_MoEToy(), ("moe.route_permutation",))

        self.assertEqual(sorted(_get_layer_id(node) for node in chosen), [0, 1])
        for node in chosen:
            self.assertEqual(
                node.meta["custom"][REMAT_REGION], MOE_ROUTE_PERMUTATION
            )


if __name__ == "__main__":
    absltest.main()
