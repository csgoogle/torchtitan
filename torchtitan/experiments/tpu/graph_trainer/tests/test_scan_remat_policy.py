"""The scan body saves its expert counts, and nothing else changes.

The counts are the one thing the body's full remat no longer redoes in
backward. Everything here is checked against the plain budget-0.0 partition
it replaces: the gradients bit for bit, and the backward graph for what it
stopped (and did not stop) recomputing.
"""

import unittest
from unittest import mock

from absl.testing import absltest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch._functorch import config as functorch_config, partitioners
from torch._higher_order_ops.scan import scan
from torch.fx.experimental.proxy_tensor import make_fx

# Imported for the same reason as in `test_layer_stack`: production scans run
# with it installed, so these do too.
from torchtitan.experiments.tpu import scan_backward_patch  # noqa: F401  pylint: disable=unused-import,g-bad-import-order
from torchtitan.experiments.tpu.graph_trainer.scan_remat_policy import (
    count_collectives,
    scan_body_remat_window,
)

SEED = 1234
N_LAYERS = 3
TOKENS = 16
DIM = 8
EXPERTS = 4
TOP_K = 2


def _counts(scores: torch.Tensor) -> torch.Tensor:
    """The batch-wise aux loss's global counts, as `moe.py` computes them."""
    ids = scores.topk(TOP_K, dim=-1).indices
    counts = torch.histc(
        ids.reshape(-1).float(), bins=EXPERTS, min=0, max=EXPERTS
    ).to(scores.dtype)
    return funcol.all_reduce(counts, "sum", dist.group.WORLD)


def _body(h: torch.Tensor, w: torch.Tensor, *, with_counts: bool):
    """A router plus an aux-loss-like term, so the counts reach the gradient."""
    scores = torch.softmax(h @ w, dim=-1)
    out = torch.tanh(h + scores @ w.T)
    if with_counts:
        counts = _counts(scores)
        aux = (counts * scores.sum(dim=0)).sum() / counts.sum()
        out = out + aux
    return out, out.new_zeros(())


def _run(window, *, with_counts: bool = True):
    """Forward and backward one scan; returns grads and the body's partitions."""
    torch._dynamo.reset()
    torch.manual_seed(SEED)
    h = torch.randn(TOKENS, DIM, requires_grad=True)
    w = torch.randn(N_LAYERS, DIM, EXPERTS, requires_grad=True)

    captured = []
    partition = partitioners.min_cut_rematerialization_partition

    def capture(joint, inputs, **kwargs):
        forward, backward = partition(joint, inputs, **kwargs)
        captured.append((functorch_config.activation_memory_budget, backward))
        return forward, backward

    def body(carry, x):
        return _body(carry, x, with_counts=with_counts)

    with mock.patch.object(
        partitioners, "min_cut_rematerialization_partition", capture
    ):
        with window():
            carry, _ = scan(body, h, w)
    carry.square().sum().backward()
    return (carry.detach(), h.grad, w.grad), captured


def _legacy_window():
    return functorch_config.patch(activation_memory_budget=0.0)


def _ops(graph_module) -> list[str]:
    return [str(n.target) for n in graph_module.graph.nodes if n.op == "call_function"]


class TestCountCollectives(unittest.TestCase):
    """Only a collective whose operand *is* a count is picked up."""

    def setUp(self):
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12391",
                world_size=1,
                rank=0,
            )

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_finds_the_count_all_reduce_and_not_the_others(self):
        def f(scores, weight):
            counts = funcol.wait_tensor(_counts(scores))
            # Collectives on values that are not counts: a weight, and a count
            # once it has been mixed with something else.
            gathered = funcol.wait_tensor(
                funcol.all_reduce(weight, "sum", dist.group.WORLD)
            )
            mixed = funcol.wait_tensor(
                funcol.all_reduce(counts * weight.sum(), "sum", dist.group.WORLD)
            )
            return counts, gathered, mixed

        graph = make_fx(f)(
            torch.randn(TOKENS, EXPERTS).softmax(-1), torch.randn(EXPERTS)
        ).graph
        found = count_collectives(graph)

        self.assertTrue(found, "the count all-reduce was not found")
        # The count's own all_reduce and its wait(s) -- tracing may emit more
        # than one wait for it -- and nothing for the weight or the mix.
        self.assertEqual(
            {str(n.target) for n in found},
            {"_c10d_functional.all_reduce.default", "_c10d_functional.wait_tensor.default"},
        )
        all_reduces = [
            n for n in found if n.target is torch.ops._c10d_functional.all_reduce.default
        ]
        self.assertEqual(len(all_reduces), 1, found)
        (all_reduce,) = all_reduces
        self.assertIn(
            all_reduce.args[0].target,
            (torch.ops.aten._to_copy.default, torch.ops.aten.histc.default),
        )


class TestScanBodyRematWindow(unittest.TestCase):

    def setUp(self):
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12392",
                world_size=1,
                rank=0,
            )

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_gradients_are_bitwise_identical(self):
        """The whole point: the counts move, the math does not."""
        want, _ = _run(_legacy_window)
        got, _ = _run(scan_body_remat_window)
        for name, g, w in zip(("out", "h.grad", "w.grad"), got, want):
            torch.testing.assert_close(g, w, rtol=0, atol=0, msg=name)

    def test_backward_no_longer_recomputes_the_counts(self):
        _, legacy = _run(_legacy_window)
        _, captured = _run(scan_body_remat_window)
        self.assertTrue(captured, "the scan body was never partitioned")

        # Sanity: before, backward redid the histogram and its all-reduce.
        legacy_ops = _ops(legacy[0][1])
        self.assertIn("aten.histc.default", legacy_ops)
        self.assertIn("_c10d_functional.all_reduce.default", legacy_ops)

        budget, backward = captured[0]
        self.assertEqual(budget, 1.0)
        ops = _ops(backward)
        self.assertNotIn("aten.histc.default", ops)
        self.assertNotIn("_c10d_functional.all_reduce.default", ops)
        # ... while the rest of the body is still recomputed.
        self.assertIn("aten.mm.default", ops)
        self.assertIn("aten.tanh.default", ops)

    def test_a_body_without_counts_is_partitioned_as_before(self):
        """No counts, no tags: the old budget-0.0 path, untouched."""
        want, legacy = _run(_legacy_window, with_counts=False)
        got, captured = _run(scan_body_remat_window, with_counts=False)
        self.assertEqual(captured[0][0], 0.0)
        self.assertEqual(_ops(captured[0][1]), _ops(legacy[0][1]))
        for name, g, w in zip(("out", "h.grad", "w.grad"), got, want):
            torch.testing.assert_close(g, w, rtol=0, atol=0, msg=name)

    def test_the_window_closes(self):
        default_budget = functorch_config.activation_memory_budget
        partition = partitioners.min_cut_rematerialization_partition
        _run(scan_body_remat_window)
        self.assertEqual(functorch_config.activation_memory_budget, default_budget)
        self.assertIs(partitioners.min_cut_rematerialization_partition, partition)


if __name__ == "__main__":
    absltest.main()
