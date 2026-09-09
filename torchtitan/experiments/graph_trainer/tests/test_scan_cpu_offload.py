# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The scan offload must pick the layer stack and place the ops in the bodies.

Every case here is one that runs to completion and reports a plausible loss
while offloading the wrong tensor, offloading it in the wrong place, or doing
nothing at all. None of them raise.

The fixture mirrors the two asymmetries the real joint graph has: the forward
hands the stack out as a ``ys`` output, and the backward takes it in
``additional_inputs`` -- whole, indexed in-body by the loop counter it carries
in ``xs``.
"""

import operator
import unittest

import torch

from torchtitan.experiments.graph_trainer.tpu_cpu_offload import (
    offload_scan_activations,
)

_SCAN = torch.ops.higher_order.scan
_AO = (
    torch.ops.ao.offload.default,
    torch.ops.ao.reload.default,
    torch.ops.ao.wait_tensor.default,
)


def _ao_targets(graph):
    return [n.target for n in graph.nodes if n.target in _AO]


def _body_of(gm, scan_node):
    return getattr(gm, scan_node.args[0].target)


class ScanOffloadTest(unittest.TestCase):
    """Builds the fwd/bwd scan pair the joint-graph tracer produces."""

    def setUp(self):
        self.root = torch.nn.Module()
        self.graph = torch.fx.Graph()
        self.x = self.graph.placeholder("x")
        self.x.meta["val"] = torch.empty(8, 8)
        self._n_bodies = 0

    def _gm(self):
        return torch.fx.GraphModule(self.root, self.graph)

    def _fwd(self, name, shape, arg=None):
        """A forward tensor produced by a real (non-view) op."""
        arg = self.x if arg is None else arg
        node = self.graph.call_function(torch.ops.aten.mm.default, args=(arg, arg))
        node.name = name
        node.meta["val"] = torch.empty(shape)
        node.meta["autograd_backward"] = False
        return node

    def _body(self, n_in, out_shapes, stack_shape=None, slice_stack=True):
        """A loop body: ``n_in`` placeholders in, one mm per output out.

        ``stack_shape`` appends one more placeholder holding the whole stack,
        which is how the backward body receives it. ``slice_stack=False`` reads
        it whole instead -- the shape the offload has to refuse.
        """
        g = torch.fx.Graph()
        params = []
        for i in range(n_in):
            p = g.placeholder(f"p{i}")
            p.meta["val"] = torch.empty(4, 4)
            params.append(p)

        read = None
        if stack_shape is not None:
            stack = g.placeholder("stack")
            stack.meta["val"] = torch.empty(stack_shape)
            if slice_stack:
                read = g.call_function(
                    torch.ops.aten.index_select.default, args=(stack, 0, params[0])
                )
                read.meta["val"] = torch.empty(stack_shape[1:])
            else:
                read = g.call_function(torch.ops.aten.mm.default, args=(stack, stack))
                read.meta["val"] = torch.empty(stack_shape)

        outs = []
        for shape in out_shapes:
            src = read if read is not None else params[0]
            o = g.call_function(torch.ops.aten.mm.default, args=(src, src))
            o.meta["val"] = torch.empty(shape)
            read = None
            outs.append(o)
        g.output(tuple(outs))

        body = torch.fx.GraphModule(torch.nn.Module(), g)
        self._n_bodies += 1
        name = f"scan_combine_graph_{self._n_bodies}"
        self.root.add_module(name, body)
        return name

    def _scan(
        self,
        *,
        init,
        xs,
        out_shapes,
        backward,
        additional=(),
        stack_shape=None,
        slice_stack=True,
    ):
        n_plain = len(init) + len(xs) + len(additional) - (stack_shape is not None)
        name = self._body(n_plain, out_shapes, stack_shape, slice_stack)
        fn = self.graph.get_attr(name)
        node = self.graph.call_function(
            _SCAN, args=(fn, init, xs, tuple(additional))
        )
        node.meta["autograd_backward"] = backward
        return node

    def _fwd_scan_producing(self, stack_shape):
        """A forward scan whose second output is the stacked layer input."""
        carry = self._fwd("carry_init", (4, 4))
        xs = self._fwd("xs_in", (6, 4, 4))
        scan = self._scan(
            init=[carry], xs=[xs], out_shapes=[(4, 4), stack_shape], backward=False
        )
        stack = self.graph.call_function(operator.getitem, args=(scan, 1))
        stack.name = "layer_stack"
        stack.meta["val"] = torch.empty(stack_shape)
        stack.meta["autograd_backward"] = False
        return scan, stack

    def _bwd_scan_reading(self, stack, stack_shape, **kwargs):
        """A backward scan taking the stack in ``additional_inputs``."""
        counter = self._fwd("arange", (6,))
        grad = self._fwd("grad_init", (4, 4))
        return self._scan(
            init=[grad],
            xs=[counter],
            additional=[stack],
            stack_shape=stack_shape,
            out_shapes=[(4, 4)],
            backward=True,
            **kwargs,
        )

    def test_the_stack_wins_over_a_larger_tensor_the_backward_also_reads(self):
        # The bug the selector exists for. At debugmodel scale the loss logits
        # are larger than the layer stack -- vocabulary beats depth -- and a
        # size-only rule ships them to the host, where they are reloaded
        # immediately for a gradient that never needed to leave.
        logits = self._fwd("log_softmax", (4096, 2048))
        fwd, stack = self._fwd_scan_producing((6, 2, 512, 256))
        bwd = self._bwd_scan_reading(stack, (6, 2, 512, 256))
        tail = self.graph.call_function(
            torch.ops.aten.mm.default, args=(logits, logits)
        )
        tail.meta["autograd_backward"] = True
        self.graph.output((bwd, tail))

        gm = self._gm()
        self.assertGreater(logits.meta["val"].nelement(), stack.meta["val"].nelement())
        offload_scan_activations(gm)

        # The offload is on the stack's body output, and the logits are untouched.
        self.assertIn(
            torch.ops.ao.offload.default, _ao_targets(_body_of(gm, fwd).graph)
        )
        self.assertEqual(_ao_targets(gm.graph), [])

    def test_the_ops_land_inside_the_bodies_not_around_the_loop(self):
        # Placing the reload before the first backward consumer puts it before
        # the *loop*, so every slice lands on device at once and the program
        # grows by the size of the stack instead of shrinking by it.
        fwd, stack = self._fwd_scan_producing((6, 2, 512, 256))
        bwd = self._bwd_scan_reading(stack, (6, 2, 512, 256))
        self.graph.output(bwd)

        gm = self._gm()
        offload_scan_activations(gm)

        self.assertEqual(_ao_targets(gm.graph), [])
        self.assertIn(
            torch.ops.ao.offload.default, _ao_targets(_body_of(gm, fwd).graph)
        )
        self.assertIn(
            torch.ops.ao.reload.default, _ao_targets(_body_of(gm, bwd).graph)
        )

    def test_the_reload_is_on_the_slice_and_not_on_the_whole_stack(self):
        # Reloading the placeholder is legal, compiles, and pulls all n_layers
        # slices back on every iteration -- worse than not offloading at all.
        fwd, stack = self._fwd_scan_producing((6, 2, 512, 256))
        bwd = self._bwd_scan_reading(stack, (6, 2, 512, 256))
        self.graph.output(bwd)

        gm = self._gm()
        offload_scan_activations(gm)

        body = _body_of(gm, bwd).graph
        reload = next(n for n in body.nodes if n.target is torch.ops.ao.reload.default)
        self.assertEqual(reload.args[0].target, torch.ops.aten.index_select.default)

    def test_a_body_that_reads_the_stack_whole_is_left_on_device(self):
        # No slicing means no per-iteration win to have, so the pass declines
        # rather than shipping the stack back and forth for nothing.
        fwd, stack = self._fwd_scan_producing((6, 2, 512, 256))
        bwd = self._bwd_scan_reading(stack, (6, 2, 512, 256), slice_stack=False)
        self.graph.output(bwd)

        gm = self._gm()
        with self.assertLogs(level="WARNING") as logged:
            offload_scan_activations(gm)
        self.assertEqual(_ao_targets(_body_of(gm, fwd).graph), [])
        self.assertEqual(_ao_targets(_body_of(gm, bwd).graph), [])
        self.assertIn("whole", "\n".join(logged.output))

    def test_the_stack_is_found_through_the_view_that_reaches_the_scan(self):
        # What the backward scan is handed is an alias of the forward scan's
        # stacked output, so a users-only check finds nothing and the run
        # silently offloads nothing.
        fwd, stack = self._fwd_scan_producing((6, 2, 512, 256))
        view = self.graph.call_function(torch.ops.aten.slice.Tensor, args=(stack, 0))
        view.meta["val"] = stack.meta["val"]
        view.meta["autograd_backward"] = False
        bwd = self._bwd_scan_reading(view, (6, 2, 512, 256))
        self.graph.output(bwd)

        gm = self._gm()
        offload_scan_activations(gm)
        self.assertIn(
            torch.ops.ao.reload.default, _ao_targets(_body_of(gm, bwd).graph)
        )

    def test_a_run_with_no_backward_scan_offloads_nothing_and_says_so(self):
        # Asking for this policy without --tpu-config.use-scan-layers. The run
        # is correct and uses exactly the memory it would have without the
        # policy, so the warning is the only signal the user gets.
        big = self._fwd("log_softmax", (4096, 2048))
        tail = self.graph.call_function(torch.ops.aten.mm.default, args=(big, big))
        tail.meta["autograd_backward"] = True
        self.graph.output(tail)

        gm = self._gm()
        with self.assertLogs(level="WARNING") as logged:
            offload_scan_activations(gm)
        self.assertEqual(_ao_targets(gm.graph), [])
        self.assertIn("use-scan-layers", "\n".join(logged.output))


if __name__ == "__main__":
    unittest.main()
