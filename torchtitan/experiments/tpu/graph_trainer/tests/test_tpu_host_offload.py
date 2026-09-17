"""What the scan offload does to a forward/backward scan pair.

`_pair()` builds the graph the joint-graph tracer produces for scanned layers,
and every test is that graph with one thing changed:

    stack = getitem(fwd_scan, 1)     # [6, 2, 512, 256], one row saved per layer
    bwd_scan(init=[grad], xs=[counter], additional_inputs=[stack])

The forward loop stacks one save per iteration. The backward loop receives that
stack *whole* -- it is an `additional_input`, not an `xs` -- and its body indexes
out the row it needs with the counter. So the offload belongs on the forward
body's output, and the reload on the backward body's index.

The working case is also asserted on a real llama3 run, in
test_tpu_host_offload_distributed.py. The cases where the pass cannot offload
are here because no real tracer produces them.
"""

import operator

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu.graph_trainer.tpu_host_offload import (
    offload_scan_activations,
)

_SCAN = torch.ops.higher_order.scan
_OFFLOAD = torch.ops.ao.offload.default
_RELOAD = torch.ops.ao.reload.default
_AO = (_OFFLOAD, _RELOAD, torch.ops.ao.wait_tensor.default)

_STACK = (6, 2, 512, 256)  # 6 layers, one [2, 512, 256] save each
_ROW = _STACK[1:]


def _ao_targets(graph):
    return [n.target for n in graph.nodes if n.target in _AO]


def _body_of(gm, scan_node):
    return getattr(gm, scan_node.args[0].target)


def _mm(graph, src, shape, *, name=None, bwd=False):
    """A tensor of ``shape`` produced by a real (non-view) op."""
    node = graph.call_function(torch.ops.aten.mm.default, args=(src, src))
    if name:
        node.name = name
    node.meta["val"] = torch.empty(shape)
    node.meta["autograd_backward"] = bwd
    return node


def _fwd_body():
    """``(carry, x) -> (carry_out, save)``. The save is what gets offloaded."""
    g = torch.fx.Graph()
    for name in ("carry", "x"):
        g.placeholder(name).meta["val"] = torch.empty(4, 4)
    carry = next(iter(g.nodes))
    g.output((_mm(g, carry, (4, 4)), _mm(g, carry, _ROW)))
    return g


def _bwd_body(reads):
    """``(grad, i, stack) -> (grad_out,)``, reading one row or the whole stack."""
    g = torch.fx.Graph()
    g.placeholder("grad").meta["val"] = torch.empty(4, 4)
    i = g.placeholder("i")
    i.meta["val"] = torch.empty(())
    stack = g.placeholder("stack")
    stack.meta["val"] = torch.empty(_STACK)

    if reads == "row":
        read = g.call_function(
            torch.ops.aten.index_select.default, args=(stack, 0, i)
        )
        read.meta["val"] = torch.empty(_ROW)
    else:
        read = _mm(g, stack, _STACK)
    g.output((_mm(g, read, (4, 4)),))
    return g


class ScanOffloadTest(absltest.TestCase):

    def setUp(self):
        super().setUp()
        self.root = torch.nn.Module()
        self.graph = torch.fx.Graph()
        self.x = self.graph.placeholder("x")
        self.x.meta["val"] = torch.empty(8, 8)

    def _gm(self):
        return torch.fx.GraphModule(self.root, self.graph)

    def _attach(self, body):
        """Register a loop body the way ``trace_scan`` does, as a submodule."""
        name = f"scan_combine_graph_{len(list(self.root.children())) + 1}"
        self.root.add_module(name, torch.fx.GraphModule(torch.nn.Module(), body))
        return self.graph.get_attr(name)

    def _pair(self, reads="row"):
        """The forward scan, the stack it produces, and the backward scan."""
        g = self.graph
        carry = _mm(g, self.x, (4, 4), name="carry_init")
        xs = _mm(g, self.x, (6, 4, 4), name="xs_in")
        fwd = g.call_function(
            _SCAN, args=(self._attach(_fwd_body()), [carry], [xs], ())
        )
        fwd.meta["autograd_backward"] = False

        stack = g.call_function(operator.getitem, args=(fwd, 1))
        stack.name = "layer_stack"
        stack.meta["val"] = torch.empty(_STACK)
        stack.meta["autograd_backward"] = False

        counter = _mm(g, self.x, (6,), name="arange")
        grad = _mm(g, self.x, (4, 4), name="grad_init")
        bwd = g.call_function(
            _SCAN,
            args=(self._attach(_bwd_body(reads)), [grad], [counter], (stack,)),
        )
        bwd.meta["autograd_backward"] = True
        return fwd, stack, bwd

    def test_the_offload_goes_in_the_forward_body_and_the_reload_in_the_backward(
        self,
    ):
        fwd, _, bwd = self._pair()
        self.graph.output(bwd)

        gm = self._gm()
        offload_scan_activations(gm)

        fwd_body = _body_of(gm, fwd).graph
        bwd_body = _body_of(gm, bwd).graph
        self.assertIn(_OFFLOAD, _ao_targets(fwd_body))
        self.assertIn(_RELOAD, _ao_targets(bwd_body))
        # Outside the bodies means one transfer of the whole stack instead of
        # n_layers transfers of one row, which costs memory rather than saves.
        self.assertEqual(_ao_targets(gm.graph), [])
        # On the row, not on the placeholder -- the placeholder is all 6 rows.
        reload_node = next(n for n in bwd_body.nodes if n.target is _RELOAD)
        self.assertIs(
            reload_node.args[0].target, torch.ops.aten.index_select.default
        )

    def test_a_dead_reader_of_the_stack_does_not_block_the_offload(self):
        # Tracing a scan slices element 0 off every operand to build the body,
        # and those slices stay in the graph with nothing reading them. A real
        # llama3 graph has ten. Counting them as readers turns the pass off.
        fwd, stack, bwd = self._pair()
        dead = self.graph.call_function(
            torch.ops.aten.select_copy.int, args=(stack, 0, 0)
        )
        dead.meta["val"] = torch.empty(_ROW)
        self.graph.output(bwd)

        gm = self._gm()
        offload_scan_activations(gm)

        self.assertIn(_OFFLOAD, _ao_targets(_body_of(gm, fwd).graph))
        self.assertIn(_RELOAD, _ao_targets(_body_of(gm, bwd).graph))

    def test_a_live_second_reader_leaves_the_stack_on_device(self):
        # Offloading makes the stack host-resident and does not tell its
        # readers, so a second one would silently read host memory.
        fwd, stack, bwd = self._pair()
        self.graph.output((bwd, _mm(self.graph, stack, _STACK)))

        gm = self._gm()
        with self.assertLogs(level="WARNING") as logged:
            with self.assertRaises(RuntimeError):
                offload_scan_activations(gm)
        self.assertEqual(_ao_targets(_body_of(gm, fwd).graph), [])
        self.assertEqual(_ao_targets(_body_of(gm, bwd).graph), [])
        # The warning names the reader; the raise is because that leaves the
        # policy with nothing it can move.
        self.assertIn("other than the backward scan", "\n".join(logged.output))

    def test_a_body_that_reads_the_stack_whole_is_left_on_device(self):
        # No row indexing means no per-iteration win to have, so the pass
        # declines rather than shipping the stack back and forth for nothing.
        fwd, _, bwd = self._pair(reads="whole")
        self.graph.output(bwd)

        gm = self._gm()
        with self.assertLogs(level="WARNING") as logged:
            offload_scan_activations(gm)
        self.assertEqual(_ao_targets(_body_of(gm, fwd).graph), [])
        self.assertEqual(_ao_targets(_body_of(gm, bwd).graph), [])
        self.assertIn("whole", "\n".join(logged.output))

    def test_a_graph_with_no_scan_at_all_is_an_error_not_a_quiet_no_op(self):
        # The policy without --use-scan-layers. Offloading nothing is a correct
        # run at the memory of no policy at all, so it has to be loud.
        big = _mm(self.graph, self.x, (4096, 2048), name="log_softmax")
        self.graph.output(_mm(self.graph, big, (4096, 2048), bwd=True))

        gm = self._gm()
        with self.assertRaisesRegex(RuntimeError, "use-scan-layers"):
            offload_scan_activations(gm)


if __name__ == "__main__":
    absltest.main()
