from __future__ import annotations

import operator
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.experiments.tpu.graph_trainer.passes.elide_paired_dweight_casts import (
    apply_elide_paired_dweight_casts_pass,
)


def paired_ragged_dot_bwd_bf16_rounded_fp32_tm256(*args):
  raise AssertionError("FX test target must not execute")


paired_ragged_dot_bwd_bf16_rounded_fp32_tm256.name = lambda: (
    "tt_redacted::paired_ragged_dot_bwd_bf16_rounded_fp32_tm256"
)


class TestElidePairedDWeightCasts(unittest.TestCase):

  def _make_graph(self):
    graph = torch.fx.Graph()
    dout1 = graph.placeholder("dout1")
    dout3 = graph.placeholder("dout3")
    lhs = graph.placeholder("lhs")
    rhs = graph.placeholder("rhs")
    counts = graph.placeholder("counts")
    backward = graph.call_function(
        paired_ragged_dot_bwd_bf16_rounded_fp32_tm256,
        args=(dout1, dout3, lhs, rhs, counts),
    )
    dinput = graph.call_function(operator.getitem, args=(backward, 0))
    dweight = graph.call_function(operator.getitem, args=(backward, 1))
    downcast = graph.call_function(
        torch.ops.aten._to_copy.default,
        args=(dweight,),
        kwargs={"dtype": torch.bfloat16},
    )
    view = graph.call_function(
        torch.ops.aten.view.default,
        args=(downcast, (8, 2, 16, 24)),
    )
    upcast = graph.call_function(
        torch.ops.aten._to_copy.default,
        args=(view,),
        kwargs={"dtype": torch.float32},
    )
    reduce_scatter = graph.call_function(
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        args=(upcast, "sum", 8, "0"),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(reduce_scatter,),
    )
    graph.output((wait, dinput))
    gm = torch.fx.GraphModule({}, graph)

    with FakeTensorMode():
      dinput_value = torch.empty((32, 16), dtype=torch.bfloat16)
      dweight_value = torch.empty((8, 2, 16, 24), dtype=torch.float32)
      dinput.meta["val"] = dinput_value
      dweight.meta["val"] = dweight_value
      downcast.meta["val"] = torch.empty_like(
          dweight_value, dtype=torch.bfloat16
      )
      view.meta["val"] = downcast.meta["val"]
      upcast.meta["val"] = dweight_value
      reduce_scatter.meta["val"] = torch.empty(
          (1, 2, 16, 24), dtype=torch.float32
      )
      wait.meta["val"] = reduce_scatter.meta["val"]
    return gm, dweight, reduce_scatter

  def test_rewires_reduce_scatter(self):
    gm, dweight, reduce_scatter = self._make_graph()
    apply_elide_paired_dweight_casts_pass(gm, ())

    self.assertIs(reduce_scatter.args[0], dweight)
    targets = [node.target for node in gm.graph.nodes]
    self.assertNotIn(torch.ops.aten._to_copy.default, targets)
    self.assertNotIn(torch.ops.aten.view.default, targets)

  def test_rejects_an_extra_bf16_consumer(self):
    gm, _, _ = self._make_graph()
    view = next(
        node for node in gm.graph.nodes if node.target is torch.ops.aten.view.default
    )
    output = next(node for node in gm.graph.nodes if node.op == "output")
    with gm.graph.inserting_before(output):
      extra = gm.graph.call_function(torch.ops.aten.neg.default, args=(view,))
    output.args = ((output.args[0], extra),)
    gm.graph.lint()
    with self.assertRaisesRegex(ValueError, "Expected one user"):
      apply_elide_paired_dweight_casts_pass(gm, ())

  def test_rejects_a_graph_without_the_enabled_kernel(self):
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    graph.output(value)
    gm = torch.fx.GraphModule({}, graph)
    with self.assertRaisesRegex(ValueError, "found no matching kernels"):
      apply_elide_paired_dweight_casts_pass(gm, ())


if __name__ == "__main__":
  unittest.main()
