# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Can the report be wrong without failing?

Every number `hlo_report` prints is used to decide whether a change worked, and
a parser that quietly matches nothing reports zero -- which reads exactly like
"the op is not there", which is the answer we are usually hoping for. So the
cases below are mostly about the ways a count can be wrong in the *reassuring*
direction.

Runs on CPU with no accelerator: the fixture is text.

    python -m torchtitan.experiments.tpu.tests.test_hlo_report
"""

import os
import tempfile

from absl.testing import absltest
from torchtitan.experiments.tpu.tests import hlo_report


# Shaped after a real `after_optimizations` dump: computations at column 0,
# instructions indented, fusions holding the arithmetic, `metadata=` carrying
# pass names, and the layer loops as two `while`s in ENTRY.
#
# The awkward parts are deliberate. `%kernel_body` is a `while` *nested inside*
# the backward loop -- the Pallas attention backward carries one, and counting
# it as a layer loop would report three loops for a two-loop program. The
# `metadata={op_name="while-loop-simplifier"}` on `%bitcast` is the other trap:
# a bare `while` word match finds a loop there that does not exist.
_HLO = """\
HloModule jit_step, is_scheduled=true, entry_computation_layout={()->f32[]}

%fused_mlp (param_0: f32[8,8]) -> f32[8,8] {
  %param_0 = f32[8,8]{1,0} parameter(0)
  ROOT %convolution.1 = f32[8,8]{1,0} convolution(%param_0, %param_0), dim_labels=bf_io->bf
}

%kernel_body (arg: (s32[], f32[8,8])) -> (s32[], f32[8,8]) {
  %arg = (s32[], f32[8,8]) parameter(0)
  ROOT %custom-call.9 = f32[8,8]{1,0} custom-call(%arg), custom_call_target="tpu_custom_call"
}

%kernel_cond (arg: (s32[], f32[8,8])) -> pred[] {
  %arg.2 = (s32[], f32[8,8]) parameter(0)
  ROOT %compare.1 = pred[] compare(%arg.2, %arg.2), direction=LT
}

%fwd_body (arg_tuple: (s32[], f32[8,8])) -> (s32[], f32[8,8]) {
  %arg_tuple = (s32[], f32[8,8]) parameter(0)
  %all-gather.2 = f32[8,8]{1,0} all-gather(%arg_tuple), dimensions={0}
  %dynamic-slice.1 = f32[8,8]{1,0} dynamic-slice(%all-gather.2), dynamic_slice_sizes={8,8}
  ROOT %fusion.3 = f32[8,8]{1,0} fusion(%dynamic-slice.1), kind=kOutput, calls=%fused_mlp
}

%fwd_cond (arg_tuple: (s32[], f32[8,8])) -> pred[] {
  %arg_tuple.2 = (s32[], f32[8,8]) parameter(0)
  ROOT %compare.2 = pred[] compare(%arg_tuple.2, %arg_tuple.2), direction=LT
}

%bwd_body (arg_tuple: (s32[], f32[8,8])) -> (s32[], f32[8,8]) {
  %arg_tuple.3 = (s32[], f32[8,8]) parameter(0)
  %all-gather.4 = f32[8,8]{1,0} all-gather(%arg_tuple.3), dimensions={0}
  %while.5 = (s32[], f32[8,8]) while(%arg_tuple.3), condition=%kernel_cond, body=%kernel_body
  %fusion.6 = f32[8,8]{1,0} fusion(%while.5), kind=kOutput, calls=%fused_mlp
  ROOT %reduce-scatter.7 = f32[8,8]{1,0} reduce-scatter(%fusion.6), dimensions={0}
}

%bwd_cond (arg_tuple: (s32[], f32[8,8])) -> pred[] {
  %arg_tuple.4 = (s32[], f32[8,8]) parameter(0)
  ROOT %compare.3 = pred[] compare(%arg_tuple.4, %arg_tuple.4), direction=LT
}

ENTRY %main.1 () -> f32[] {
  %constant.1 = s32[] constant(0)
  %tuple.1 = (s32[], f32[8,8]) tuple(%constant.1, %constant.1)
  %bitcast.1 = f32[8,8]{1,0} bitcast(%tuple.1), metadata={op_name="while-loop-simplifier"}
  %while.2 = (s32[], f32[8,8]) while(%tuple.1), condition=%fwd_cond, body=%fwd_body
  %while.3 = (s32[], f32[8,8]) while(%while.2), condition=%bwd_cond, body=%bwd_body
  ROOT %get-tuple-element.1 = f32[] get-tuple-element(%while.3), index=0
}
"""

# Two layer-axis buffers (leading dim 6) of different sizes, plus a larger
# buffer that is not layer-axis -- so "pick the largest" and "pick the largest
# *layer-axis*" cannot accidentally agree.
_BUFFERS = """\
BufferAssignment:
allocation 0: size 65536, output shape is |f32[8,16,128]|, maybe-live-out:
 value: <11 fusion @0> (size=65536,offset=0): f32[8,16,128]{2,1,0}
allocation 1: size 6144, parameter 1, shape |f32[6,32,8]| at ShapeIndex {}:
 value: <8 stacked @0> (size=6144,offset=0): f32[6,32,8]{2,1,0}
allocation 2: size 1536, color 6, preallocated-temp:
 value: <10 slice @6> (size=1536,offset=0): f32[6,8,8]{2,1,0}
"""


class ParseTest(absltest.TestCase):
  """The module structure, before any counting."""

  def setUp(self):
    super().setUp()
    self.module = hlo_report.Module(_HLO)

  def test_entry_is_found(self):
    self.assertEqual(self.module.entry, "main.1")

  def test_every_computation_is_parsed(self):
    self.assertCountEqual(
        self.module.computations,
        [
            "fused_mlp", "kernel_body", "kernel_cond", "fwd_body", "fwd_cond",
            "bwd_body", "bwd_cond", "main.1",
        ],
    )

  def test_only_top_level_whiles_count_as_layer_loops(self):
    """The nested Pallas kernel loop is not a layer loop.

    Without this the unrolled baseline -- which has one attention-backward
    kernel loop per layer -- reports 80 loops for the arm that has no layer loop
    at all, and the whole comparison inverts.
    """
    self.assertEqual(self.module.top_level_loops(), ["fwd_body", "bwd_body"])

  def test_a_pass_name_in_metadata_is_not_a_loop(self):
    """`while-loop-simplifier` names a pass, not a `while`.

    A bare-word match reports a loop here, which is the reassuring direction:
    it would let a program whose loop XLA deleted still report one.
    """
    entry = self.module.computations["main.1"]
    self.assertEqual(len(entry.while_bodies), 2)

  def test_loops_are_in_program_order(self):
    self.assertEqual(self.module.top_level_loops()[0], "fwd_body")

  def test_a_loop_that_runs_no_matmul_is_not_a_layer_loop(self):
    """The unrolled baseline's one ENTRY-level `while` is an RNG cumsum.

    Reported as `loop 1` it reads as though the baseline already loops over
    layers, which is the opposite of what the baseline is for.
    """
    elementwise = _HLO.replace(
        "ROOT %fusion.3 = f32[8,8]{1,0} fusion(%dynamic-slice.1),"
        " kind=kOutput, calls=%fused_mlp",
        "ROOT %add.3 = f32[8,8]{1,0} add(%dynamic-slice.1, %dynamic-slice.1)",
    )
    self.assertNotEqual(elementwise, _HLO)
    module = hlo_report.Module(elementwise)
    self.assertEqual(module.top_level_loops(), ["bwd_body"])


class RegionTest(absltest.TestCase):
  """Where an op is, which is the only thing this report is really about."""

  def setUp(self):
    super().setUp()
    self.module = hlo_report.Module(_HLO)
    self.regions = self.module.regions()

  def test_loops_are_labelled_by_direction(self):
    """The loop that reduce-scatters gradients is the backward one."""
    self.assertEqual(list(self.regions), ["fwd loop", "bwd loop", "outside"])

  def test_regions_are_disjoint(self):
    """`2 inside and 0 outside` means nothing if a computation is in both."""
    self.assertEmpty(self.regions["fwd loop"] & self.regions["outside"])
    self.assertEmpty(self.regions["bwd loop"] & self.regions["outside"])

  def test_all_gather_is_counted_inside_and_not_outside(self):
    """The P2 claim, in the form the report states it."""
    self.assertEqual(
        self.module.count(self.regions["fwd loop"], ("all-gather",)), 1
    )
    self.assertEqual(
        self.module.count(self.regions["outside"], ("all-gather",)), 0
    )

  def test_ops_inside_a_called_fusion_belong_to_the_caller(self):
    """After optimizations the arithmetic is inside fusions, not inline.

    `%fused_mlp` holds the only convolution in the module and is called from
    both loop bodies. Counting per-computation instead of per-closure reports
    zero matmuls in both loops.
    """
    self.assertEqual(
        self.module.count(self.regions["fwd loop"], ("dot", "convolution")), 1
    )

  def test_convolution_counts_as_a_matmul(self):
    """A torch `mm` lowers to `convolution`, not `dot`, at these shapes.

    Measured on a real TPU dump. A `dot`-only row reports no matmuls for a model
    that is nothing but matmuls.
    """
    row = dict(hlo_report._OP_ROWS)["dot/convolution"]
    self.assertIn("convolution", row)

  def test_a_nested_kernel_loop_is_counted_inside_its_enclosing_loop(self):
    """Not a layer loop, but still real work that happens in the backward."""
    self.assertEqual(self.module.count(self.regions["bwd loop"], ("while",)), 1)
    self.assertEqual(self.module.count(self.regions["outside"], ("while",)), 2)


class BufferTest(absltest.TestCase):
  """Byte sizes, which is where the memory claim actually lives."""

  def setUp(self):
    super().setUp()
    self.allocations = hlo_report.parse_buffer_assignment(_BUFFERS)

  def test_sizes_and_shapes_are_read(self):
    self.assertEqual([a.size for a in self.allocations], [65536, 6144, 1536])
    self.assertEqual(self.allocations[1].shape, "f32[6,32,8]")

  def test_shape_falls_back_to_the_value_line(self):
    """`preallocated-temp` headers carry no `shape |...|`."""
    self.assertEqual(self.allocations[2].shape, "f32[6,8,8]")

  def test_largest_layer_axis_buffer_ignores_larger_non_layer_buffers(self):
    """The 64 KiB output is bigger, and is not what P1 is about."""
    found = hlo_report.largest_layer_axis_buffer(self.allocations, n_layers=6)
    self.assertEqual(found.shape, "f32[6,32,8]")

  def test_no_layer_axis_buffer_is_reported_as_none(self):
    """The passing state, and it has to be distinguishable from a parse miss."""
    self.assertIsNone(
        hlo_report.largest_layer_axis_buffer(self.allocations, n_layers=99)
    )


class VacuousPassTest(absltest.TestCase):
  """The report must fail rather than report zeros it never looked for."""

  def test_an_empty_dump_raises(self):
    with tempfile.TemporaryDirectory() as empty:
      with self.assertRaisesRegex(FileNotFoundError, "never compiled"):
        hlo_report.load_dump(empty)

  def test_the_training_program_is_picked_over_a_bigger_optimizer_one(self):
    """The model is not the biggest program a step compiles.

    Measured: a `DEFER_AND_FUSE` step compiles 92 programs and the largest is a
    `foreach_pow` from the optimizer, at 8631 instructions against the model's
    7805. Picking by size picks that, and then every row of the table reads zero
    -- which is the answer the report is usually hoping for. `_BIG_ELEMENTWISE`
    is that case in miniature: more instructions than `_HLO`, no matmul.
    """
    with tempfile.TemporaryDirectory() as dump:
      _write(dump, "module_0001.foreach.after_optimizations.txt",
             _BIG_ELEMENTWISE)
      _write(dump, "module_0002.step.after_optimizations.txt", _HLO)
      modules, _ = hlo_report.load_dump(dump)
      self.assertGreater(
          modules[0].instruction_count, modules[1].instruction_count
      )
      self.assertEqual(hlo_report.main_module(modules).name, "jit_step")

  def test_a_dump_with_no_model_in_it_raises(self):
    """Rather than reporting a table of zeros about the optimizer."""
    with tempfile.TemporaryDirectory() as dump:
      _write(dump, "module_0001.foreach.after_optimizations.txt",
             _BIG_ELEMENTWISE)
      modules, _ = hlo_report.load_dump(dump)
      with self.assertRaisesRegex(ValueError, "none of them is the model"):
        hlo_report.main_module(modules)

  def test_a_recompiled_model_is_visible(self):
    """Two programs with matmuls means the model recompiled."""
    with tempfile.TemporaryDirectory() as dump:
      _write(dump, "module_0001.step.after_optimizations.txt", _HLO)
      _write(dump, "module_0002.step.after_optimizations.txt", _HLO)
      _write(dump, "module_0003.foreach.after_optimizations.txt",
             _BIG_ELEMENTWISE)
      modules, buffers = hlo_report.load_dump(dump)
      report = hlo_report.build_report(modules, buffers, n_layers=6)
      self.assertEqual(report["programs_compiled"], 3)
      self.assertEqual(report["training_programs"], 2)


# Bigger than `_HLO` and entirely elementwise, like the optimizer's programs.
_BIG_ELEMENTWISE = "".join(
    [
        "HloModule tt_jit_foreach_pow_pow, is_scheduled=true\n\n",
        "ENTRY %main.2 () -> f32[] {\n",
        *[
            f"  %multiply.{i} = f32[8,8]{{1,0}} multiply(%multiply.{i - 1})\n"
            for i in range(1, 200)
        ],
        "  ROOT %constant.2 = f32[] constant(1)\n}\n",
    ]
)


def _write(directory: str, name: str, text: str) -> None:
  with open(os.path.join(directory, name), "w") as handle:
    handle.write(text)


class RenderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.report = hlo_report.build_report(
        [hlo_report.Module(_HLO)], {}, n_layers=6
    )

  def test_the_table_names_every_region(self):
    table = hlo_report.render(self.report)
    for expected in ("fwd loop", "bwd loop", "outside", "all-gather"):
      self.assertIn(expected, table)

  def test_the_golden_holds_no_number_a_compiler_upgrade_moves(self):
    """Instruction counts and byte sizes stay out of the compared part.

    A golden that fails whenever XLA changes gets regenerated without being
    read, and then it is not checking anything.
    """
    stable = hlo_report.stable_table(self.report)
    self.assertNotIn(str(self.report["instruction_count"]), stable)
    self.assertNotIn(self.report["module"], stable)
    self.assertNotIn("MiB", stable)

  def test_compare_is_silent_on_a_match_and_diffs_otherwise(self):
    with tempfile.TemporaryDirectory() as scratch:
      golden = os.path.join(scratch, "golden.txt")
      _write(scratch, "golden.txt", hlo_report.stable_table(self.report))
      self.assertIsNone(hlo_report.compare(self.report, golden))

      hoisted = dict(self.report, rows=dict(self.report["rows"]))
      hoisted["rows"]["all-gather"] = {"fwd loop": 0, "bwd loop": 0,
                                       "outside": 1}
      diff = hlo_report.compare(hoisted, golden)
      self.assertIn("all-gather", diff)


if __name__ == "__main__":
  absltest.main()
