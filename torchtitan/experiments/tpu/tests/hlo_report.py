# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What did XLA actually compile?

A single table, printed the same way for every change, so a claim about the
compiled program is something a reviewer can check rather than take on trust.

The rows exist because the failure modes they catch are silent. A layer loop
that XLA unrolled still produces the right loss; a weight all-gather hoisted
above the loop still produces the right loss; activation checkpointing that
quietly became a no-op still produces the right loss -- and in that last case
the memory metric *improves* while it happens. None of it shows up in training
curves, so it has to be read out of the compiled module.

Two halves, deliberately separable:

* **Analysis** is pure text processing over a dump directory. No TPU, no torch,
  no accelerator -- so `test_hlo_report.py` exercises it on fixtures in ordinary
  CI, which is where the load-bearing coverage belongs.
* **Running** shells out to `torchrun` with `XLA_FLAGS=--xla_dump_to=...` in the
  environment. The flag is read when PJRT initialises and cannot be set
  afterwards, hence a subprocess rather than an in-process context manager.

Usage. Produce a dump by training with XLA's dump flags in the environment --
they are read when PJRT initialises, so they have to be set before the process
starts, whatever launcher you use:

    XLA_FLAGS="--xla_dump_to=DIR --xla_dump_hlo_as_text \
--xla_dump_hlo_pass_re=NOTHING" \
        torchrun --nproc_per_node=8 -m torchtitan.experiments.tpu.train \
            --module=torchtitan.experiments.tpu.llama3 \
            --config=llama3_debugmodel --training.steps=3 \
            --parallelism.use_simple_fsdp --compile.enable \
            --compile.backend=tpu --tpu_config.use_graph_trainer \
            --tpu_config.compile_mode=aot_fx_trace \
            --tpu_config.eager_mode=DEFER_AND_FUSE

then read it:

    python -m torchtitan.experiments.tpu.tests.hlo_report --dump-dir=DIR
    python -m torchtitan.experiments.tpu.tests.hlo_report --dump-dir=DIR \
        --expect=torchtitan/experiments/tpu/tests/testdata/\
llama3_debugmodel_unrolled.txt

`testdata/llama3_debugmodel_unrolled.txt` is that golden: the unrolled
`llama3_debugmodel` recipe on 8 v6e chips at `seq_len=2048`, written with
`--write-expected`. It is the "before" the scan work is measured against -- no
layer loop, every all-gather outside one, no stacked layer-axis buffer.

It is SDPA, not splash: `debugmodel` is `head_dim=16` and splash attention
needs 128. A baseline against the kernels 8B actually trains with needs a debug
flavour shaped like 8B, which lands with the scan work that requires it.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import difflib
import glob
import os
import re
import sys


# A computation header sits at column 0: `%name (a: f32[]) -> f32[] {`, and the
# entry is spelled `ENTRY %name (...) {`. Anchoring at column 0 is what keeps
# instruction bodies (always indented) from matching.
_COMPUTATION_RE = re.compile(r"^(ENTRY\s+)?%([\w.\-]+)\s*\(", re.MULTILINE)

# `%foo.1 = f32[8,128]{1,0} opcode(...)`. The shape is skipped by matching up to
# the last `]` or the bare scalar/tuple forms, then the opcode is the identifier
# immediately before `(`. Opcodes are lowercase with dashes: `all-gather`,
# `dynamic-slice`, `custom-call`.
_INSTRUCTION_RE = re.compile(
    r"^\s+(?:ROOT\s+)?%[\w.\-]+ = .*?\b([a-z][a-z\-]*)\(", re.MULTILINE
)

# Every way one computation names another. `calls=` covers fusion and map,
# `body=`/`condition=` cover while, `to_apply=` covers reductions and the
# combiner on a collective, and conditionals use `branch_computations={...}`.
_CALLEE_RE = re.compile(
    r"(?:calls|body|condition|to_apply|select|scatter|update_computation|"
    r"true_computation|false_computation)=%([\w.\-]+)"
)
_BRANCH_RE = re.compile(r"branch_computations=\{([^}]*)\}")

# A `while` instruction, and only a `while` instruction. The `\bwhile\(` form is
# load-bearing: the bare word also appears in pass names such as
# `while-loop-simplifier` inside `metadata=`, which would report a loop that a
# pass had just deleted. Matched per instruction line so the `body=` picked up
# is the one belonging to this `while` and the order is program order.
_WHILE_INSTRUCTION_RE = re.compile(
    r"^\s+(?:ROOT\s+)?%[\w.\-]+ = .*?\bwhile\(.*?\bbody=%([\w.\-]+)",
    re.MULTILINE,
)

# `allocation 3: size 512, color 6, preallocated-temp:` followed by indented
# `value:` lines. The header carries the byte size and the role; the shape comes
# from the header when present (`shape |f32[8,16,128]|`) and from the value line
# otherwise.
_ALLOCATION_RE = re.compile(r"^allocation (\d+): size (\d+),(.*)$", re.MULTILINE)
_ALLOC_SHAPE_RE = re.compile(r"shape \|([^|]+)\|")
_VALUE_SHAPE_RE = re.compile(r"^\s+value:.*?\)\: ([a-z0-9]+\[[^\]]*\])", re.MULTILINE)

# A torch `mm` lowers to either, depending on shape -- matching only `dot`
# reports zero matmuls for a model that is nothing but matmuls. Measured on a
# real dump: the 6-layer debug model compiles to 90 convolutions and no dots.
_MATMULS: tuple[str, ...] = ("dot", "convolution")

# The rows of the table, in print order.
_OP_ROWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("while", ("while",)),
    ("all-gather", ("all-gather", "all-gather-start")),
    ("reduce-scatter", ("reduce-scatter", "reduce-scatter-start")),
    ("all-reduce", ("all-reduce", "all-reduce-start")),
    ("dynamic-slice", ("dynamic-slice",)),
    ("dot/convolution", _MATMULS),
    ("custom-call", ("custom-call",)),
    ("reverse", ("reverse",)),
)

# A loop that reduce-scatters gradients is the backward one. Used only to label
# the columns; the counts do not depend on getting this right.
_BACKWARD_MARKERS = ("reduce-scatter", "reduce-scatter-start")


@dataclasses.dataclass(frozen=True)
class Computation:
  """One HLO computation: its opcodes, and the computations it calls."""

  name: str
  is_entry: bool
  opcodes: collections.Counter
  callees: frozenset
  # `body=` of each `while`, in program order. Order matters: under a joint
  # forward+backward graph the forward loop is emitted before the backward one,
  # which is what lets the report label the columns without reading contents.
  while_bodies: tuple[str, ...]


class Module:
  """One parsed `after_optimizations` HLO module.

  Op counts are attributed to *regions*, not to computations, because the
  question is always "inside the layer loop or outside it" and after
  optimizations almost every real op lives in a fusion the loop body calls. A
  region is therefore the transitive closure of a computation over its callees.
  """

  def __init__(self, text: str):
    self.name = _module_name(text)
    self.computations = _parse_computations(text)
    self.entry = next(
        (c.name for c in self.computations.values() if c.is_entry), None
    )

  @property
  def instruction_count(self) -> int:
    return sum(sum(c.opcodes.values()) for c in self.computations.values())

  @property
  def matmul_count(self) -> int:
    """How this module tells itself apart from the optimizer's -- see
    `training_modules`."""
    return sum(
        c.opcodes[op] for c in self.computations.values() for op in _MATMULS
    )

  def closure(self, root: str) -> frozenset:
    """Every computation reachable from `root`, including `root` itself."""
    seen, stack = set(), [root]
    while stack:
      name = stack.pop()
      if name in seen or name not in self.computations:
        continue
      seen.add(name)
      stack.extend(self.computations[name].callees)
    return frozenset(seen)

  def top_level_loops(self) -> list[str]:
    """Body computations of the ENTRY-level `while`s that run layers.

    Two filters, both of which a naive whole-module `while` count fails.

    Only top-level: a Pallas attention backward kernel carries its own `while`
    over the batch, nested inside whatever loop encloses it. The unrolled
    baseline has one such kernel loop per layer, so counting nested loops
    reports 80 loops for the arm that has no layer loop at all.

    Only loops containing matmuls: the unrolled baseline still has one
    ENTRY-level `while`, a `cumsum` from the RNG. Reporting that as `loop 1`
    invites a reviewer to read the baseline as already looping over layers.
    A layer loop runs the model, so it has matmuls; a cumsum does not.
    """
    if self.entry is None:
      return []
    return [
        body
        for body in self.computations[self.entry].while_bodies
        if self.count(self.closure(body), _MATMULS) > 0
    ]

  def regions(self) -> dict[str, frozenset]:
    """Named regions covering the module: each layer loop, and everything else.

    `outside` is the entry's closure minus every loop's, so an op inside a loop
    is never also counted outside it. That subtraction is the whole point of the
    report: "all-gather appears 2 inside and 0 outside" is the P2 claim, and it
    is only meaningful if the two columns are disjoint.
    """
    if self.entry is None:
      return {}
    loops = self.top_level_loops()
    labels = _loop_labels(self, loops)
    regions, covered = {}, set()
    for label, body in zip(labels, loops):
      body_closure = self.closure(body)
      regions[label] = body_closure
      covered |= body_closure
    regions["outside"] = self.closure(self.entry) - covered
    return regions

  def count(self, region: frozenset, opcodes: tuple[str, ...]) -> int:
    return sum(
        self.computations[c].opcodes[op]
        for c in region
        if c in self.computations
        for op in opcodes
    )


def _module_name(text: str) -> str:
  first = text.split("\n", 1)[0]
  match = re.match(r"HloModule ([\w.\-]+)", first)
  return match.group(1) if match else "<unnamed>"


def _parse_computations(text: str) -> dict[str, Computation]:
  """Split the module at column-0 computation headers.

  Each computation runs from its header to the next one (or to end of file), so
  no brace matching is needed -- HLO always closes a computation at column 0 and
  never opens one anywhere else.
  """
  headers = list(_COMPUTATION_RE.finditer(text))
  out = {}
  for i, match in enumerate(headers):
    end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
    body = text[match.start() : end]
    name = match.group(2)
    callees = set(_CALLEE_RE.findall(body))
    for group in _BRANCH_RE.findall(body):
      callees.update(re.findall(r"%([\w.\-]+)", group))
    out[name] = Computation(
        name=name,
        is_entry=bool(match.group(1)),
        opcodes=collections.Counter(_INSTRUCTION_RE.findall(body)),
        callees=frozenset(callees),
        while_bodies=_while_bodies(body),
    )
  return out


def _while_bodies(text: str) -> tuple[str, ...]:
  """The `body=` of each `while` instruction in one computation, in order."""
  return tuple(
      match.group(1)
      for match in _WHILE_INSTRUCTION_RE.finditer(text)
  )


def _loop_labels(module: "Module", loops: list[str]) -> list[str]:
  """`fwd loop` / `bwd loop` when there are two, `loop N` otherwise.

  With one loop per direction the named columns read the way the design does.
  Any other count means something unmodelled is going on and numbered columns
  are the honest presentation.
  """
  if len(loops) == 2:
    backward = [
        module.count(module.closure(body), _BACKWARD_MARKERS) > 0
        for body in loops
    ]
    if backward == [False, True]:
      return ["fwd loop", "bwd loop"]
  return [f"loop {i + 1}" for i in range(len(loops))]


@dataclasses.dataclass(frozen=True)
class Allocation:
  """One buffer XLA reserved, from `*-buffer-assignment.txt`."""

  index: int
  size: int
  shape: str

  @property
  def leading_dim(self) -> int | None:
    match = re.search(r"\[(\d+)", self.shape)
    return int(match.group(1)) if match else None


def parse_buffer_assignment(text: str) -> list[Allocation]:
  """Byte sizes, straight from the compiler.

  Counting all-gather *instructions* says the gather moved; only the byte size
  says it got smaller. This is the difference between "the structure looks
  right" and "the memory claim is true", and it is why the report reads this
  file as well as the HLO.
  """
  matches = list(_ALLOCATION_RE.finditer(text))
  out = []
  for i, match in enumerate(matches):
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    tail = match.group(3)
    body = text[match.start() : end]
    shape = _ALLOC_SHAPE_RE.search(tail) or _VALUE_SHAPE_RE.search(body)
    out.append(
        Allocation(
            index=int(match.group(1)),
            size=int(match.group(2)),
            shape=shape.group(1) if shape else "?",
        )
    )
  return out


def largest_layer_axis_buffer(
    allocations: list[Allocation], n_layers: int
) -> Allocation | None:
  """The biggest buffer whose leading dimension is the layer count.

  This is the P1 check stated as a number. A stacked `[n_layers, ...]` weight
  that reaches the device materialised -- rather than being sliced inside the
  loop -- shows up here, and nowhere else, because it is numerically correct.

  It is a heuristic and says so: any tensor that happens to have `n_layers` as
  its leading dimension matches. That is acceptable for a report whose job is to
  make a reviewer look, and the shape is printed so they can.
  """
  candidates = [a for a in allocations if a.leading_dim == n_layers]
  return max(candidates, key=lambda a: a.size, default=None)


def load_dump(dump_dir: str) -> tuple[list[Module], dict[str, list[Allocation]]]:
  """Read every compiled module in a dump directory.

  Fails loudly on an empty directory. An empty dump would let every count in
  this report come back zero and every assertion pass vacuously -- which is the
  exact shape of the silent success the report exists to catch.
  """
  paths = sorted(glob.glob(os.path.join(dump_dir, "*after_optimizations.txt")))
  if not paths:
    raise FileNotFoundError(
        f"no *after_optimizations.txt in {dump_dir}: XLA never compiled"
        " anything, so nothing in this report would mean anything. Check that"
        " the run actually executed a step, and that XLA_FLAGS reached it."
    )
  modules, buffers = [], {}
  for path in paths:
    module = Module(_read(path))
    modules.append(module)
    assignment = path.replace(
        "after_optimizations.txt", "after_optimizations-buffer-assignment.txt"
    )
    if os.path.exists(assignment):
      buffers[module.name] = parse_buffer_assignment(_read(assignment))
  return modules, buffers


def _read(path: str) -> str:
  with open(path) as handle:
    return handle.read()


def training_modules(modules: list[Module]) -> list[Module]:
  """The modules that do matmuls, which is the model and nothing else.

  A step does not compile one program. Measured on a `DEFER_AND_FUSE` run of
  `llama3_debugmodel`: 76 programs, of which 75 are the optimizer
  (`foreach_pow`, `foreach_div_`, ...), the RNG, and collectives issued outside
  the graph. Those are elementwise, and several are *larger* than the model --
  the biggest is a `foreach_pow` at 8665 instructions against the model's 8549 --
  so picking the largest program picks the optimizer.

  Matmuls are the discriminator that survives that: only the model has any.
  """
  return [m for m in modules if m.matmul_count > 0]


def main_module(modules: list[Module]) -> Module:
  """The one training program.

  More than one means the model recompiled, which invalidates every number
  below -- the report would describe whichever copy happened to be biggest.
  """
  found = training_modules(modules)
  if not found:
    raise ValueError(
        f"none of the {len(modules)} compiled programs contains a dot or"
        " convolution, so none of them is the model. Every count in this report"
        " would be zero, which reads like a pass. Check that the run trained"
        " rather than failing during setup."
    )
  return max(found, key=lambda m: m.instruction_count)


def build_report(
    modules: list[Module],
    buffers: dict[str, list[Allocation]],
    n_layers: int,
) -> dict:
  """The table, as data. Rendering is separate so tests assert on numbers."""
  module = main_module(modules)
  regions = module.regions()
  rows = {
      label: {
          region: module.count(members, opcodes)
          for region, members in regions.items()
      }
      for label, opcodes in _OP_ROWS
  }
  allocations = buffers.get(module.name, [])
  layer_buffer = largest_layer_axis_buffer(allocations, n_layers)
  return {
      "module": module.name,
      "programs_compiled": len(modules),
      # >1 means the model recompiled. Most of `programs_compiled` is the
      # optimizer under `DEFER_AND_FUSE` and its count is noise; this one is not.
      "training_programs": len(training_modules(modules)),
      "regions": list(regions),
      "rows": rows,
      "instruction_count": module.instruction_count,
      "largest_layer_axis_buffer": (
          None
          if layer_buffer is None
          else {"size": layer_buffer.size, "shape": layer_buffer.shape}
      ),
      "largest_buffer": (
          max(a.size for a in allocations) if allocations else None
      ),
      "total_allocated": sum(a.size for a in allocations),
  }


def stable_table(report: dict) -> str:
  """The part of the report that a compiler upgrade should not move.

  This is what gets checked in as a golden and what `--expect` compares. It is
  deliberately not the whole report: the module name carries a source line
  number, and instruction counts and buffer bytes drift with every XLA change.
  Pinning those would give a golden that fails for reasons no one here caused,
  and a golden that cries wolf gets deleted. What is pinned is the structure --
  which regions exist and what is in them -- because that is what the scan work
  changes and what a regression would change back.
  """
  regions = report["regions"]
  width = max([len(r) for r in regions] + [8])
  lines = [" " * 20 + "".join(r.rjust(width + 2) for r in regions)]
  for label, counts in report["rows"].items():
    lines.append(
        label.ljust(20)
        + "".join(str(counts[r]).rjust(width + 2) for r in regions)
    )
  buf = report["largest_layer_axis_buffer"]
  lines += [
      "",
      "layer-axis buffer:  " + ("none" if buf is None else buf["shape"]),
      f"training programs:  {report['training_programs']}",
  ]
  return "\n".join(lines) + "\n"


def render(report: dict) -> str:
  """Everything, for a human: the stable table plus the numbers that drift."""
  lines = [
      f"module: {report['module']}  ({report['instruction_count']} instructions)",
      "",
      stable_table(report).rstrip("\n"),
      "",
  ]
  buf = report["largest_layer_axis_buffer"]
  if buf is not None:
    lines.append(f"layer-axis buffer size:     {_mib(buf['size'])}")
  if report["largest_buffer"] is not None:
    lines.append(f"largest buffer:             {_mib(report['largest_buffer'])}")
    lines.append(f"total allocated:            {_mib(report['total_allocated'])}")
  lines.append(f"programs compiled:          {report['programs_compiled']}")
  return "\n".join(lines)


def compare(report: dict, expected_path: str) -> str | None:
  """A unified diff against a checked-in golden, or None if they agree."""
  actual = stable_table(report)
  with open(expected_path) as handle:
    expected = handle.read()
  if actual == expected:
    return None
  return "".join(
      difflib.unified_diff(
          expected.splitlines(keepends=True),
          actual.splitlines(keepends=True),
          fromfile=expected_path,
          tofile="this run",
      )
  )


def _mib(size: int) -> str:
  return f"{size / (1 << 20):.2f} MiB"


def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dump-dir", required=True, help="an XLA dump to read")
  parser.add_argument(
      "--n-layers",
      type=int,
      default=6,
      help="layer count, for the layer-axis buffer check",
  )
  parser.add_argument(
      "--expect", help="golden table to compare against; exit 1 on any diff"
  )
  parser.add_argument("--write-expected", help="(re)write a golden table")
  return parser.parse_args(argv)


def main(argv: list[str]) -> int:
  args = _parse_args(argv)
  modules, buffers = load_dump(args.dump_dir)
  report = build_report(modules, buffers, args.n_layers)
  print(render(report))
  if args.write_expected:
    with open(args.write_expected, "w") as handle:
      handle.write(stable_table(report))
  if args.expect:
    diff = compare(report, args.expect)
    if diff:
      print(f"\nthe compiled program is not the one recorded in {args.expect}:")
      print(diff)
      return 1
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
