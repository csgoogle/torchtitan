# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A reverse-direction scan HOP, and the backward that uses it.

`torch.ops.higher_order.scan` has no `reverse` argument, so
`ScanAutogradImpl.call_backward` emulates the reversed scan its own docstring
asks for by calling `torch.flip` on the saved forward intermediates. On TPU
that flip is a materialized copy of the whole residual stack: at LLaMA-3 8B,
seq 8192, local batch 4 it is a second `bf16[31,4,8192,4096]` buffer -- 7.75
GiB, 28% of the arena, holding nothing that is not already resident. Upstream
knows; the call site carries `# TODO: torch.flip copies the tensor, we should
optimize it away`.

torch_tpu can already lower a genuine reverse scan (`ScanDirection.kReverse`)
and writes its `ys` aligned to the *input* index rather than to iteration
order, so the input flip and the output flip cancel and both can be dropped.
Nothing in the stack asks for that lowering today, which is what this module
fixes.

Two shapes of workaround, both deliberate:

  * A sibling HOP rather than a `reverse` argument on the existing one.
    `ScanOp.__call__` and `ScanOp.gen_schema` share a fixed four-argument
    signature that nine dispatch registrations are written against.
  * A `TorchFunctionMode` that elides the two flips rather than a rewritten
    `call_backward`. Rewriting it means vendoring ninety lines of torch
    internals and re-vendoring them on every torch bump; the flips are the
    only part that has to change. The guards at the end of the patched
    `call_backward` are what make that safe -- if upstream's shape changes,
    this raises instead of silently returning a reversed gradient.

The upstream ask that deletes this file is a `reverse=` argument on the scan
HOP, lowered by whoever owns the backend.
"""

import contextlib
import importlib
import itertools
from typing import Any

import torch
from torch._C import DispatchKey
from torch._higher_order_ops.scan import (
    _extract_carry_and_out,
    generic_scan,
    ScanOp,
    stack_y,
    trace_scan,
)
from torch._higher_order_ops.utils import autograd_not_implemented, first_slice_copy
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.overrides import TorchFunctionMode
import torch.utils._pytree as pytree

from torchtitan.tools.logging import logger

# `import torch._higher_order_ops.scan as m` would bind the re-exported `scan`
# *function*, not the module -- the package's `__init__` shadows the submodule
# name. Ask the import machinery for the module directly.
_scan_mod = importlib.import_module("torch._higher_order_ops.scan")


class ScanReverseOp(ScanOp):
  """`scan`, iterating from the last index to the first.

  Identical to `torch.ops.higher_order.scan` in signature, carry semantics, and
  output shapes. Two differences: the body runs on index `N-1` first and index
  `0` last, and `ys[i]` holds the output of the step that consumed `xs[i]` --
  indexed by input position, not by iteration order. That is the JAX
  convention, it is what the TPU lowering already does, and it is what makes
  the flips cancel.

  Deliberately not autograd-registered. This op exists to *be* the backward of
  `scan` and is only ever reached under no-grad from
  `ScanAutogradImpl.call_backward`; differentiating through it fails loudly at
  the autograd key rather than returning a silently wrong gradient.
  """

  def __init__(self):
    # Skip `ScanOp.__init__` -- it hardcodes the "scan" name.
    torch._ops.HigherOrderOperator.__init__(self, "scan_reverse")  # pylint: disable=protected-access


scan_reverse_op = ScanReverseOp()


@scan_reverse_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def _dense(combine_fn, init, xs, additional_inputs):
  """Reference implementation: flip, scan forward, flip back.

  Deliberately the naive construction. This is the CPU path, and its whole job
  is to be an oracle the TPU lowering can be tested against, so it must share
  no machinery with the thing under test. The copies it makes are the copies
  this module removes on TPU; on CPU they are irrelevant.
  """
  flipped = [torch.flip(x, (0,)) for x in xs]
  outs = generic_scan(
      combine_fn, init, flipped, additional_inputs=additional_inputs
  )
  carry, ys = _extract_carry_and_out(list(outs), len(init))
  return (*carry, *[torch.flip(y, (0,)) for y in ys])


scan_reverse_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(scan_reverse_op, deferred_error=True)
)


@scan_reverse_op.py_impl(ProxyTorchDispatchMode)
def _proxy(mode, combine_fn, init, xs, additional_inputs):
  return trace_scan(
      mode, scan_reverse_op, combine_fn, init, xs, additional_inputs
  )


@scan_reverse_op.py_impl(FakeTensorMode)
def _fake(mode, combine_fn, init, xs, additional_inputs):
  # Direction affects neither shapes nor dtypes, so this is `scan`'s rule
  # verbatim. `first_slice_copy` is still the right sample slice: every slice
  # of `xs` has the same shape, the reverse scan just visits them in the other
  # order.
  with mode:
    carry, outputs = _extract_carry_and_out(
        combine_fn(
            *init,
            *[first_slice_copy(inp) for inp in xs],
            *additional_inputs,
        ),
        len(init),
    )
    return (*carry, *(stack_y(t, xs[0].shape[0]) for t in outputs))


@scan_reverse_op.py_functionalize_impl
def _functionalize(ctx, combine_fn, init, xs, additional_inputs):
  from torch._higher_order_ops.utils import (  # pylint: disable=g-import-not-at-top
      _check_alias_and_mutation,
      _maybe_run_with_interpreter,
  )

  unwrapped_xs = ctx.unwrap_tensors(xs)
  unwrapped_init = ctx.unwrap_tensors(init)
  unwrapped_additional = ctx.unwrap_tensors(additional_inputs)
  with ctx.redispatch_to_next():
    functional_combine_fn = ctx.functionalize(
        _maybe_run_with_interpreter(combine_fn)
    )
    sample_inputs = list(
        itertools.chain(
            unwrapped_init,
            [first_slice_copy(inp) for inp in unwrapped_xs],
            unwrapped_additional,
        )
    )
    _check_alias_and_mutation(
        combine_fn,
        sample_inputs,
        "scan_reverse",
        hasattr(ctx, "mode") and ctx.mode.pre_dispatch,
    )
    ret = scan_reverse_op(
        functional_combine_fn, unwrapped_init, unwrapped_xs, unwrapped_additional
    )
  return ctx.wrap_tensors(ret)


def _register_tpu_lowering() -> bool:
  """Points the device keys at the `kReverse` lowering the forward scan uses.

  torch_tpu's `_handle_scan_impl` already takes `reverse`, and the forward scan
  on this lane goes through that same function, so the reverse gets the
  lowering the lane actually uses rather than a second copy of it. The three
  dispatch keys mirror torch_tpu's own registration for the forward scan.

  Requires torch_tpu with the Bug A fix (google-pytorch/torch_tpu#3301):
  without it `_handle_scan_impl` hands the carry inits and additional inputs to
  `fx_to_mlir` as live tpu tensors, which in a real model are pending eager
  values, and the lowering dies with "argument tensor has deferred ops".
  """
  try:
    from torch_tpu._internal.compile.scan import _handle_scan_impl  # pylint: disable=g-import-not-at-top
  except ImportError:
    return False

  def tpu_scan_reverse(combine_fn, init, xs, additional_inputs):
    return _handle_scan_impl(combine_fn, init, xs, additional_inputs, reverse=True)

  for key in (DispatchKey.XLA, DispatchKey.PrivateUse1, DispatchKey.Meta):
    with contextlib.suppress(RuntimeError):  # Already registered: imported twice.
      scan_reverse_op.py_impl(key)(tpu_scan_reverse)
  return True


_TPU_LOWERING_REGISTERED = _register_tpu_lowering()


class _ElideDim0Flip(TorchFunctionMode):
  """Turns the two `torch.flip(x, (0,))` calls in `call_backward` into no-ops.

  Eliding by shape alone would also silence a `flip` the scanned body itself
  performs, so the mode runs in two phases. Before the scan call it elides only
  tensors it was handed by identity -- the flat `bw_xs`, which the caller
  recomputes from the same `ScanAutogradImpl` fields -- and anything else
  traced during `materialize_as_graph` passes through. After the scan call,
  tracing is over and the only remaining flips are the ones applied to
  `grad_xs`, so those are elided unconditionally.

  The mode also disarms itself for the duration of the scan call. Lowering the
  scan runs a lot of code -- torch_tpu traces the body and compiles it to MLIR
  underneath the dispatch -- and none of it is `call_backward`'s flips. Eliding
  one in there would corrupt the body graph rather than save a copy.
  """

  def __init__(self, expected_inputs: list[Any]):
    super().__init__()
    self.expected_ids = {id(t) for t in expected_inputs}
    self.armed = True
    self.after_scan = False
    self.elided_inputs = 0
    self.elided_outputs = 0

  @contextlib.contextmanager
  def disarmed(self):
    self.armed = False
    try:
      yield
    finally:
      self.armed = True

  def __torch_function__(self, func, types, args=(), kwargs=None):
    kwargs = kwargs or {}
    if (
        self.armed
        and func is torch.flip
        and len(args) == 2
        and tuple(args[1]) == (0,)
    ):
      if self.after_scan:
        self.elided_outputs += 1
        return args[0]
      if id(args[0]) in self.expected_ids:
        self.elided_inputs += 1
        return args[0]
    return func(*args, **kwargs)


_original_call_backward = None


def install() -> None:
  """Makes `scan`'s backward a real reverse scan instead of a flipped one.

  Idempotent, and import-order-independent: it patches
  `ScanAutogradImpl.call_backward`, which is resolved per call.

  The rewrite is exact, not an approximation. `scan_reverse_op` visits the same
  steps in the same order the flipped forward scan did, and writes each
  `grad_x` to the input index the flipped output would have been permuted back
  to. Both flips are therefore redundant and removing them changes no value at
  all; what changes is that the residual stack is read in place instead of
  copied.

  Raises:
    RuntimeError: if torch_tpu is not importable, so there is no reverse
      lowering to dispatch to. Falling back to the flip-based dense path would
      silently reintroduce the copy this exists to remove.
  """
  global _original_call_backward
  if _original_call_backward is not None:
    return
  if not _TPU_LOWERING_REGISTERED:
    raise RuntimeError(
        "scan_reverse has no device lowering:"
        " torch_tpu._internal.compile.scan is not importable. Without it the"
        " reverse scan falls back to the flip-based dense path, which is what"
        " this patch exists to avoid."
    )

  impl_cls = _scan_mod.ScanAutogradImpl
  _original_call_backward = impl_cls.call_backward

  def call_backward(self, *grad_fw_outputs):
    n_carry = len(self.init)
    bw_xs = [
        list(grad_fw_outputs[n_carry:]),
        self.saved_fw_xs,
        self.saved_intermediates,
    ]
    mode = _ElideDim0Flip(pytree.tree_flatten(bw_xs)[0])
    original_scan_op = _scan_mod.scan_op

    class _ArmedScanOp:
      """Redirects the one `scan_op(...)` call inside `call_backward`."""

      def __call__(self, *args, **kwargs):
        try:
          with mode.disarmed():
            return scan_reverse_op(*args, **kwargs)
        finally:
          mode.after_scan = True

      def __getattr__(self, name):
        return getattr(original_scan_op, name)

    _scan_mod.scan_op = _ArmedScanOp()
    try:
      with mode:
        out = _original_call_backward(self, *grad_fw_outputs)
    finally:
      _scan_mod.scan_op = original_scan_op

    # Three ways upstream can drift out from under this patch, each of which
    # would otherwise show up as a wrong gradient rather than as an error.
    if not mode.after_scan:
      raise RuntimeError(
          "scan_reverse: call_backward returned without dispatching a scan;"
          " the upstream implementation has changed shape and this patch no"
          " longer applies."
      )
    if mode.elided_inputs != len(mode.expected_ids):
      raise RuntimeError(
          f"scan_reverse: elided {mode.elided_inputs} of"
          f" {len(mode.expected_ids)} input flips. Some backward input is"
          " still being copied, so the reverse scan would read it in the wrong"
          " order."
      )
    if mode.elided_outputs == 0:
      raise RuntimeError(
          "scan_reverse: no output flip was elided; grad_xs would be returned"
          " reversed."
      )
    logger.info(
        "scan_reverse: backward is a native reverse scan (%d input and %d"
        " output flip(s) removed).",
        mode.elided_inputs,
        mode.elided_outputs,
    )
    return out

  impl_cls.call_backward = call_backward
  logger.info("scan_reverse: installed the reverse-scan backward.")


def uninstall() -> None:
  """Restores the stock flip-based backward. Idempotent."""
  global _original_call_backward
  if _original_call_backward is None:
    return
  _scan_mod.ScanAutogradImpl.call_backward = _original_call_backward
  _original_call_backward = None
  logger.info("scan_reverse: restored the stock backward.")


__all__ = ["install", "scan_reverse_op", "uninstall"]
