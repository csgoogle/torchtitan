# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A reverse-direction scan HOP, and the backward-pass rewrite that uses it.

`torch.ops.higher_order.scan` has no `reverse` argument, so
`ScanAutogradImpl.call_backward` emulates the reversed scan its own docstring
asks for by calling `torch.flip` on the saved forward intermediates. On TPU
that flip is a materialised copy of the whole residual stack: at llama3 8B,
seq 8192, local batch 4 it is a second `bf16[31,4,8192,4096]` buffer, 7.75 GiB,
28% of the arena, holding nothing that is not already resident. Upstream knows
-- the call site carries `# TODO: torch.flip copies the tensor, we should
optimize it away`.

torch_tpu can lower a genuine reverse scan (`ScanDirection.kReverse`) and
writes its `ys` aligned to the *input* index rather than to iteration order, so
the input flip and the output flip cancel and both can be dropped. Nothing in
the stack asks for that lowering today, which is what this module fixes.

Adding `reverse` to the existing HOP is not available: `ScanOp.__call__` and
`ScanOp.gen_schema` share a fixed four-argument signature that nine dispatch
registrations are written against. So this is a sibling HOP instead.
"""

import contextlib
import importlib
import itertools
import logging
import os
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.scan import (
    ScanOp,
    _extract_carry_and_out,
    generic_scan,
    stack_y,
    trace_scan,
)
from torch._higher_order_ops.utils import autograd_not_implemented, first_slice_copy
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.overrides import TorchFunctionMode

logger = logging.getLogger(__name__)

# `import torch._higher_order_ops.scan as m` would bind the re-exported `scan`
# *function*, not the module -- the package's `__init__` shadows the submodule
# name. Ask the import machinery for the module directly.
_scan_mod = importlib.import_module("torch._higher_order_ops.scan")


class ScanReverseOp(ScanOp):
    """`scan`, iterating from the last index to the first.

    Identical to `torch.ops.higher_order.scan` in signature, in carry
    semantics, and in output shapes. The two differences: the body runs on
    index `N-1` first and index `0` last, and `ys[i]` holds the output of the
    step that consumed `xs[i]` -- indexed by input position, not by iteration
    order. That is the JAX convention, and it is what the TPU lowering already
    does; it is also what makes the flips cancel.

    No autograd registration. This op exists to *be* the backward of `scan`,
    and is only ever called under `torch.no_grad`-equivalent conditions from
    `ScanAutogradImpl.call_backward`. Calling it from user code inside a graph
    that needs to differentiate through it will fail loudly at the autograd
    key, which is the correct outcome rather than a silent wrong gradient.
    """

    def __init__(self):
        # Skip ScanOp.__init__ -- it hardcodes the "scan" name.
        torch._ops.HigherOrderOperator.__init__(self, "scan_reverse")


scan_reverse_op = ScanReverseOp()


@scan_reverse_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def _scan_reverse_dense(combine_fn, init, xs, additional_inputs):
    """Reference implementation: flip, scan forward, flip back.

    Deliberately the naive construction. This is the CPU path, and its whole
    job is to be an oracle the TPU lowering can be tested against -- so it must
    not share any machinery with the thing under test. The copies it makes are
    the copies this module exists to remove on TPU; on CPU they are irrelevant.
    """
    flipped = [torch.flip(x, (0,)) for x in xs]
    outs = generic_scan(combine_fn, init, flipped, additional_inputs=additional_inputs)
    carry, ys = _extract_carry_and_out(list(outs), len(init))
    return (*carry, *[torch.flip(y, (0,)) for y in ys])


# No differentiation through the reverse scan. It exists to *be* the backward
# of `scan`, so it is always reached under no-grad; `deferred_error` makes a
# caller who does try to differentiate it fail with a clear message rather than
# receive a silently wrong gradient.
scan_reverse_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(scan_reverse_op, deferred_error=True)
)


@scan_reverse_op.py_impl(ProxyTorchDispatchMode)
def _scan_reverse_proxy(mode, combine_fn, init, xs, additional_inputs):
    return trace_scan(mode, scan_reverse_op, combine_fn, init, xs, additional_inputs)


@scan_reverse_op.py_impl(FakeTensorMode)
def _scan_reverse_fake(mode, combine_fn, init, xs, additional_inputs):
    # Direction does not affect shapes or dtypes, so this is `scan`'s rule
    # verbatim. `first_slice_copy` is still the right sample slice: every
    # slice of `xs` has the same shape, the reverse scan just visits them in
    # the other order.
    with mode:
        scan_length = xs[0].shape[0]
        carry, outputs = _extract_carry_and_out(
            combine_fn(*init, *[first_slice_copy(inp) for inp in xs], *additional_inputs),
            len(init),
        )
        return (*carry, *(stack_y(t, scan_length) for t in outputs))


@scan_reverse_op.py_functionalize_impl
def _scan_reverse_functionalize(ctx, combine_fn, init, xs, additional_inputs):
    from torch._higher_order_ops.utils import (
        _check_alias_and_mutation,
        _maybe_run_with_interpreter,
    )

    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_init = ctx.unwrap_tensors(init)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    with ctx.redispatch_to_next():
        functional_combine_fn = ctx.functionalize(_maybe_run_with_interpreter(combine_fn))
        sample_inputs = list(
            itertools.chain(
                unwrapped_init,
                [first_slice_copy(inp) for inp in unwrapped_xs],
                unwrapped_additional_inputs,
            )
        )
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        _check_alias_and_mutation(
            combine_fn, sample_inputs, "scan_reverse", pre_dispatch
        )
        ret = scan_reverse_op(
            functional_combine_fn,
            unwrapped_init,
            unwrapped_xs,
            unwrapped_additional_inputs,
        )
    return ctx.wrap_tensors(ret)


def _register_tpu_lowering() -> bool:
    """Point the device keys at the `kReverse` lowering the forward scan uses.

    Deliberately the fork's `_handle_scan_impl_fixed`, not torch_tpu's
    `_handle_scan_impl`. The upstream one passes the carry inits and additional
    inputs to `fx_to_mlir` as live tpu tensors, and in a real model those are
    pending eager values, so it dies with "argument tensor has deferred ops"
    (`torch_tpu_fix.py`, Bug A). The forward scan on this lane has been
    running on the fixed version all along; the reverse must use the same one
    or it is testing a lowering the lane does not use.
    """
    try:
        from torchtitan.experiments.tpu.scan.torch_tpu_fix import (
            _handle_scan_impl_fixed,
        )
    except ImportError:
        return False

    def tpu_scan_reverse(combine_fn, init, xs, additional_inputs):
        return _handle_scan_impl_fixed(
            combine_fn, init, xs, additional_inputs, reverse=True
        )

    registered = False
    for key in (DispatchKey.XLA, DispatchKey.PrivateUse1, DispatchKey.Meta):
        try:
            scan_reverse_op.py_impl(key)(tpu_scan_reverse)
            registered = True
        except RuntimeError:
            # Already registered; this module was imported twice.
            registered = True
    return registered


_TPU_LOWERING_REGISTERED = _register_tpu_lowering()


class _ElideDim0Flip(TorchFunctionMode):
    """Turns the two `torch.flip(x, (0,))` calls in `call_backward` into no-ops.

    Rewriting `call_backward` would mean vendoring ninety lines of torch
    internals into this repo and re-vendoring them on every torch bump. The
    flips are the only part that has to change, so they are what gets
    intercepted.

    Eliding by shape alone would also silence a `flip` the scanned body itself
    performs, so the mode runs in two phases. Before the scan call it elides
    only tensors it was handed by identity -- the flat `bw_xs`, which the
    caller computes independently from the same `ScanAutogradImpl` fields.
    Anything else traced during `materialize_as_graph` passes through. After
    the scan call, tracing is over and the only remaining flips are the ones
    applied to `grad_xs`, so those are elided unconditionally.

    The mode also disarms itself for the duration of the scan call. Lowering
    the scan runs a lot of code -- torch_tpu traces the body and compiles it to
    MLIR underneath the dispatch -- and none of it is `call_backward`'s flips.
    Eliding a `flip` in there would corrupt the body graph rather than save a
    copy.
    """

    def __init__(self, expected_inputs: list[Any]):
        super().__init__()
        self._expected_ids = {id(t) for t in expected_inputs}
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
            if id(args[0]) in self._expected_ids:
                self.elided_inputs += 1
                return args[0]
        return func(*args, **kwargs)


_ORIGINAL_CALL_BACKWARD = None


def install_scan_reverse_backward() -> None:
    """Makes `scan`'s backward a real reverse scan instead of a flipped one.

    Idempotent. Import-order-independent: it patches
    `ScanAutogradImpl.call_backward`, which is resolved per call.

    The rewrite is exact, not an approximation. `scan_reverse_op` visits the
    same steps in the same order the flipped forward scan did, and writes each
    `grad_x` to the input index the flipped output would have been permuted
    back to -- so both flips are redundant, and removing them changes the
    values by nothing at all. What it changes is that the residual stack is
    read in place instead of being copied.
    """
    global _ORIGINAL_CALL_BACKWARD
    if _ORIGINAL_CALL_BACKWARD is not None:
        return
    if not _TPU_LOWERING_REGISTERED:
        raise RuntimeError(
            "scan_reverse has no device lowering: torch_tpu_fix is not "
            "importable. Without it the reverse scan would fall back to the "
            "flip-based dense path, which is what this patch exists to avoid."
        )

    impl_cls = _scan_mod.ScanAutogradImpl
    _ORIGINAL_CALL_BACKWARD = impl_cls.call_backward

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
                out = _ORIGINAL_CALL_BACKWARD(self, *grad_fw_outputs)
        finally:
            _scan_mod.scan_op = original_scan_op

        if not mode.after_scan:
            raise RuntimeError(
                "scan_reverse: call_backward returned without dispatching a "
                "scan; the upstream implementation has changed shape and this "
                "patch no longer applies."
            )
        n_expected_in = len(mode._expected_ids)
        if mode.elided_inputs != n_expected_in:
            raise RuntimeError(
                "scan_reverse: elided "
                f"{mode.elided_inputs} of {n_expected_in} input flips. Some "
                "backward input is still being copied, so the reverse scan "
                "would read it in the wrong order."
            )
        if mode.elided_outputs == 0:
            raise RuntimeError(
                "scan_reverse: no output flip was elided; grad_xs would be "
                "returned reversed."
            )
        logger.info(
            "scan_reverse: backward is a native reverse scan (%d input flip(s)"
            " and %d output flip(s) removed).",
            mode.elided_inputs,
            mode.elided_outputs,
        )
        return out

    impl_cls.call_backward = call_backward
    logger.info("scan_reverse: installed the reverse-scan backward.")


def uninstall_scan_reverse_backward() -> None:
    """Restores the stock flip-based backward. For tests."""
    global _ORIGINAL_CALL_BACKWARD
    if _ORIGINAL_CALL_BACKWARD is None:
        return
    _scan_mod.ScanAutogradImpl.call_backward = _ORIGINAL_CALL_BACKWARD
    _ORIGINAL_CALL_BACKWARD = None


@contextlib.contextmanager
def scan_reverse_backward():
    """Scoped form of `install_scan_reverse_backward`, for tests."""
    already_installed = _ORIGINAL_CALL_BACKWARD is not None
    if not already_installed:
        install_scan_reverse_backward()
    try:
        yield
    finally:
        if not already_installed:
            uninstall_scan_reverse_backward()


def enabled() -> bool:
    """Whether the rewrite is switched on. Off unless asked for."""
    return os.environ.get("TPU_SCAN_REVERSE_BACKWARD", "0") == "1"
