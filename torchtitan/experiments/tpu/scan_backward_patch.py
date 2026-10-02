"""Make `scan`'s backward stop copying every saved activation.

`ScanAutogradImpl.call_backward` reverses the loop in three steps:

  1. `torch.flip` on the saved activation stacks
  2. `scan_op` forwards over the flipped copies
  3. `torch.flip` on the resulting gradient stacks

Step 1 copies every activation the forward saved, and on TPU costs twice that
again, because a reversed tensor needs a new memory layout. At any model size
where the saved stacks are the bulk of memory, that copy is what runs the
machine out of it.

This file replaces `torch.flip` and `scan_op` for the duration of one
`call_backward`:

  1. `torch.flip` returns the stack unchanged
  2. `scan_op` scans over a descending index instead; each step reads its own
     row with `index_select` and writes its gradient back with `index_copy`
  3. `torch.flip` returns the gradients unchanged -- step 2 already wrote every
     row in its final position

Same rows in the same order, no copy. The rest of `call_backward` is torch's
code running unmodified, so most torch updates need nothing here. One that
changes how the backward reverses raises instead of falling back to copying.

Assumes one thread per process: the two replacements are module-level globals
in torch's `scan.py`, so a concurrent `call_backward` would see them.

Delete this file once the fix lands upstream: google-pytorch/torchtitan#396.
"""

import contextlib
import enum
import inspect
import os
import sys

import torch
from torch._higher_order_ops.scan import ScanAutogradImpl, scan_op
from torch._higher_order_ops.utils import first_slice_copy, materialize_as_graph
from torch.fx.traceback import annotate
import torch.utils._pytree as pytree

from torchtitan.tools.logging import logger


_UPSTREAM_CALL_BACKWARD = ScanAutogradImpl.call_backward

# The module whose globals get swapped below. It has to come from `sys.modules`:
# the name `torch._higher_order_ops.scan` resolves to a function, not the file.
_SCAN_MODULE = sys.modules["torch._higher_order_ops.scan"]


# --- Installing ------------------------------------------------------------


def apply_scan_reverse_backward_patch() -> bool:
    """Installs the patch. Returns whether it took."""
    if getattr(ScanAutogradImpl.call_backward, "_torchtitan_patched", False):
        return True
    if ScanAutogradImpl.call_backward is not _UPSTREAM_CALL_BACKWARD:
        logger.warning(
            "Not patching scan's backward: something else has already replaced"
            " ScanAutogradImpl.call_backward."
        )
        return False
    if not _upstream_still_reverses_by_flip():
        logger.warning(
            "Not patching scan's backward: torch's"
            " ScanAutogradImpl.call_backward no longer reverses with"
            " torch.flip. If the upstream fix has landed, delete"
            " torchtitan/experiments/tpu/scan_backward_patch.py."
        )
        return False
    _call_backward_reverse_indexed._torchtitan_patched = True
    ScanAutogradImpl.call_backward = _call_backward_reverse_indexed
    logger.info(
        "Patched torch._higher_order_ops.scan.ScanAutogradImpl.call_backward"
        " to reverse by index instead of torch.flip"
    )
    return True


def _upstream_still_reverses_by_flip() -> bool:
    """The startup check, so the usual break is caught before any work.

    The likely break is the upstream fix landing, which shows up as the flips
    leaving the source. Catching it here costs a startup error instead of a
    compile's worth of work followed by one.
    """
    try:
        src = inspect.getsource(_UPSTREAM_CALL_BACKWARD)
    except OSError:
        return True  # No source to read; the runtime check still applies.
    return "torch.flip" in src and "scan_op(" in src


# --- The one call we rewrite -----------------------------------------------


def _call_backward_reverse_indexed(self, *grad_fw_outputs):
    state = _ReverseByIndex()
    with state.installed():
        grads = _UPSTREAM_CALL_BACKWARD(self, *grad_fw_outputs)
    if not state.reversed_by_index:
        raise RuntimeError(
            "scan's backward no longer reverses the way"
            " torchtitan/experiments/tpu/scan_backward_patch.py expects"
            f" (torch {torch.__version__}), so it was not rewritten to reverse"
            " by index and may be reversing by copy instead, which roughly"
            " doubles peak memory. Refresh that file against the new"
            " ScanAutogradImpl.call_backward, or delete it if the upstream fix"
            " has landed."
        )
    return grads


# --- The two stand-ins -----------------------------------------------------
#
# Two globals get swapped rather than `call_backward` forked, because upstream
# edits that method every few weeks and a fork silently reverts their fixes.
#
# The cost of swapping is that `scan.py` labels the graph nodes it emits with
# whatever `scan_op` names at the time, so our own scan has to be launched with
# torch's name restored -- see `_paused`. Without that, downstream passes look
# for `torch.ops.higher_order.scan` and do not find it.


class _ReverseByIndex:
    """Rewrites one `call_backward` while it runs.

    It assumes two things about torch: the backward reverses with
    `torch.flip(x, (0,))`, and those same tensors are what it hands to
    `scan_op`. Both are checked as the call runs, so any other upstream edit is
    absorbed. An edit that breaks either leaves `reversed_by_index` false, and
    the caller raises rather than return a gradient computed from stacks this
    claimed to have flipped and did not.
    """

    def __init__(self):
        self.skipped: dict[int, torch.Tensor] = {}
        self.written_in_order: dict[int, torch.Tensor] = {}
        self.reversed_by_index = False
        self._real: tuple = ()

    @contextlib.contextmanager
    def installed(self):
        """Swaps the two names for the dynamic extent of one call.

        Both are globals of `scan.py`, so nothing outside that file sees them
        change. Each swap saves and restores what it found, so a nested
        `call_backward` unwinds to the right thing.
        """
        self._real = (_SCAN_MODULE.torch, _SCAN_MODULE.scan_op)
        _SCAN_MODULE.torch = _TorchWithoutFlip(self)
        _SCAN_MODULE.scan_op = _ScanOpWithOneCallRewritten(self)
        try:
            yield
        finally:
            _SCAN_MODULE.torch, _SCAN_MODULE.scan_op = self._real

    @contextlib.contextmanager
    def _paused(self):
        """Puts torch's own two names back while a scan of ours is traced.

        `scan.py`'s proxy handler labels the node it emits with whatever
        `scan_op` names at that moment, so a scan launched with the stand-in
        still installed comes out labelled with the stand-in -- and then every
        pass that looks for `torch.ops.higher_order.scan` misses it. The
        activation-offloading pass is one, and it silently offloads nothing.
        """
        ours = (_SCAN_MODULE.torch, _SCAN_MODULE.scan_op)
        _SCAN_MODULE.torch, _SCAN_MODULE.scan_op = self._real
        try:
            yield
        finally:
            _SCAN_MODULE.torch, _SCAN_MODULE.scan_op = ours

    def flip(self, x, dims):
        if not isinstance(x, torch.Tensor) or tuple(dims) != (0,):
            return torch.flip(x, dims)
        if self.reversed_by_index:
            # Only the stacks our own scan wrote come out in order. Anything
            # else gets the flip it asked for, so an added one is not swallowed.
            if id(x) in self.written_in_order:
                return x
            return torch.flip(x, dims)
        # The input flip. Hand back the original and remember it; the
        # `scan_op` just below wants the stack unreversed. Keyed by id, with the
        # tensor as the value, so nothing this is keyed on can be collected and
        # have its id reused underneath us.
        self.skipped[id(x)] = x
        return x

    def scan_op(self, combine_gm, init, xs, additional_inputs=(), **kwargs):
        # `mutated_arg_indices` numbers the step's arguments, and the rewrite
        # renumbers them, so leave a scan that carries any of those alone.
        #
        # Equality, not subset: a skipped flip that is not one of these stacks
        # is one nobody performed, so fall through and let the caller raise.
        if (
            not kwargs.get("mutated_arg_indices")
            and self.skipped
            and set(self.skipped) == {id(x) for x in xs}
        ):
            self.skipped.clear()
            self.reversed_by_index = True
            with self._paused():
                out = _scan_over_descending_index(
                    combine_gm, init, xs, additional_inputs
                )
            # The tail is the gradient stacks. Held by value so an id cannot
            # be reused out from under us. In `GradWrite.YS` mode they are in
            # iteration order, so leave them to upstream's flip.
            self.written_in_order = (
                {}
                if _grad_write_mode() is GradWrite.YS
                else {id(t): t for t in out[len(init) :]}
            )
            return out
        # Some other scan -- a nested one traced out of the step body, say.
        with self._paused():
            return scan_op(combine_gm, init, xs, additional_inputs, **kwargs)


class _TorchWithoutFlip:
    """The `torch` module with `flip` replaced and nothing else touched."""

    def __init__(self, state: _ReverseByIndex):
        self.flip = state.flip

    def __getattr__(self, name):
        return getattr(torch, name)


class _ScanOpWithOneCallRewritten(type(scan_op)):
    """The `scan_op` HOP with one call rewritten and nothing else touched.

    `scan.py` hands `scan_op` to `HopInstance`, which type-checks it, so a
    look-alike is not enough -- this has to be a real HOP. It borrows the real
    one's state rather than building a second one, because a HOP registers its
    name on construction and there can only be one "scan".
    """

    def __init__(self, state: _ReverseByIndex):
        self.__dict__.update(scan_op.__dict__)
        self._rewritten_call = state.scan_op

    def __call__(self, *args, **kwargs):
        return self._rewritten_call(*args, **kwargs)


# --- The scan that replaces the flipped one --------------------------------


class GradWrite(enum.Enum):
    """How `_scan_over_descending_index` stores each step's gradient row.

    INDEX_COPY: carry the gradient stacks and `index_copy` each row into its
      final position. No copy after the loop, but `index_copy` lowers to a
      scatter that XLA expands into a masked dynamic-update-slice indexed by a
      loaded value, which XLA's collective pipeliner cannot move collectives
      through -- so the per-layer cross-slice gradient all-reduce always
      finishes inside its own iteration.
    YS: return each row as a scan output. The rows land at the loop index with
      a plain dynamic-update-slice, which the pipeliner accepts, and torch's
      own flip puts them back in order after the loop, at the cost of one
      reversed copy of every gradient stack.
    YS_REVERSE: as YS, but ask torch_tpu for a reverse loop, whose step i
      reads and writes row `len - 1 - i`. Scanning over an ascending index
      then visits the rows in the same descending order, and each gradient row
      lands in its final position with a plain dynamic-update-slice at
      `len - 1 - i` -- so neither the `index_copy` nor the flip is needed.
      Only torch_tpu honours the request; any other backend would run the loop
      forwards and return the gradient stacks reversed.
    """

    INDEX_COPY = "index_copy"
    YS = "ys"
    YS_REVERSE = "ys_reverse"


# Must match `torch_tpu._internal.compile.scan.SCAN_REVERSE_ANNOTATION`. Spelled
# out rather than imported so this file still imports without torch_tpu.
_SCAN_REVERSE_ANNOTATION = "torch_tpu_scan_reverse"


def _grad_write_mode() -> GradWrite:
    return GradWrite(
        os.environ.get("TORCHTITAN_SCAN_BWD_GRAD_WRITE", GradWrite.YS_REVERSE.value)
    )


def _scan_over_descending_index(combine_gm, init, xs, additional_inputs):
    """`scan(combine_gm, init, xs, reverse=True)` without reversing the inputs.

    The stacks move to `additional_inputs`, which scan passes through whole,
    and the thing scanned over becomes `scan_length` int64s counting down. Each
    step picks its own row out of the stacks, so the loop runs backwards over
    the data while running forwards over the index.

    `combine_gm` returns one row of gradient per step, in iteration order --
    which is backwards. With `GradWrite.INDEX_COPY` the output stacks are
    carried as buffers and each step writes its own row in place, so they come
    out already in order. With `GradWrite.YS` they come out as scan outputs in
    iteration order and the caller's flip reverses them. With
    `GradWrite.YS_REVERSE` the index counts up and the loop runs in reverse,
    which visits the same rows in the same order and writes each output row
    where it belongs.
    """
    n_init = len(init)
    n_additional = len(additional_inputs)
    scan_length = xs[0].shape[0]

    # The flip used to double as a copy, and that copy is what kept the init and
    # the stacks in separate storages. `grad_carry` and `grad_ys` are routinely
    # two views of one broadcast, and scan rejects a combine_fn whose inputs
    # alias. Clone the init, one slice, instead of the stacks, the whole history.
    init = [t.clone() if isinstance(t, torch.Tensor) else t for t in init]

    mode = _grad_write_mode()
    if mode in (GradWrite.YS, GradWrite.YS_REVERSE):
        if mode is GradWrite.YS_REVERSE:
            index = torch.arange(scan_length, dtype=torch.int64, device=xs[0].device)
            # On the nodes, not the graph module: scan re-traces the body into a
            # new module, which keeps node metadata and drops attributes.
            direction = annotate({_SCAN_REVERSE_ANNOTATION: True})
        else:
            index = torch.arange(
                scan_length - 1, -1, -1, dtype=torch.int64, device=xs[0].device
            )
            direction = contextlib.nullcontext()

        def step_ys(*args):
            carry = args[:n_init]
            row = args[n_init].reshape(1)
            rest = args[n_init + 1 :]
            additional, stacks = rest[:n_additional], rest[n_additional:]
            rows = [torch.index_select(s, 0, row).squeeze(0) for s in stacks]
            return list(combine_gm(*carry, *rows, *additional))

        with direction:
            step_gm = materialize_as_graph(
                step_ys,
                (*init, first_slice_copy(index), *additional_inputs, *xs),
            )
            # `(*carry, *stacked_grad_xs)`: the stacks in iteration order for
            # YS, in final order for YS_REVERSE.
            return scan_op(step_gm, init, [index], (*additional_inputs, *xs))

    buffers = [
        torch.zeros(
            (scan_length, *meta.shape), dtype=meta.dtype, device=meta.device
        )
        for meta in _output_metas(combine_gm)[n_init:]
    ]
    n_buffers = len(buffers)

    def step(*args):
        carry = args[:n_init]
        bufs = args[n_init : n_init + n_buffers]
        row = args[n_init + n_buffers].reshape(1)
        rest = args[n_init + n_buffers + 1 :]
        additional, stacks = rest[:n_additional], rest[n_additional:]
        # `index_select` rather than `select` because the index is a traced
        # value, not a Python int; it lowers to a dynamic-slice.
        rows = [torch.index_select(s, 0, row).squeeze(0) for s in stacks]
        out = combine_gm(*carry, *rows, *additional)
        return [
            *out[:n_init],
            *[
                buf.index_copy(0, row, grad.unsqueeze(0))
                for buf, grad in zip(bufs, out[n_init:])
            ],
        ]

    descending = torch.arange(
        scan_length - 1, -1, -1, dtype=torch.int64, device=xs[0].device
    )
    scan_init = [*init, *buffers]
    step_gm = materialize_as_graph(
        step,
        (*scan_init, first_slice_copy(descending), *additional_inputs, *xs),
    )
    # All carry, no ys, so the result is already `(*carry, *stacked_grad_xs)`:
    # the same flat layout upstream's own `scan_op` call returns.
    return scan_op(step_gm, scan_init, [descending], (*additional_inputs, *xs))


def _output_metas(gm: torch.fx.GraphModule) -> list[torch.Tensor]:
    """The fake vals a graph returns, flat, so buffers can be sized from them."""
    (output_node,) = gm.graph.find_nodes(op="output")
    return [node.meta["val"] for node in pytree.tree_leaves(output_node.args[0])]
