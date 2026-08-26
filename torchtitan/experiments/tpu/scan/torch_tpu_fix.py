# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local workaround for a torch_tpu defect that breaks the `scan` HOP on TPU.

TEMPORARY. Delete this file once the fix lands in torch_tpu. It is carried here
so scan-over-layers work is not blocked on an external release. The diagnosis
of each defect, and the lane matrix showing which lanes each one gates, are
recorded below.

Diagnosed against torch_tpu 0.1.1 (source @ 4957e9e5). There are three
independent defects. A and C are worked around here; B is C++-only and gates
lane selection.

BUG A -- FIXED HERE -- torch_tpu/_internal/compile/scan.py:86
--------------------------------------------------------------------------
    body_args = init_list + x_list + extra_list
    body_mlir = export.fx_to_mlir(actual_combine_fn, args=body_args)

`x_list` (the scanned inputs) is correctly built from
`tpu_torch_compile.placeholder(...)`, but the carry inits and the additional
inputs are passed as LIVE tpu tensors. `fx_to_mlir` runs the compiled-mode MLIR
extraction path, which requires non-deferred arguments:

    compiled_mode.cc:180   TT_RET_CHECK(!buffer_ref.is_deferred(), ...)
    compiled_mode.cc:236   ABSL_CHECK(!buffer_ref.is_deferred())   -> SIGABRT

In a real model the carry init and the additional inputs *are* pending eager
values (the embedding output; weights after an optimizer step), so this fires.
Fix: give them the same placeholder treatment `x_list` already gets.

BUG B -- NOT FIXABLE FROM PYTHON -- torch_tpu/_internal/compile/dispatch_scan.cc:296
--------------------------------------------------------------------------
`PyCreateScanOp` calls `DeviceBufferList::CreateDeferred()` directly. Every
other op-creation site in torch_tpu goes through
`internal::CreateDeferredDeviceBufferList()` (device_buffer_utils.cc:182), the
*only* caller of `RecordDeferredOpCreated()` (events_queue.h:50) -- the function
that puts a new deferred node on the global EventsQueue.

The scan's `custom_kernel` node therefore never lands on the queue.
`PrepareTraversals()` builds each traversal's execution order FROM THE QUEUE
(not by DFS over the buffer DAG), so the node never enters a traversal and never
gets `SetAsPendingMaterialization()`. The first consumer's traversal then sees a
still-deferred argument and dies:

    materialize_common.cc:265  "has deferred argument custom_kernel[...].
                                All arguments must be set to pending
                                materialization before creating chained
                                ExecutionTasks"

Consequence for lane selection, measured on one v6e chip of a v6e-8 host:

("+ this fix" is this Python file, i.e. Bugs A + C only. It cannot reach Bug B,
which is why the eager row still fails with B in the right-hand column.)

    lane                                      stock          + this fix
    ---------------------------------------   ------------   ----------------
    torch.compile / TpuBackend ("fullgraph")  SIGABRT (139)  RUNS
    TpuGraphTrainer ("aot_fx_trace")          RuntimeError   RUNS  [1]
    eager (scan_op directly)                  RuntimeError   RuntimeError (B)

    [1] measured RuntimeError (B) when this file was written, against a
        graph-trainer lane that had no scan wiring and fell back to eager
        replay. Once the scan was wired into `parallelize_scan.py` the lane
        stopped replaying eagerly, and the paragraph below gives the mechanism
        for why Bug B should then be unreachable. But that is a prediction, not
        a measurement: every graph-trainer run on record (8B/v6e-8 at 29.51%
        MFU, and all the 70B/tpu7x-32 runs) used a container whose torch_tpu
        already carried the C++ Bug B fix, so no run exists with scan wired and
        the fix absent. Until one does, reproduce with the fix applied.

Bug B only bites lanes that replay ops eagerly through torch_tpu's deferred
executor. Both compiled lanes build the scan inside compiled mode, where every
buffer is a placeholder, so the unqueued node is never materialized.

BUG C -- WORKED AROUND HERE (symptom side) -- rank-0 `ys` stack to the wrong rank
--------------------------------------------------------------------------
A combine_fn output `y` of rank 0 must stack to `[seq_len]`. That is what
PyTorch's scan meta says and what `dummy_ys` (scan.py:83) says -- verified at
trace time: `create_scan_op` returns a `(4,)` int64 tensor. But the compiled
`stablehlo.while` returns `(4, 1)` at runtime: the C++ lowering normalizes the
loop-carried accumulator slice to rank >= 1 and never squeezes it back. Output
reconstruction then compares the two and dies:

    export.py:295  force_strides(...)
    RuntimeError: target strides must have the same number of dimensions as
                  the tensor, got 2 and 1

This is not exotic -- it fires on any transformer. SDPA
(`aten._scaled_dot_product_fused_attention_overrideable`) returns `philox_seed`
and `philox_offset` as rank-0 int64 tensors at output indices 6 and 7; the HOP
partitioner saves both for backward, and scan stacks every saved intermediate.
So a single attention layer in a scan body is enough.

The real fix is in the C++ scan lowering. Until then we repair the symptom in
`_reconstruct_fx_outputs`: an output whose dtype and element count match the
recorded layout but whose shape does not is reshaped to the recorded shape.
That is a pure rank disagreement and always safe; every other mismatch is
passed through unchanged. See `_install_rank0_ys_reshape` below.

=> With this fix, scan-over-layers works today on both compiled lanes:
   `--tpu-config.compile-mode=fullgraph` (the regular TPUTrainer, 27.40% MFU on
   8B/v6e-8) and `--tpu-config.use-graph-trainer` with `aot_fx_trace` (29.51%,
   the frontier). Only the eager lane is still blocked on Bug B, and nothing
   needs it.

   Unrelated and still open: on tpu7x the graph-trainer lane can abort inside
   torch_tpu's `TraverseAndCompile` rather than raising. That is a v7x compile
   failure, not Bug B. It is hard to attribute because `TraverseAndCompile`
   `CHECK`s on several statuses that its twin `ExtractMlirFromGraph`
   propagates, so the real error is replaced by a generic abort.
"""

from typing import Any

import torch
import torch.export
import torch.utils._pytree as pytree

from torchtitan.tools.logging import logger

_APPLIED = False
_TAG = "torch_tpu_scan_fix/bug-a-placeholder-inits"

# The defect's signature in upstream's `_handle_scan_impl`, normalized by
# `_normalize_source` below.
_BUG_A_SIGNATURE = "body_args=init_list+x_list+extra_list"


def _normalize_source(src: str) -> str:
  """Strips whitespace and grouping parentheses before signature matching.

  Whitespace alone is not enough. A formatter wrapping the `body_args` line --
  the usual consequence of the line growing past the column limit -- produces
  `body_args = (\\n init_list\\n + x_list\\n + extra_list\\n )`, which survives
  whitespace removal as `body_args=(init_list+x_list+extra_list)` and no longer
  matches. That reads as "upstream fixed it", suppresses the patch, and leaves
  the scan body built with the defect: wrong results on TPU, no error, no log.
  Dropping parentheses too costs nothing -- no plausible *fixed* spelling
  becomes this string once its parentheses are removed.
  """
  return "".join(src.split()).replace("(", "").replace(")", "")


def _handle_scan_impl_fixed(
    combine_fn, init, xs, additional_inputs, reverse: bool = False
) -> Any:
  """`torch_tpu._internal.compile.scan._handle_scan_impl` with Bug A fixed.

  Kept a line-for-line mirror of upstream's body so it is trivial to diff and to
  drop once the upstream fix lands. The only behavioral delta is the
  `clean_inits` / `clean_extras` block, marked below. Two non-behavioral deltas
  are unavoidable and are the whole of the rest of the diff: this signature and
  docstring, and the two imports below, which upstream does at module scope and
  this file must defer because it is imported whether or not torch_tpu is.
  """
  # pylint: disable=g-import-not-at-top,protected-access
  from torch_tpu._internal.compile import tpu_torch_compile
  from torch_tpu._internal.export import export

  # Normalize inputs to lists.
  init_list, _ = pytree.tree_flatten(init)
  xs_list, _ = pytree.tree_flatten(xs)
  extra_list, _ = pytree.tree_flatten(additional_inputs)

  # Construct sample args for body function.
  x_list = [
      tpu_torch_compile.placeholder(
          list(single_xs.shape)[1:], single_xs.dtype, single_xs.requires_grad
      )
      for single_xs in xs_list
  ]

  # Resolve the actual callable body function.
  actual_combine_fn = combine_fn
  if isinstance(combine_fn, torch.export.ExportedProgram):
    actual_combine_fn = combine_fn.module

  # Perform a meta-run to infer the output shapes and dtypes.
  with torch._subclasses.fake_tensor.FakeTensorMode(allow_non_fake_inputs=True):

    def to_fake(tensors):
      return [
          tpu_torch_compile.placeholder_like(t)
          if isinstance(t, torch.Tensor)
          else t
          for t in tensors
      ]

    fake_inits = to_fake(init_list)
    fake_xs_heads = to_fake(x_list)
    fake_extras = to_fake(extra_list)

    fake_result = actual_combine_fn(*fake_inits, *fake_xs_heads, *fake_extras)
    fake_result_flat, _ = pytree.tree_flatten(fake_result)

    dummy_ys = []
    seq_len = xs_list[0].shape[0]
    num_carries = len(init_list)
    for y in fake_result_flat[num_carries:]:
      ys_shape = [seq_len] + list(y.shape)
      dummy_ys.append(torch.zeros(ys_shape, dtype=y.dtype, device="meta"))

  # ---- BUG A FIX ---------------------------------------------------------
  # Upstream: `body_args = init_list + x_list + extra_list`, which hands LIVE
  # (possibly deferred) tpu tensors to the compiled-mode MLIR extractor.
  # Trace the body against placeholders instead, exactly as x_list already does.
  clean_inits = [
      tpu_torch_compile.placeholder(list(t.shape), t.dtype, t.requires_grad)
      for t in init_list
  ]
  clean_extras = [
      tpu_torch_compile.placeholder(list(t.shape), t.dtype, t.requires_grad)
      for t in extra_list
  ]
  body_args = clean_inits + x_list + clean_extras
  # ------------------------------------------------------------------------

  # Compile the body sub-graph to MLIR.
  body_mlir = export.fx_to_mlir(actual_combine_fn, args=body_args)

  # Call the C++ binding to create a deferred scan operation.
  scan_direction = (
      tpu_torch_compile.ScanDirection.kReverse
      if reverse
      else tpu_torch_compile.ScanDirection.kForward
  )
  num_scan_inputs = len(xs_list)
  inputs = xs_list + extra_list
  return tpu_torch_compile.create_scan_op(
      inits=init_list,
      inputs=inputs,
      body_module=body_mlir.module,
      scan_direction=scan_direction,
      dummy_ys=dummy_ys,
      num_scan_inputs=num_scan_inputs,
  )


def _install_rank0_ys_reshape() -> bool:
  """Installs the BUG C workaround around `export._reconstruct_fx_outputs`.

  See the BUG C block in the module docstring. Narrow by construction: the
  wrapper only touches an output whose dtype and element count already agree
  with the layout the FX graph recorded and whose *shape* does not -- i.e. a
  pure rank/reshape disagreement, which is always safe to resolve by
  reshaping. Anything else is passed through untouched, so a genuine shape
  bug elsewhere in torch_tpu still surfaces as the same error it does today.
  """
  # pylint: disable=g-import-not-at-top,protected-access
  from torch_tpu._internal.export import export

  original = export._reconstruct_fx_outputs
  if getattr(original, "_scan_rank0_fix", False):
    return True

  def reconstruct_fx_outputs_reshaped(
      inputs, deduped_outputs, expected_layouts, output_map, spec,
      dynamic_outputs=None,
  ):
    patched = None
    for idx, (tensor, layout) in enumerate(
        zip(deduped_outputs, expected_layouts)
    ):
      if (
          tensor.shape != layout.shape
          and tensor.dtype == layout.dtype
          and tensor.numel() == layout.shape.numel()
      ):
        if patched is None:
          patched = list(deduped_outputs)
        patched[idx] = tensor.reshape(layout.shape)
    return original(
        inputs,
        deduped_outputs if patched is None else patched,
        expected_layouts,
        output_map,
        spec,
        dynamic_outputs,
    )

  reconstruct_fx_outputs_reshaped._scan_rank0_fix = True
  export._reconstruct_fx_outputs = reconstruct_fx_outputs_reshaped
  return True


def _upstream_still_has_bug_a(scan_module: Any) -> bool | None:
  """Whether the installed torch_tpu still builds the scan body with Bug A.

  `_handle_scan_impl_fixed` is a line-for-line mirror of one specific upstream
  revision, so forcing it over a torch_tpu that has moved on is worse than not
  patching at all. Checks for the defect's signature -- the `body_args` line
  that concatenates the live `init_list` / `extra_list` with the placeholdered
  `x_list` -- under `_normalize_source`, so reformatting alone does not trip it.

  Returns True if the defect is present, False if upstream has changed (fixed
  or restructured), and None if the question cannot be answered, in which case
  the caller should patch as before.
  """
  import inspect  # pylint: disable=g-import-not-at-top

  impl = getattr(scan_module, "_handle_scan_impl", None)
  if impl is None:
    return None
  try:
    src = inspect.getsource(impl)
  except (OSError, TypeError):
    return None
  return _BUG_A_SIGNATURE in _normalize_source(src)


def use_torch_tpu_scan_fix() -> bool:
  """Re-registers the fixed scan impl over torch_tpu's. Idempotent; safe no-op.

  Returns False only when torch_tpu is absent or the re-registration itself
  fails; True otherwise. True does NOT imply Bug A's patch is live: when
  upstream's `_handle_scan_impl` no longer matches the revision this file
  mirrors, the patch is deliberately skipped and this still returns True, on the
  reasoning that a changed upstream is more likely fixed than broken and
  overwriting it blindly would be worse. That path logs a warning; a caller that
  needs to know whether the patch is actually installed must read the log or
  call `_upstream_still_has_bug_a` itself, not branch on this return value.
  """
  global _APPLIED
  if _APPLIED:
    return True

  try:
    import torch_tpu._internal.compile.scan as _tt_scan  # noqa: F401  pylint: disable=g-import-not-at-top,unused-import
  except ImportError:
    return False

  # Bug C lives in a different file and its wrapper is a no-op against a
  # torch_tpu that does not have the defect, so it is installed either way.
  _install_rank0_ys_reshape()

  if _upstream_still_has_bug_a(_tt_scan) is False:
    logger.warning(
        "torch_tpu scan fix: upstream's _handle_scan_impl no longer matches the"
        " revision this workaround mirrors, so Bug A's patch is NOT installed."
        " Re-diagnose against the installed torch_tpu and delete"
        " torch_tpu_fix.py if the fix has landed."
    )
    _APPLIED = True
    return True

  scan_hop = torch.ops.higher_order.scan
  keys = (
      torch._C.DispatchKey.XLA,
      torch._C.DispatchKey.PrivateUse1,
      torch._C.DispatchKey.Meta,
  )

  installed = 0
  for key in keys:
    try:
      # py_impl refuses to overwrite an existing entry, so drop it first.
      # `py_kernels` is a private dict whose type is not part of any API
      # contract; a torch release is free to make it a mapping proxy or drop it
      # entirely. Either way the `py_impl` below then raises and is reported --
      # this branch only decides whether the *pre-clear* is possible.
      scan_hop.py_kernels.pop(key, None)
    except Exception:  # pylint: disable=broad-except
      pass
    try:
      scan_hop.py_impl(key)(_handle_scan_impl_fixed)
      installed += 1
    except Exception as e:  # pylint: disable=broad-except
      # A failure on one dispatch key must not lose the other two: the lane only
      # needs PrivateUse1 at runtime, and Meta only under fake-tensor tracing.
      # `installed == 0` below is what turns "all three failed" into a signal.
      logger.warning("torch_tpu scan fix: could not register on %s: %s", key, e)

  try:
    # Stale negative-cache entries would keep routing to the old impl. The
    # attribute is private and has changed shape across torch releases; if it is
    # gone the cache it guarded is gone too, so there is nothing to recover from.
    scan_hop._dispatch_cache.clear()  # pylint: disable=protected-access
  except Exception:  # pylint: disable=broad-except
    pass

  if installed == 0:
    return False

  _APPLIED = True
  logger.info(
      "Applied TEMPORARY torch_tpu scan workaround '%s' to %d/%d dispatch keys."
      " Remove once torch_tpu ships the fix (see torch_tpu_fix.py).",
      _TAG,
      installed,
      len(keys),
  )
  return True
