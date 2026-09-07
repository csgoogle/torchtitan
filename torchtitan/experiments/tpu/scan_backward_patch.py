"""Reverse `scan`'s autograd backward by index instead of by copy.

`scan_op`, the HOP, has no `reverse=` parameter, so
`ScanAutogradImpl.call_backward` in `torch/_higher_order_ops/scan.py` drives its
backwards loop forwards over `torch.flip`ped stacks. That flip is a full
materialisation of every saved intermediate, and on XLA a second one on top,
because flipping also changes the tiling. At llama3-70B batch 4 the input-side
flip alone is `bf16[80, 4, 8192, 8192]` -- 40 GiB, twice -- and it is the whole
of the OOM that stops the scan arm running.

This installs a `call_backward` that scans over a descending index instead:
the stacks go in as loop-invariant `additional_inputs` and each step slices out
its own row, and the stacked gradients are carried as buffers each step writes
into, so neither the inputs nor the outputs are ever reversed. Measured at 70B
batch 4 on tpu7x-32: OOM at 98.43 G -> 61.24 GiB and 42.76 % MFU, with the
`reverse` op count in the HLO going 15 -> 0 and loss and grad norm unchanged
bit for bit.

This belongs upstream in PyTorch and is written to go there -- the body below
is `call_backward` copied verbatim with one block replaced, so it diffs cleanly
against the original. It lives here because torch is a wheel in our images, not
a build, and waiting for an upstream release would block the 70B runs. Delete
this file once the upstream fix lands; `_UPSTREAM_DIGEST` below will tell you.
"""

import hashlib
import inspect

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.scan import (
    ScanAutogradImpl,
    ScanForwardIntermediatesHandlingPolicy,
    scan_op,
)
from torch._higher_order_ops.utils import (
    fill_none_with_masks,
    filter_with_masks,
    first_slice_copy,
    materialize_as_graph,
    split_into_chunks,
)

from torchtitan.tools.logging import logger


# SHA-256 of the upstream `call_backward` this was forked from, whitespace
# normalised. The patch replaces a block in the middle of the method, so it has
# to be shipped as a whole-method copy; if upstream changes the method, the copy
# is stale and silently reverts unrelated fixes. Refuse to install in that case
# rather than guess -- the run still works, it just pays for the flips again,
# which is loud in the `memory:` metric.
_UPSTREAM_DIGEST = (
    "1a5c159a10d0a88f9917c727a021df81000a1edbbddd8fd410f0bf43eba57817"
)


def _digest(fn) -> str:
    src = inspect.getsource(fn)
    return hashlib.sha256(
        "\n".join(line.rstrip() for line in src.splitlines()).strip().encode()
    ).hexdigest()


def _output_nodes(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    """The nodes a graph returns, flat, so their fake vals can be read."""
    (output_node,) = gm.graph.find_nodes(op="output")
    return pytree.tree_leaves(output_node.args[0])


def _call_backward_reverse_indexed(self, *grad_fw_outputs):
    """
    Recall that fw_outputs = (*carry, *ys), bw_gm takes in (*fw_intermediates, *grad_carry, *grad_ys)
    and returns (*grad_init, *grad_xs, *grad_additional_inputs)
    The backward is a reversed scan that can be constructed as follows:

      grad_additional_inputs = torch.zeros_like(additional_inputs)
      bw_init = (grad_carry, grad_additional_inputs)
      bw_xs = (fw_intermediates, grad_ys)
      grad_init, grad_additional_inputs, grad_xs = scan(
        combine_fn,
        bw_init,
        bw_xs,
        reverse = True
      )
      , where combine_fn is defined as follows:

       def combine_fn(bw_init, bw_xs):
         grad_carry, grad_additional_inputs = bw_init
         fw_intermediates, grad_y = bw_xs
         nxt_grad_carry, grad_x, nxt_grad_additional_inputs = bw_gm(*fw_intermediates, *grad_carry, *grad_y)
         return (nxt_grad_carry, grad_additional_inputs + nxt_grad_additional_inputs), grad_x

      Note that grad_additional_inputs is accumulated with add, grad_carry is carried over to next iteration and
      grad_x is the ys output, which will be stacked together after the loop and will have the same shape as xs.
    """
    fw_policy = self.forward_intermediates_handling_policies
    saved_intermediates = self.saved_intermediates
    saved_fw_xs = self.saved_fw_xs
    saved_fw_additional_inputs = self.saved_fw_additional_inputs

    n_carry = len(self.init)

    grad_carry, grad_ys = grad_fw_outputs[:n_carry], grad_fw_outputs[n_carry:]
    additional_inputs_tensor_masks = [
        bool(isinstance(t, torch.Tensor)) for t in self.additional_inputs
    ]
    grad_additional_inputs = [
        torch.zeros_like(t)
        for t in filter_with_masks(
            self.additional_inputs, additional_inputs_tensor_masks
        )
    ]

    bw_init = [grad_carry, grad_additional_inputs]
    bw_xs = [
        grad_ys,
        saved_fw_xs,
        saved_intermediates,
    ]
    bw_additional_inputs = saved_fw_additional_inputs

    _, flat_spec = pytree.tree_flatten((bw_init, bw_xs, bw_additional_inputs))

    grad_spec = None

    def bw_single_step_wrapper(*args):
        bw_init, bw_xs, bw_additional_inputs = pytree.tree_unflatten(
            args, flat_spec
        )
        grad_carry, grad_additional_inputs = bw_init
        grad_y, saved_fw_xs, saved_intermediates = bw_xs
        saved_fw_additional_inputs = bw_additional_inputs

        fw_intermediates = []
        xs_it = iter(saved_fw_xs)
        carry_it = iter(saved_intermediates)
        addi_it = iter(saved_fw_additional_inputs)
        for policy in fw_policy:
            if policy in (
                ScanForwardIntermediatesHandlingPolicy.CLONE,
                ScanForwardIntermediatesHandlingPolicy.KEEP,
            ):
                fw_intermediates.append(next(carry_it))
            elif policy == ScanForwardIntermediatesHandlingPolicy.REMOVE_XS:
                fw_intermediates.append(next(xs_it))
            elif (
                policy
                == ScanForwardIntermediatesHandlingPolicy.REMOVE_ADDITIONAL_INPUTS
            ):
                fw_intermediates.append(next(addi_it))
            else:
                raise RuntimeError(f"Unknown policy: {policy}")

        grad_fw_outputs = (*grad_carry, *grad_y)

        flat_out = self.hop_partitioned_graph.bw_gm(
            *fw_intermediates,
            *grad_fw_outputs,
        )

        next_grad_carry, grad_xs, grad_addi = split_into_chunks(
            flat_out,  # type: ignore[arg-type]
            [len(self.init), len(self.xs), len(self.additional_inputs)],
        )

        nonlocal grad_spec
        flat_grads, grad_spec = pytree.tree_flatten(
            (
                next_grad_carry,
                [
                    prev + cur
                    for prev, cur in zip(
                        grad_additional_inputs,
                        filter_with_masks(
                            grad_addi, additional_inputs_tensor_masks
                        ),
                    )
                ],
                grad_xs,
            )
        )
        return flat_grads

    # Walk `bw_xs` backwards without moving it. `scan_op` has no `reverse=`,
    # so the loop can only be driven forwards; flipping the stacks to suit it
    # costs a full copy of every saved intermediate, and on XLA a second copy
    # on top, because the flip also changes the tiling. Instead pass the
    # stacks in as loop-invariant `additional_inputs` and scan over a
    # descending index, leaving each step to slice out row `L-1-i` itself.
    # The only thing scanned over is `scan_length` int64s.
    # The flip used to double as a copy, which is what kept `bw_init` and
    # the stacks in separate storages: an incoming `grad_carry` and an
    # incoming `grad_ys` are routinely two views of one broadcast, and scan
    # rejects a combine_fn whose inputs alias. Break it on the carry, which
    # is one slice, rather than on the stacks, which are the whole history.
    bw_init_flat = [
        t.clone() if isinstance(t, torch.Tensor) else t
        for t in pytree.tree_flatten(bw_init)[0]
    ]
    bw_xs_flat = pytree.tree_flatten(bw_xs)[0]
    bw_additional_inputs_flat = pytree.tree_flatten(bw_additional_inputs)[0]
    n_init = len(bw_init_flat)
    n_additional = len(bw_additional_inputs_flat)
    scan_length = bw_xs_flat[0].shape[0]

    # The same argument applies to the loop's outputs. `scan_op` stacks the
    # step outputs in iteration order, so a backwards loop produces `grad_xs`
    # backwards and something has to reverse it -- a second flip, this time
    # over the stacked parameter gradients. Carry the output buffers instead
    # and have each step write its row at `L-1-i`, which the loop can do in
    # place. Nothing is produced as a stacked output, so nothing needs
    # reversing afterwards.
    grad_xs_metas = [
        node.meta["val"]
        for node in _output_nodes(self.hop_partitioned_graph.bw_gm)[
            len(self.init) : len(self.init) + len(self.xs)
        ]
    ]
    grad_xs_buffers = [
        torch.zeros(
            (scan_length, *meta.shape), dtype=meta.dtype, device=meta.device
        )
        for meta in grad_xs_metas
    ]
    n_grad_xs = len(grad_xs_buffers)

    def bw_reverse_indexed_wrapper(*args):
        init_flat = args[:n_init]
        buffers = args[n_init : n_init + n_grad_xs]
        row = args[n_init + n_grad_xs].reshape(1)
        rest = args[n_init + n_grad_xs + 1 :]
        additional_flat, stacks = rest[:n_additional], rest[n_additional:]
        # `index_select` rather than `select` because the index is a traced
        # value, not a Python int; it lowers to a dynamic-slice.
        xs_flat = [
            torch.index_select(stack, 0, row).squeeze(0) for stack in stacks
        ]
        step = bw_single_step_wrapper(*init_flat, *xs_flat, *additional_flat)
        carry_out, grad_xs = step[:n_init], step[n_init:]
        return [
            *carry_out,
            *[
                buf.index_copy(0, row, grad.unsqueeze(0))
                for buf, grad in zip(buffers, grad_xs)
            ],
        ]

    reverse_index = torch.arange(
        scan_length - 1,
        -1,
        -1,
        dtype=torch.int64,
        device=bw_xs_flat[0].device,
    )
    bw_scan_init = [*bw_init_flat, *grad_xs_buffers]
    bw_single_step_gm = materialize_as_graph(
        bw_reverse_indexed_wrapper,
        (
            *bw_scan_init,
            first_slice_copy(reverse_index),
            *bw_additional_inputs_flat,
            *bw_xs_flat,
        ),
    )

    # Everything now comes back as carry, in the order `grad_spec` expects:
    # grad_init, then grad_additional_inputs, then grad_xs.
    flat_grads = scan_op(
        bw_single_step_gm,
        bw_scan_init,
        [reverse_index],
        (*bw_additional_inputs_flat, *bw_xs_flat),
    )
    if grad_spec is None:
        raise AssertionError("grad_spec must not be None after scan_op")
    grad_init, grad_additional_inputs, grad_xs = pytree.tree_unflatten(
        flat_grads, grad_spec
    )
    return (
        *grad_init,
        *grad_xs,
        *fill_none_with_masks(
            grad_additional_inputs, additional_inputs_tensor_masks
        ),
    )

def apply_scan_reverse_backward_patch() -> bool:
    """Installs the patch. Returns whether it took."""
    if getattr(ScanAutogradImpl.call_backward, "_torchtitan_patched", False):
        return True

    found = _digest(ScanAutogradImpl.call_backward)
    if found != _UPSTREAM_DIGEST:
        logger.warning(
            "Not patching scan's backward: torch's ScanAutogradImpl."
            "call_backward has changed (digest %s, expected %s). Either the"
            " upstream fix landed, in which case delete"
            " torchtitan/experiments/tpu/scan_backward_patch.py, or it moved"
            " under us and the copy needs refreshing. Scan will run, but its"
            " backward will reverse by torch.flip and the peak memory roughly"
            " doubles.",
            found[:12],
            _UPSTREAM_DIGEST[:12],
        )
        return False

    _call_backward_reverse_indexed._torchtitan_patched = True
    ScanAutogradImpl.call_backward = _call_backward_reverse_indexed
    logger.info(
        "Patched torch._higher_order_ops.scan.ScanAutogradImpl.call_backward"
        " to reverse by index instead of torch.flip"
    )
    return True
