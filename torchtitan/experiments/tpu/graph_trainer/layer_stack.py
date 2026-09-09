# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Store sibling decoder layers' weights in one stacked tensor per parameter.

    before:  layers.7.attention.wq.weight  ->  its own [d, d] tensor
    after:   layers.7.attention.wq.weight  ->  row 7 of a shared [n, d, d] tensor

Storage only. The ``n`` layer modules stay registered where they were, so
anything that walks the module tree keeps working, and the layers still run one
at a time in a Python ``for``. A later PR runs them as a ``scan``, which is why
the weights need a leading loop axis in the first place.

Everything below follows from three facts, so they are stated once here rather
than at each use:

1.  **The stacks must exist, and the rows must not, before ``data_parallel``
    walks the tree.** It has to find one stack to shard rather than ``n`` rows:
    leave the rows registered and FSDP shards them too, so one weight ends up
    with two sets of gradients. Later is no good either -- an unsharded stack is
    ``n`` times too big on every rank.

2.  **The per-layer names have to come back for ``init_weights`` and
    ``state_dict``, but only for as long as they run.** Both address weights as
    ``layers.i.<p>``. ``unrolled()`` re-registers those names as ``Parameter``
    views onto the rows, so both work unchanged and checkpoints stay in the
    unrolled format. The views cannot just be left up: ``parameters()`` would
    then yield every weight twice, once as a row and once inside the stack, and
    the optimizer would step it twice. Hence a scope rather than a pair of
    calls -- and why ``_attach_views``/``_drop_views`` are private.

    They are views, not copies, and taken after ``to_empty`` -- which
    reallocates storage without preserving aliasing, so one made on meta would
    be orphaned and every write would land where nothing reads.

3.  **Reach the stacks through ``_parameters``, not ``getattr``, anywhere the
    result is stored.** After SimpleFSDP the attribute is a property returning
    the *gathered* tensor, so a view of it would alias a temporary rather than
    this rank's shard. The two loops are the deliberate exception: they want
    the gather.

Because the views *are* the parameters while ``init_states`` runs,
initialization is bit-identical to the unrolled model by construction, depth
scaling included -- and stays so under sharding, since ``stack[i]`` of a
``Shard(1)`` stack is ``Shard(0)`` on the slice, exactly what the unrolled
``wq.weight`` gets.

Two things still keyed by per-layer names are *not* handled here: the optimizer
state dict, and ``ParamGroupConfig.pattern``, which matches
``named_parameters()``. A config using anything but a catch-all pattern needs
both first.
"""

import contextlib
from collections.abc import Callable, Iterator

import torch
from torch import nn
from torch._higher_order_ops.scan import scan
from torch.func import functional_call

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.tpu.graph_trainer.scan_guards import (
    check_layers_are_stackable,
    check_scan_parallelism,
)
from torchtitan.models.common.decoder import LayerDict
from torchtitan.tools.logging import logger


__all__ = ["apply_scan", "LayerStack", "StackedLayers", "STACK_ATTR"]


# The stacks live on their own module, registered on the model under this name.
# It owns the stacks and *nothing else* on purpose: SimpleFSDP skips a module
# whole once it has sharded it, so anything else parked here would silently go
# unsharded.
STACK_ATTR = "layer_stacks"

# ``register_parameter`` rejects names containing a dot, so a per-layer FQN like
# ``attention.wq.weight`` cannot be a stack's name as-is. The mangling never
# leaves this module -- the state-dict hooks below emit the real FQNs.
def _mangle(name: str) -> str:
    return name.replace(".", "__")


class StackedLayers(LayerDict):
    """``model.layers``, running its layers out of the stacks.

    ``Decoder.forward`` calls its layers container instead of looping over it,
    so replacing the container is all it takes to change how the layers are run
    -- as ``n_layers`` ``functional_call``s, or as one ``scan`` over the stacks.
    The layer modules inside are the same objects either way, so their
    parameters, FQNs and SimpleFSDP parametrizations are untouched.
    """

    def __init__(self, layers: LayerDict, stack: "LayerStack", *, use_scan: bool):
        super().__init__(layers)
        self.stack = stack
        self.use_scan = use_scan

    def forward(self, h: torch.Tensor, *args) -> torch.Tensor:
        loop = self.stack.scan_loop if self.use_scan else self.stack.layer_loop
        return loop(h, *args)


class LayerStack:
    """Owns the stacks and the transitions between the two layouts.

    Only ``unrolled`` is called from outside, by ``init_states`` and the
    state-dict hooks. ``apply_stacked_layers`` below drives the rest, in the one
    order that works; each step asserts what has already run, so the order
    cannot silently go wrong.
    """

    def __init__(self, layers: nn.ModuleDict, holder: nn.Module):
        self.layers = layers
        # The bare `nn.Module` the stacks are registered on -- see STACK_ATTR.
        self.holder = holder
        self.n_layers = len(layers)
        # Per-layer parameter FQNs relative to a layer, e.g. "attention.wq.weight".
        self.names: list[str] = []
        # What has already run, so a caller cannot reorder the steps.
        self._unregistered = False
        self._views_attached = False
        # Stacks parked here while the views stand in for them, inside
        # `unrolled()`.
        self._hidden: dict[str, nn.Parameter] = {}

    def _slots(self) -> Iterator[tuple[int, str, nn.Module, str]]:
        """Every per-layer parameter, as (row, FQN, owning module, attribute).

        A per-layer FQN like "attention.wq.weight" names a parameter two modules
        down, and `_parameters` is keyed by the last component alone, so every
        walk below needs the pair -- and they must all visit the same set, since
        attaching and dropping views are inverses.
        """
        for i, layer in enumerate(self.layers.values()):
            for name in self.names:
                *path, attr = name.split(".")
                mod = layer
                for part in path:
                    mod = getattr(mod, part)
                yield i, name, mod, attr

    # -- construction ------------------------------------------------------

    def unregister_and_allocate(self) -> None:
        """Replace each *kind* of per-layer weight with one stacked tensor.

        The layers hold the same kinds of weight as each other -- ``n_layers``
        separate ``attention.wq.weight``, ``n_layers`` separate
        ``feed_forward.w1.weight``, and so on. Each of those groups becomes a
        single ``[n_layers, *shape]`` tensor, so the first loop is over kinds,
        not layers. The second drops the originals, leaving the layer modules
        owning no parameters at all -- fact 1 above.

        Nothing is allocated for real: the model is still on meta, so these are
        shape-and-dtype placeholders that ``to_empty`` will materialize.
        """
        assert not self._unregistered, "unregister_and_allocate() ran twice"
        # Layer 0 defines the kinds; every other layer must match it, or its
        # weights cannot share a stack.
        check_layers_are_stackable(self.layers)
        for name, p in self.layers["0"].named_parameters():
            self.names.append(name)
            self.holder.register_parameter(
                _mangle(name),
                nn.Parameter(
                    torch.empty(
                        self.n_layers, *p.shape, dtype=p.dtype, device=p.device
                    ),
                    requires_grad=p.requires_grad,
                ),
            )
        # Row i of the stack is now the only copy: `layers.7.attention.wq` no
        # longer resolves to a parameter, and nothing walking the tree finds it.
        for _, _, mod, attr in self._slots():
            del mod._parameters[attr]
        self._unregistered = True

    # -- the two layouts ---------------------------------------------------

    @contextlib.contextmanager
    def unrolled(self):
        """Make the model look unstacked for a block -- fact 2 above.

        Inside it the model is indistinguishable from the unrolled one: same
        parameters, same names, same ``state_dict``, and writes land in the
        stack. A scope and not two calls, so the views come down even if the
        block raises.
        """
        self._attach_views()
        try:
            yield
        finally:
            self._drop_views()

    def _attach_views(self) -> None:
        """Put the per-layer names back, aliasing their rows."""
        assert self._unregistered and not self._views_attached
        for i, name, mod, attr in self._slots():
            stack = self.holder._parameters[_mangle(name)]  # fact 3
            # `stack[i]` is a view, so this parameter shares storage with row i
            # -- writing the "layer" writes the stack. Registered by hand for
            # the same reason: `setattr` would go through the parametrization.
            mod._parameters[attr] = nn.Parameter(
                stack[i], requires_grad=stack.requires_grad
            )
        # Park the stacks off to the side, or every weight is reachable twice.
        self._hidden = dict(self.holder._parameters)
        self.holder._parameters.clear()
        self._views_attached = True

    def _drop_views(self) -> None:
        """Exact inverse of ``_attach_views``."""
        assert self._views_attached
        for _, _, mod, attr in self._slots():
            del mod._parameters[attr]
        self.holder._parameters.update(self._hidden)
        self._hidden = {}
        self._views_attached = False

    # -- installation ------------------------------------------------------

    def install(self, model: nn.Module, *, use_scan: bool = False) -> None:
        """Attach the stack to the model, after ``data_parallel`` has sharded it.

        Three attachments, and the model is not stacked until all three are on:
        initialization runs with the views up, checkpoints keep the unrolled
        keys, and ``model.layers`` runs its layers out of the stacks.

        The state-dict hooks use the halves of ``unrolled`` rather than the
        context manager, because torch's API is a pre-hook and a post-hook with
        the caller's own ``state_dict`` call in between -- there is no block to
        wrap. Plain closures rather than bound methods, because the hook
        registrations set an attribute on whatever they are given and a bound
        method rejects that.
        """
        assert self._unregistered, "install() before unregister_and_allocate()"
        inner = model.init_states

        def init_states(**kwargs):
            with self.unrolled():
                inner(**kwargs)

        def attach(*args):
            self._attach_views()

        def drop(*args):
            self._drop_views()

        model.init_states = init_states
        # Save and load in the unrolled format, so a run with the flag on loads
        # a checkpoint written with it off and the reverse -- stacking is an
        # execution detail, not a change to the model. Putting the views up and
        # letting the model produce its own state dict, rather than translating
        # keys afterwards, is what keeps every other hook in the tree working:
        # notably the fused-QKV split, which emits wq/wk/wv from one wqkv
        # parameter and would be bypassed by anything rewriting keys outside it.
        model.register_state_dict_pre_hook(attach)
        model.register_state_dict_post_hook(drop)
        model.register_load_state_dict_pre_hook(attach)
        model.register_load_state_dict_post_hook(drop)
        model.layers = StackedLayers(model.layers, self, use_scan=use_scan)
        self.layers = model.layers

    # -- running the layers ------------------------------------------------

    def layer_loop(self, h: torch.Tensor, *args):
        """The layers run one at a time, reading their weights out of the stacks.

        Layer 0 is the template: ``functional_call`` runs its ``forward`` with
        another layer's weights. The layers' own buffers (the RoPE cache) are
        identical across layers, so the template's are correct every iteration.

        ``getattr``, unlike everywhere else, is deliberate here: it goes through
        SimpleFSDP's parametrization, so the stack is all-gathered once and then
        sliced -- the same collectives the unrolled model issues.
        """
        assert self._unregistered and not self._views_attached
        template = self.layers["0"]
        stacks = {name: getattr(self.holder, _mangle(name)) for name in self.names}
        for i in range(self.n_layers):
            h = functional_call(
                template, {name: s[i] for name, s in stacks.items()}, (h, *args)
            )
        return h

    def scan_loop(self, h: torch.Tensor, *args):
        """The same loop as ``layer_loop``, run as one ``scan``.

        The stacks are the ``xs``, sliced along their leading layer axis, and
        the hidden state is the carry. ``n_layers`` copies of the block collapse
        into a single ``stablehlo.while`` body, so compile time and program size
        stop growing with depth.

        This does *not* yet change where the all-gather happens: ``getattr``
        still gathers the whole ``[n_layers, ...]`` stack before the loop, the
        same as ``layer_loop`` does. Moving it into the body is a separate
        change.
        """
        assert self._unregistered and not self._views_attached
        template = self.layers["0"]
        if template.training and not h.requires_grad:
            # `scan`'s backward returns `zeros_like(carry)` when the init carry
            # does not require grad, which severs the inter-layer gradient chain
            # while keeping the loss bit-exact. Fail loudly instead.
            raise RuntimeError(
                "the scan init carry does not require grad, so every layer's "
                "weight gradient would reflect only itself -- with a correct "
                "loss and no error"
            )

        # One whole [n_layers, *shape] stack per parameter, keyed by its mangled
        # name. For llama3-70B that is
        #
        #     {"attention__wq__weight":     [80, 8192, 8192],
        #      "attention__wk__weight":     [80, 1024, 8192],
        #      ...
        #      "feed_forward__w1__weight":  [80, 28672, 8192]}
        #
        # `scan` slices each along its leading axis, so iteration i gets a dict
        # of the same keys holding row i -- [8192, 8192], [1024, 8192], ...
        xs = {_mangle(name): getattr(self.holder, _mangle(name)) for name in self.names}

        def body(carry, slices):
            # Layer 0's module stands in for every layer, one row at a time,
            # which is what makes the body a single traced graph rather than
            # n_layers of them. It has no parameters of its own to conflict
            # with -- they were unregistered when the stacks were allocated.
            params = {name: slices[_mangle(name)] for name in self.names}
            out = functional_call(template, params, (carry, *args))
            # `scan` rejects a body that returns no per-iteration output, so
            # emit the cheapest possible one and drop it; only the carry is used.
            return out, out.new_zeros(())

        carry, _ = scan(body, h, xs)
        return carry


def apply_scan(
    model: nn.Module,
    *,
    shard_fn: Callable[..., nn.Module],
    parallel_dims: ParallelDims,
    use_scan: bool = True,
) -> nn.Module:
    """Stack the decoder layers' weights and run them as one ``scan``.

    Renamed from ``apply_stacked_layers``: with ``use_scan`` the stack is no
    longer only a storage layout, and the loop is the point of it.

    The whole opt-in, on a model still on meta::

        model = apply_scan(
            model,
            shard_fn=partial(data_parallel, device_mesh=..., mode=...),
            parallel_dims=parallel_dims,
        )

    Requirements on the model -- the porting checklist for a new one:

    * ``model.layers`` is a ``LayerDict``: a container that runs its own layers
      in its ``forward``. This works by swapping that container, so a parent
      that loops over the layers itself cannot be stacked.
    * The layers are homogeneous: identical parameter names, shapes and dtypes.
      Checked; a per-layer shape cannot share a stack.
    * The layer signature is ``(h, *args)``, with only ``h`` varying from one
      layer to the next. Everything else must be loop-invariant.
    * Per-layer buffers are identical across layers. They are shared, not
      stacked, so every layer sees layer 0's.
    * ``model.init_states`` exists and is where the weights are initialized. It
      is wrapped, so that the per-layer views are up while it runs.

    And on the run, with ``use_scan``:

    * The layer must not mutate its inputs. ``scan`` is a higher-order op, and
      a body that writes to what it was handed raises
      ``UncapturedHigherOrderOpError`` at trace time.
    * The step has to be traced -- ``--tpu-config.compile-mode=aot_fx_trace``.
      Eager ``scan`` does not materialize on TPU.

    Nothing else changes: FQNs, ``state_dict``, the optimizer and the
    checkpoint format are all as they were unstacked.

    ``shard_fn(module, *, shard_dim=0)`` is called twice, and in this order,
    which is the part that is easy to get wrong:

    1. the stacks, on ``shard_dim=1``. Dim 0 is the layer axis; sharding it
       would split one layer's weight across ranks and -- ``n_layers`` being
       smaller than the mesh -- silently fall back to ``Replicate``.
    2. the rest of the model, on its default axis, second: the stacks must
       exist before the model is walked, and ``shard_fn`` skips a module it has
       already sharded.

    With ``use_scan=False`` the layers still run one at a time, just out of the
    stacks -- the storage change without the loop change, which is what the
    reshard and checkpoint tests compare against.
    """
    if use_scan:
        check_scan_parallelism(parallel_dims)
    holder = nn.Module()
    setattr(model, STACK_ATTR, holder)
    stack = LayerStack(model.layers, holder)
    stack.unregister_and_allocate()
    shard_fn(stack.holder, shard_dim=1)
    model = shard_fn(model)
    stack.install(model, use_scan=use_scan)
    logger.info(
        "Stacked %d weights across %d layers, run as a %s",
        len(stack.names),
        stack.n_layers,
        "scan" if use_scan else "loop",
    )
    return model
