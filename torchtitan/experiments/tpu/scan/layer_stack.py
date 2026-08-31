# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runs a decoder's transformer blocks under a single `scan`.

`Decoder.forward` runs `for layer in self.layers.values(): h = layer(h, ...)`.
Unrolled, that compiles N structurally identical blocks into N copies of the
same code: compile time and program size grow with depth. `ScannedLayerStack`
replaces the `ModuleDict` at `model.layers` and collapses the loop into one
`torch.ops.higher_order.scan`, which lowers to a single `stablehlo.while`.

The decoder's loop is not modified. The stack exposes `.values()` returning
`[self]` and is itself callable with the block signature, so the loop runs once
and the single "layer" it gets is the whole scan. Reverting is
`model.layers = original`.

Parameters are stored **stacked**: one `[n_layers, *shape]` tensor per
layer-relative FQN, registered on this module under a mangled name (see `_SEP`),
because that is the layout `scan` consumes as `xs`. The per-layer modules stay
alive as *shadows* -- unregistered, parameters moved to `meta`, so they cost no
HBM -- for two reasons: layer 0 is the template `functional_call` runs, and each
shadow owns its `_param_init` closures so `init_states` can replay the stock
per-layer initializer directly into the stack.

One rule governs every read of a stacked parameter: go through
`self._parameters[...]`, never `getattr`. SimpleFSDP installs its all-gather as
a property on the parameter name, so a `getattr` gathers all N layers at once --
136.9 GB of bf16 weights at 70B, against a 96 GB budget. `self._parameters`
returns the sharded DTensor, whose local shard `forward` hands to the body; the
body all-gathers one layer's slice at a time, so the collective lands inside the
`stablehlo.while` and only the layer in flight is ever gathered. That is 1.71 GB
instead of 136.9 GB, and it is the whole reason to scan at 70B.
"""

import contextlib
from typing import Any, Iterable

import torch
from torch import nn
from torch._higher_order_ops.scan import scan
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard
from torch.func import functional_call
import torch.utils.checkpoint

from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.experiments.tpu.scan.recompute import body_checkpointed
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger

# Layer-relative FQNs contain ".", which neither `nn.ParameterDict` (it rejects
# such keys) nor `register_parameter` (it reads "." as a submodule separator)
# accepts, so "." is mangled to `_SEP` on the way in.
_SEP = "___"


def _mangle(fqn: str) -> str:
  return fqn.replace(".", _SEP)


def _to_meta_(module: nn.Module) -> None:
  """Moves every parameter of `module` to `meta` in place, keeping shapes."""
  for submodule in module.modules():
    for name, param in list(submodule._parameters.items()):  # pylint: disable=protected-access
      if param is not None:
        submodule._parameters[name] = nn.Parameter(  # pylint: disable=protected-access
            torch.empty_like(param, device="meta"), requires_grad=False
        )


@contextlib.contextmanager
def _bound_to(module: nn.Module, values: dict[str, torch.Tensor]):
  """Points `module`'s parameters at `values` for the duration of the block.

  `nn.Parameter(t)` shares storage with `t`, so binding a slice of the stack
  makes an in-place initializer write straight into the stack -- no copy, and no
  temporary materialization of a whole block.
  """
  saved = {}
  for fqn, value in values.items():
    *path, leaf = fqn.split(".")
    owner = module.get_submodule(".".join(path)) if path else module
    saved[fqn] = (owner, leaf, owner._parameters[leaf])  # pylint: disable=protected-access
    owner._parameters[leaf] = nn.Parameter(value, requires_grad=False)  # pylint: disable=protected-access
  try:
    yield
  finally:
    for owner, leaf, param in saved.values():
      owner._parameters[leaf] = param  # pylint: disable=protected-access


class _AllGatherReduceScatter(torch.autograd.Function):
  """`all_gather` forward / `reduce_scatter` backward with SimpleFSDP's dtypes.

  SimpleFSDP's own `ReplicateComputation` would be the natural thing to call
  here, and it is what the gather does above the loop. It cannot run *inside*
  the body, for two reasons that are both HOP-contract limits rather than
  anything about the collective:

    * a closed-over `DeviceMesh` is lifted as an extra HOP operand, and `scan`'s
      additional-inputs contract accepts only `Tensor | int | SymInt`, so the
      mesh arrives as a `FakeScriptObject` and the call dies;
    * `replicate_compute` reads `x._spec` to pick its branch, which Dynamo
      cannot wrap while tracing the sourceless combine_fn ("SourcelessBuilder
      .create does not know how to wrap DTensorSpec").

  `torch.distributed._functional_collectives` takes the group as a plain `str`,
  which bakes into the graph as a constant, so it is what the body uses. The
  dtypes are still SimpleFSDP's -- a plain `funcol.all_gather_tensor` already
  differentiates to a `reduce_scatter`, but it reduces in whatever dtype the
  forward gathered in, and SimpleFSDP is deliberately asymmetric: gather in bf16
  to halve comm volume, reduce in fp32 to keep the gradient accurate. No
  composition of casts around a symmetric collective reproduces that.

  Deleting this class needs upstream ask 4 in `README.md`.
  """

  @staticmethod
  def forward(  # pylint: disable=arguments-differ
      ctx: Any,
      shard: torch.Tensor,
      group_name: str,
      param_dtype: torch.dtype | None,
      reduce_dtype: torch.dtype | None,
  ) -> torch.Tensor:
    ctx.group_name = group_name
    ctx.reduce_dtype = reduce_dtype
    ctx.shard_dtype = shard.dtype
    if param_dtype is not None:
      shard = shard.to(param_dtype)
    return funcol.all_gather_tensor(shard, gather_dim=0, group=group_name)

  @staticmethod
  def backward(ctx: Any, grad: torch.Tensor):  # pylint: disable=arguments-differ
    if ctx.reduce_dtype is not None:
      grad = grad.to(ctx.reduce_dtype)
    out = funcol.reduce_scatter_tensor(
        grad, reduceOp="sum", scatter_dim=0, group=ctx.group_name
    )
    return out.to(ctx.shard_dtype), None, None, None


class ScannedLayerStack(Module):
  """A `ModuleDict`-compatible stand-in that runs its layers under `scan`.

  Args:
    layers: the per-layer modules in execution order. They must be structurally
      identical; `__init__` checks the parameter FQN set and the per-FQN shapes.
      Dtypes are not checked -- no configuration this lane can produce varies
      dtype across layers, and one that did would reach `torch.stack`, which
      promotes rather than raising.
    peel_first_layer: run layer 0 outside the `scan` and scan over layers
      1..N-1. Needed when a block calls a JAX-exported kernel; see `forward`.
    recompute_body: checkpoint the block, so backward recomputes it instead of
      keeping a per-layer residual. See `recompute.py`.
  """

  def __init__(
      self,
      layers: list[nn.Module],
      *,
      peel_first_layer: bool = False,
      recompute_body: bool = False,
  ):
    super().__init__()
    if not layers:
      raise ValueError("ScannedLayerStack needs at least one layer.")

    self.n_layers = len(layers)
    self.peel_first_layer = peel_first_layer
    self.recompute_body = recompute_body

    per_layer = [dict(layer.named_parameters()) for layer in layers]
    self.param_fqns: list[str] = sorted(per_layer[0])
    def signature(params: dict[str, nn.Parameter]) -> dict[str, tuple[int, ...]]:
      return {fqn: tuple(p.shape) for fqn, p in params.items()}

    for i, params in enumerate(per_layer[1:], start=1):
      if signature(params) != signature(per_layer[0]):
        raise ValueError(
            f"layer {i} has parameters {signature(params)}, layer 0 has"
            f" {signature(per_layer[0])}; scan requires structurally identical"
            " layers. Dtypes are not checked -- no configuration this lane can"
            " produce varies dtype across layers."
        )

    # Stack along dim 0, the scan axis. Dim 0 must never be sharded: slicing a
    # sharded axis inside the loop would force a dynamic collective per
    # iteration. `shard_stacked_layers` shards these on dim 1.
    for fqn in self.param_fqns:
      stacked = torch.stack([p[fqn].detach() for p in per_layer], dim=0)
      self.register_parameter(
          _mangle(fqn),
          nn.Parameter(stacked, requires_grad=per_layer[0][fqn].requires_grad),
      )

    # Shadows: kept alive for their `_param_init` closures and as the template,
    # but unregistered so their `meta` parameters never reach `.parameters()`,
    # `state_dict()` or `to_empty()`.
    self._shadow_layers = list(layers)
    for layer in self._shadow_layers:
      _to_meta_(layer)

    logger.info(
        "ScannedLayerStack: %d layers, %d stacked parameters, first is %s%s.",
        self.n_layers,
        len(self.param_fqns),
        self.param_fqns[0],
        tuple(self._parameters[_mangle(self.param_fqns[0])].shape),  # pylint: disable=protected-access
    )

  @property
  def scan_template(self) -> nn.Module:
    """The one block the scan body executes -- every layer runs through it.

    Public because a model-wide module patch (splash attention, a fused loss)
    walks `model.modules()`, which stops at this stack: the shadows are
    unregistered on purpose. Patching this block patches all N layers, and
    `workarounds._all_modules` looks for exactly this attribute name.
    """
    return self._shadow_layers[0]

  def stacked_parameters(self) -> list[nn.Parameter]:
    """The `[n_layers, *shape]` parameters, in `param_fqns` order."""
    return [self._parameters[_mangle(fqn)] for fqn in self.param_fqns]  # pylint: disable=protected-access

  def slice_specs(self) -> dict[str, tuple[DeviceMesh, Placement]]:
    """`{mangled name: (mesh, placement)}` for the *per-layer* slice of each
    stacked parameter, for the ones SimpleFSDP made DTensors.

    The stack is sharded on dim 1 (dim 0 is the scan axis; slicing a sharded
    axis inside the loop would need a dynamic collective per iteration), so
    dropping the layer axis makes that dim 0 of the per-layer weight.

    The placement is read back off each DTensor rather than assumed from the
    mode, because `data_parallel` silently leaves a parameter `Replicate` when
    its dim-1 extent does not divide the mesh. Wrapping such a slice as
    `Shard(0)` would all-gather `mesh.size()` copies of a full weight -- wrong
    numbers, no error, and no test at debugmodel scale would catch it.

    Raises:
      ValueError: on a mesh or placement the in-body gather cannot express.
    """
    specs: dict[str, tuple[DeviceMesh, Placement]] = {}
    for name, param in self._parameters.items():  # pylint: disable=protected-access
      if not isinstance(param, DTensor):
        continue
      mesh = param.device_mesh
      if mesh.ndim != 1:
        raise ValueError(
            f"{name}: the in-body gather supports a 1-D dp mesh only (got"
            f" {mesh.ndim}-D). SimpleFSDP's nD path collapses the mesh with a"
            " to_local/from_local round-trip that has not been shown to trace"
            " into a HOP body."
        )
      placement = param.placements[0]
      if isinstance(placement, Shard) and placement.dim != 1:
        raise ValueError(
            f"{name} is sharded on dim {placement.dim}; the scan axis is dim 0"
            " and the in-body gather assumes dim 1 (= dim 0 of the per-layer"
            " weight)."
        )
      specs[name] = (mesh, Shard(0) if isinstance(placement, Shard) else Replicate())
    return specs

  # -- device movement + initialization ----------------------------------------

  def _apply(self, fn, recurse=True):
    module = super()._apply(fn, recurse=recurse)
    if recurse:
      # `nn.Module._apply` never reaches the shadows -- that is the point, their
      # parameters must stay on `meta`. Their *buffers* are small, are read by
      # the scan body (RoPE's `cache`), and so must follow the module across
      # `.to(device)` / `.to_empty(device)`.
      for layer in self._shadow_layers:
        for submodule in layer.modules():
          for name, buf in list(submodule._buffers.items()):  # pylint: disable=protected-access
            if buf is not None:
              submodule._buffers[name] = fn(buf)  # pylint: disable=protected-access
    return module

  def init_states(self, *, buffer_device: torch.device | None = None) -> None:
    """Replays each layer's own initializer into its slice of the stack.

    Binding layer *i*'s parameters to views of `stacked[i]` means every slice is
    initialized by exactly the code the unrolled model would have run, including
    any per-layer-index scaling, and nothing larger than the stack is allocated.
    """
    stacked = {fqn: self._parameters[_mangle(fqn)] for fqn in self.param_fqns}  # pylint: disable=protected-access
    device = next(iter(stacked.values())).device
    if buffer_device is None:
      buffer_device = device

    for i, layer in enumerate(self._shadow_layers):
      # `init_states` recomputes buffers in place, so they must be off `meta`.
      for submodule in layer.modules():
        for name, buf in list(submodule._buffers.items()):  # pylint: disable=protected-access
          if buf is not None and buf.device.type == "meta":
            submodule._buffers[name] = torch.empty_like(buf, device=device)  # pylint: disable=protected-access
      with _bound_to(layer, {fqn: value[i] for fqn, value in stacked.items()}):
        layer.init_states(buffer_device=buffer_device)

  # -- ModuleDict compatibility ------------------------------------------------
  # `Decoder.forward` does `for layer in self.layers.values(): h = layer(h, ...)`.
  # Returning `[self]` makes that loop run once over the whole scanned stack.

  def values(self) -> Iterable[nn.Module]:
    return [self]

  def items(self) -> Iterable[tuple[str, nn.Module]]:
    return [("scan", self)]

  def __len__(self) -> int:
    return self.n_layers

  # -- forward -----------------------------------------------------------------

  def forward(self, h: torch.Tensor, *block_args: Any) -> torch.Tensor:
    """Runs every layer on `h`, passing `block_args` to each one unchanged.

    `block_args` is whatever the decoder's loop hands a block after the hidden
    state -- `attention_masks`, `positions`, `segment_ids` today. They are the
    same object on every iteration, so the body closes over them and they enter
    the HOP as additional inputs rather than as `xs`. Taking them as `*args`
    keeps the stack a drop-in for the block whatever that list grows to.
    """
    if self.training and not h.requires_grad:
      # With a carry that does not require grad, the HOP's backward returns
      # zeros for the carry gradient. That severs the inter-layer chain -- every
      # layer's weight gradient then reflects only itself -- while leaving the
      # loss bit-exact and raising nothing. Fail loudly instead.
      raise RuntimeError(
          "ScannedLayerStack: the scan init carry does not require grad; the"
          " backward would silently zero every inter-layer gradient."
      )

    template, fqns = self.scan_template, self.param_fqns
    # The gather's dtypes come off SimpleFSDP's own parametrization -- the object
    # behind the property that reading a parameter name would have triggered --
    # so the bf16-gather / fp32-reduce asymmetry has one definition, not two.
    # Its group is resolved to a plain `str` out here: see
    # `_AllGatherReduceScatter` for why a `DeviceMesh` cannot cross into a body.
    gather = getattr(self, "_simple_fsdp_parametrization_list", [None])[0]
    groups = {
        name: mesh.get_group().group_name if isinstance(placement, Shard) else None
        for name, (mesh, placement) in self.slice_specs().items()
    }

    def body(carry, params_slice):
      # One layer's slice of each stacked parameter. Under FSDP that slice is
      # this rank's *shard* of the layer's weight, so the all-gather that turns
      # it into the layer's full weight happens here, inside the combine_fn: it
      # lowers into the `stablehlo.while` body and only the layer in flight is
      # ever gathered.
      params = {}
      for fqn in fqns:
        mangled = _mangle(fqn)
        value = params_slice[mangled]
        if mangled in groups:
          group = groups[mangled]
          if group is None:  # replicated by SimpleFSDP: cast, do not gather
            value = value if gather.param_dtype is None else value.to(gather.param_dtype)
          else:
            value = _AllGatherReduceScatter.apply(
                value, group, gather.param_dtype, gather.reduce_dtype
            )
        params[fqn] = value
      out = functional_call(template, params, (carry, *block_args))
      # `scan` requires a non-empty per-iteration output: returning `None` (or
      # any empty pytree) trips "all fw_gm output nodes must be torch.fx.Node
      # with 'val' in meta" in the HOP's autograd path. Only the final carry is
      # wanted, so emit the cheapest possible placeholder and discard it.
      return out, out.new_zeros(())

    # `self._parameters`, NOT `getattr` -- see the module docstring. A DTensor
    # cannot cross into the body *as an operand* either: the mesh it closes over
    # is lifted as a `FakeScriptObject` and `scan` takes only Tensor/int/SymInt.
    # Rebuilding one inside the body from a plain-tensor operand is fine, which
    # is what `body` does: `from_local` and `redistribute` are traced through,
    # leaving only the functional collective behind.
    xs = {}
    for fqn in fqns:
      mangled = _mangle(fqn)
      param = self._parameters[mangled]  # pylint: disable=protected-access
      xs[mangled] = param.to_local() if isinstance(param, DTensor) else param

    if self.peel_first_layer and self.n_layers > 1:
      # torch_tpu clones only the body module's `main` into the
      # `stablehlo.while` region (`dispatch_scan.cc: BuildLoopBody`), leaving
      # behind the sibling `func.func private @<kernel>` that a JAX-exported
      # kernel is spliced in as. The cloned `func.call` is then a dangling
      # symbol and the outer module fails verification (XLA misreports this as
      # "failed to upgrade versioned StableHLO"). Running layer 0 outside the
      # scan puts an identically-named copy of that function in the outer
      # module, where the call resolves -- the symbol is a pure function of
      # kernel name plus operand shapes and dtypes.
      #
      # The peeled layer is checkpointed so its *backward* recomputes the
      # forward, emitting the forward kernel's symbol into the backward module
      # too, which the scanned body's own recompute needs. `torch.utils.
      # checkpoint` is a silent no-op inside a HOP body but works normally here.
      head = {mangled: value[0] for mangled, value in xs.items()}
      h, _ = torch.utils.checkpoint.checkpoint(body, h, head, use_reentrant=False)
      xs = {mangled: value[1:] for mangled, value in xs.items()}

    # The whole scan is traced inside the window so the body's joint graph gets
    # the MUST_RECOMPUTE tag; the outer `scan` node and the body placeholders
    # pick it up too, and `recompute.install()` strips it back off those.
    with body_checkpointed(self.recompute_body):
      carry, _ = scan(body, h, xs)
    return carry


def stack_layers(
    model: nn.Module,
    *,
    peel_first_layer: bool = False,
    recompute_body: bool = False,
) -> ScannedLayerStack:
  """Swaps `model.layers` (a `ModuleDict` of blocks) for a `ScannedLayerStack`.

  Idempotent. Must run before anything else rewrites `model.layers`: activation
  checkpointing and FSDP both iterate the `ModuleDict`'s children, and the stack
  has a different child structure.
  """
  layers = getattr(model, "layers", None)
  if isinstance(layers, ScannedLayerStack):
    return layers
  if layers is None:
    raise ValueError("model has no `.layers` to scan over.")

  ordered = [layers[str(i)] for i in range(len(layers))]
  model.layers = ScannedLayerStack(
      ordered,
      peel_first_layer=peel_first_layer,
      recompute_body=recompute_body,
  )
  return model.layers


def shard_stacked_layers(
    stack: ScannedLayerStack,
    *,
    dp_mesh: DeviceMesh,
    dp_mode: str,
    mp_policy: MixedPrecisionPolicy,
) -> dict[str, tuple[DeviceMesh, Placement]]:
  """Shards the stack on dim 1 so the body can gather one layer at a time.

  Runs after SimpleFSDP has wrapped the rest of the model with the stacked
  parameters ignored.

  `data_parallel` is used here for its *sharding*: `distribute_tensor` plus
  `OffsetBasedRNGTracker` is what keeps initialization bit-identical to a
  single-device run. The all-gather it installs alongside is bypassed --
  `forward` reads `self._parameters` and calls that same parametrization object
  on one layer's slice inside the scan body instead.

  Returns:
    The per-layer slice specs, i.e. `stack.slice_specs()`, which is also called
    here so an unsupported mesh or shard dim fails at wrap time rather than at
    the first traced step.
  """
  data_parallel(stack, dp_mesh, mode=dp_mode, mp_policy=mp_policy, shard_dim=1)
  specs = stack.slice_specs()
  n_sharded = sum(isinstance(p, Shard) for _, p in specs.values())
  logger.info(
      "scan in-body FSDP: %d/%d stacked parameters gathered per layer inside"
      " the scan body (the rest are replicated).",
      n_sharded,
      len(specs),
  )
  return specs


__all__ = ["ScannedLayerStack", "stack_layers", "shard_stacked_layers"]
