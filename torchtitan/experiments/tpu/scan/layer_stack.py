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
returns the sharded DTensor, which `forward` unwraps and the body gathers one
layer at a time. See `fsdp.py`.
"""

import contextlib
from typing import Any, Iterable

import torch
from torch import nn
from torch._higher_order_ops.scan import scan
from torch.distributed.tensor import DTensor
from torch.func import functional_call
import torch.utils.checkpoint

from torchtitan.experiments.tpu.scan.fsdp import gather_slice, ScanFSDPSpec
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
    # Mangled name -> how to gather that parameter in the body. Empty until
    # `shard_stacked_layers` runs; empty means "use the parameter as-is".
    self._fsdp_plan: dict[str, ScanFSDPSpec] = {}

    per_layer = [dict(layer.named_parameters()) for layer in layers]
    self.param_fqns: list[str] = sorted(per_layer[0])
    for i, params in enumerate(per_layer[1:], start=1):
      if sorted(params) != self.param_fqns:
        raise ValueError(
            f"layer {i} has parameter FQNs {sorted(params)}, layer 0 has"
            f" {self.param_fqns}; scan requires structurally identical layers."
        )
      for fqn in self.param_fqns:
        if params[fqn].shape != per_layer[0][fqn].shape:
          raise ValueError(
              f"layer {i}.{fqn} has shape {tuple(params[fqn].shape)}, layer"
              f" 0.{fqn} has {tuple(per_layer[0][fqn].shape)}."
          )

    # Stack along dim 0, the scan axis. Dim 0 must never be sharded: slicing a
    # sharded axis inside the loop would force a dynamic collective per
    # iteration. `fsdp.py` shards these on dim 1.
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

  def set_fsdp_plan(self, plan: dict[str, ScanFSDPSpec]) -> None:
    """Installs the in-body gather plan; called by `shard_stacked_layers`."""
    self._fsdp_plan = plan

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

    template, fqns, plan = self.scan_template, self.param_fqns, self._fsdp_plan

    def body(carry, params_slice):
      # One layer's slice of each stacked parameter. Under FSDP that slice is
      # this rank's *shard* of the layer's weight and `gather_slice` issues the
      # all-gather right here, inside the combine_fn, so it lowers into the
      # `stablehlo.while` body and only the layer in flight is ever gathered.
      params = {
          fqn: gather_slice(params_slice[_mangle(fqn)], plan.get(_mangle(fqn)))
          for fqn in fqns
      }
      out = functional_call(template, params, (carry, *block_args))
      # `scan` requires a non-empty per-iteration output: returning `None` (or
      # any empty pytree) trips "all fw_gm output nodes must be torch.fx.Node
      # with 'val' in meta" in the HOP's autograd path. Only the final carry is
      # wanted, so emit the cheapest possible placeholder and discard it.
      return out, out.new_zeros(())

    # `self._parameters`, NOT `getattr` -- see the module docstring. A DTensor
    # cannot cross into the body either: the mesh it closes over is lifted as a
    # `FakeScriptObject` HOP operand and `scan` takes only Tensor/int/SymInt. So
    # hand it the local shard, whose `Shard(1)` gradient placement is exactly
    # what `gather_slice`'s reduce-scatter backward produces.
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

  # -- state_dict adapter ------------------------------------------------------
  # On disk we keep the *unrolled* FQNs, so a scanned run and an unrolled run
  # share checkpoints in both directions.

  def stacked_to_unrolled(self) -> dict[str, torch.Tensor]:
    """Returns `{"<i>.<layer-relative fqn>": tensor}` -- the ModuleDict layout."""
    return {
        f"{i}.{fqn}": self._parameters[_mangle(fqn)][i]  # pylint: disable=protected-access
        for fqn in self.param_fqns
        for i in range(self.n_layers)
    }

  @torch.no_grad()
  def load_unrolled(self, unrolled: dict[str, torch.Tensor]) -> None:
    """Loads a `{"<i>.<layer-relative fqn>": tensor}` mapping into the stack."""
    for fqn in self.param_fqns:
      stacked = self._parameters[_mangle(fqn)]  # pylint: disable=protected-access
      for i in range(self.n_layers):
        key = f"{i}.{fqn}"
        if key not in unrolled:
          raise KeyError(f"missing {key} in unrolled state dict")
        stacked[i].copy_(unrolled[key])


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


__all__ = ["ScannedLayerStack", "stack_layers"]
