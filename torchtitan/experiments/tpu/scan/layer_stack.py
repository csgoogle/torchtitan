# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scan-over-layers for torchtitan decoder models on TPU.

Replaces the `for layer in self.layers.values(): h = layer(h, ...)` loop in
`torchtitan.models.common.decoder.Decoder.forward` with a single
`torch.ops.higher_order.scan`, so the N identical transformer blocks compile to
one `stablehlo.while` body instead of N unrolled copies.

Why this shape
--------------
`ScannedLayerStack` deliberately quacks like the `ModuleDict` it replaces: it
exposes `.values()` returning `[self]` and is itself callable with the block
signature `(h, attention_masks, positions)`. That means **the decoder's layer
loop is not touched** — it runs exactly once and the single "layer" it gets is
the whole scanned stack. Swapping back is `model.layers = original`.
(`decoder.py` does carry one unrelated change on this branch, to how tied
embeddings are re-pointed under a SimpleFSDP parametrization; it has nothing to
do with the loop.)

Parameters are stored **stacked**: one `[n_layers, *param_shape]` tensor per
layer-relative FQN, which is what `scan` needs as `xs`. The per-layer modules
survive only as *shadows* — their parameters are moved to `meta` (so they cost
no HBM; one materialized 70B block is ~0.9 GiB) and layer 0 doubles as the
`template` that `torch.func.functional_call` is invoked on with a slice of the
stack. Non-parameter state (RoPE's non-persistent `cache` buffer) stays on the
template and is shared across iterations, which is correct — it is identical for
every layer.

The shadows are not dead weight: they carry each layer's `_param_init` closures,
so `init_states` can bind layer *i*'s parameters to a view of `stacked[..., i]`
and run the stock per-layer initializer straight into the stack. Initialization
is therefore bit-identical to the unrolled model's, per layer, with no extra
allocation.

Stacked parameters are registered **directly on this module**, under the mangled
names described at `_SEP` below.

Both `forward` and `init_states` read them through `self._parameters[mangled]`,
never `getattr`. SimpleFSDP installs its all-gather as a *class property* per
parameter name, so a single `getattr` on a stacked parameter gathers all N
layers at once, before the loop the gather was supposed to happen inside: 136.9
GB of bf16 weights at 70B against a 96 GB budget. `forward` hands `scan` the
local shard and the body gathers one layer at a time — see `fsdp.py`, and
`tests/test_scan_fsdp_lowering.py`, which fails if this read path regresses
(nothing else does: both spellings are numerically correct).

Findings from scan bring-up that this file encodes:

  * Use the `scan()` **wrapper**, never the raw HOP. Under Dynamo the raw HOP is
    rejected outright (`UncapturedHigherOrderOpError: improper combine_fn …
    Expected combine_fn to be wrapped as functools.partial in scan user-facing
    api`).
  * `functional_call` in the combine_fn **works** under Dynamo — the predicted
    `_reparametrize_module` context-manager break does not happen here.
  * The **init carry must require grad**, or the backward returns
    `zeros_like(primals_0)` for the carry and severs the inter-layer gradient
    chain — with a bit-exact loss and no error.
  * `torch.utils.checkpoint` inside the body is a **silent no-op** (the HOP
    re-traces via `materialize_as_graph`, erasing the recompute tags). Do not
    rely on it; use the FX tagging pass in `fsdp.py` instead.

Lane constraint: executing a scan on TPU requires the workaround in
`torchtitan/experiments/tpu/scan/torch_tpu_fix.py`, which re-records the
deferred scan op on torch_tpu's events queue.

Both lanes work on v6e. `--tpu-config.compile-mode=fullgraph` measured 27.40%
MFU on 8B/v6e-8, and `--tpu-config.use-graph-trainer` with `aot_fx_trace`
measured 29.51% — the frontier, and the lane the 70B target runs on. On tpu7x
the graph-trainer lane can still abort inside the torch_tpu compile; that is
open. The abort is hard to attribute because torch_tpu's compile path raises it
through a `CHECK` that reports a generic failure rather than the underlying
status — see the `TraverseAndCompile` fixes this branch depends on.
"""

from typing import Any, Iterable

import torch
import torch.utils.checkpoint
from torch import nn
from torch._higher_order_ops.scan import scan
from torch.distributed.tensor import DTensor
from torch.func import functional_call

from torchtitan.experiments.tpu.scan.fsdp import gather_slice, ScanFSDPSpec
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger

# Layer-relative FQNs contain ".", which neither `nn.ParameterDict` (it rejects
# such keys) nor `register_parameter` (it treats "." as a submodule separator)
# accepts, so they are mangled on the way in and restored on the way out.
_SEP = "___"


def _mangle(fqn: str) -> str:
  return fqn.replace(".", _SEP)


def _demangle(key: str) -> str:
  return key.replace(_SEP, ".")


def _to_meta_(module: nn.Module) -> None:
  """Moves every parameter of `module` to `meta`, in place, keeping shapes."""
  for submodule in module.modules():
    for name, param in list(submodule._parameters.items()):  # pylint: disable=protected-access
      if param is None:
        continue
      submodule._parameters[name] = nn.Parameter(  # pylint: disable=protected-access
          torch.empty_like(param, device="meta"), requires_grad=False
      )


def _owner_of(module: nn.Module, fqn: str) -> tuple[nn.Module, str]:
  """Splits a parameter FQN into (owning submodule, leaf name)."""
  *path, leaf = fqn.split(".")
  owner = module
  for part in path:
    owner = getattr(owner, part)
  return owner, leaf


def _bind_params_(
    module: nn.Module, values: dict[str, torch.Tensor]
) -> dict[str, nn.Parameter]:
  """Points `module`'s parameters at `values` in place, returning the old ones.

  `nn.Parameter(t)` shares storage with `t`, so binding a slice of the stack
  makes any in-place initializer write straight into the stack — no copy, no
  temporary materialization of the block.
  """
  saved = {}
  for fqn, value in values.items():
    owner, leaf = _owner_of(module, fqn)
    saved[fqn] = owner._parameters[leaf]  # pylint: disable=protected-access
    owner._parameters[leaf] = nn.Parameter(value, requires_grad=False)  # pylint: disable=protected-access
  return saved


def _restore_params_(module: nn.Module, saved: dict[str, nn.Parameter]) -> None:
  for fqn, param in saved.items():
    owner, leaf = _owner_of(module, fqn)
    owner._parameters[leaf] = param  # pylint: disable=protected-access


def _materialize_buffers_(module: nn.Module, device: torch.device) -> None:
  """Moves `module`'s buffers off `meta` so `init_states` can recompute them."""
  for submodule in module.modules():
    for name, buf in list(submodule._buffers.items()):  # pylint: disable=protected-access
      if buf is not None and buf.device.type == "meta":
        submodule._buffers[name] = torch.empty_like(buf, device=device)  # pylint: disable=protected-access


class ScannedLayerStack(Module):
  """A `ModuleDict`-compatible stand-in that runs its layers under `scan`.

  Args:
    layers: the per-layer modules, in execution order. They must be structurally
      identical. `__init__` checks the parameter FQN set and the per-FQN shapes
      and raises on a mismatch; dtypes are assumed to agree and are not checked,
      because no configuration this lane can produce varies dtype across layers.
      A stack that did would reach `torch.stack`, which promotes rather than
      raising.
    peel_first_layer: run layer 0 outside the `scan` and scan over layers
      1..N-1. Required when the block calls a JAX-exported kernel — see
      `forward` for the torch_tpu defect this works around.
  """

  def __init__(self, layers: list[nn.Module], *, peel_first_layer: bool = False):
    super().__init__()
    if not layers:
      raise ValueError("ScannedLayerStack needs at least one layer.")

    self.n_layers = len(layers)
    self.peel_first_layer = peel_first_layer

    reference = dict(layers[0].named_parameters())
    self.param_fqns: list[str] = sorted(reference)
    for i, layer in enumerate(layers[1:], start=1):
      other = dict(layer.named_parameters())
      if sorted(other) != self.param_fqns:
        raise ValueError(
            f"layer {i} has parameter FQNs {sorted(other)}, layer 0 has"
            f" {self.param_fqns}; scan requires structurally identical layers."
        )
      for fqn in self.param_fqns:
        if other[fqn].shape != reference[fqn].shape:
          raise ValueError(
              f"layer {i}.{fqn} has shape {tuple(other[fqn].shape)}, layer"
              f" 0.{fqn} has {tuple(reference[fqn].shape)}."
          )

    # Stack along dim 0 — the scan axis. Never shard dim 0 (see the plan's
    # Trap 2): slicing a sharded axis inside the loop forces a dynamic
    # collective per iteration. `parallelize_llama` shards the stack on dim 1.
    for fqn in self.param_fqns:
      slices = [dict(layer.named_parameters())[fqn] for layer in layers]
      stack = torch.stack([s.detach() for s in slices], dim=0)
      self.register_parameter(
          _mangle(fqn),
          nn.Parameter(stack, requires_grad=slices[0].requires_grad),
      )

    # The per-layer modules are kept only as shadows: unregistered (so their
    # `meta` parameters never reach `.parameters()` / `state_dict()` /
    # `to_empty()`) but alive, because they own the `_param_init` closures that
    # `init_states` replays into the stack. Layer 0 doubles as the template
    # `functional_call` runs on.
    self._shadow_layers = list(layers)
    for layer in self._shadow_layers:
      _to_meta_(layer)

    # Set by `set_fsdp_plan` once SimpleFSDP has sharded the stack. Empty means
    # "read the parameters straight through" — the single-device path.
    self._fsdp_plan: dict[str, ScanFSDPSpec] = {}

    logger.info(
        "ScannedLayerStack: %d layers, %d stacked parameter tensors (%s).",
        self.n_layers,
        len(self.param_fqns),
        ", ".join(
            f"{fqn}{tuple(self._parameters[_mangle(fqn)].shape)}"
            for fqn in self.param_fqns[:3]
        )
        + (", …" if len(self.param_fqns) > 3 else ""),
    )

  @property
  def _template(self) -> nn.Module:
    return self._shadow_layers[0]

  # -- device movement + initialization ----------------------------------------

  def _apply(self, fn, recurse=True):
    module = super()._apply(fn, recurse=recurse)
    if recurse:
      # `nn.Module._apply` never reaches the shadow layers — that is the point,
      # their parameters must stay on `meta`. Their *buffers* are small, are
      # read by the scan body (RoPE's `cache`), and so have to follow the module
      # across `.to(device)` / `.to_empty(device)`.
      for layer in self._shadow_layers:
        for submodule in layer.modules():
          for name, buf in list(submodule._buffers.items()):  # pylint: disable=protected-access
            if buf is not None:
              submodule._buffers[name] = fn(buf)  # pylint: disable=protected-access
    return module

  def init_states(self, *, buffer_device: torch.device | None = None) -> None:
    """Initializes the stack by replaying each layer's own initializer into it.

    Binds layer *i*'s parameters to views of `stacked[i]` and calls the stock
    `init_states`, so every slice is initialized by exactly the code the
    unrolled model would have run — including any per-layer-index scaling — and
    nothing larger than the stack is ever allocated.
    """
    # Read through `_parameters`, never `getattr`: after SimpleFSDP the latter
    # returns the all-gathered *replica*, and writing into it would be dropped.
    stacked = {fqn: self._parameters[_mangle(fqn)] for fqn in self.param_fqns}  # pylint: disable=protected-access
    device = next(iter(stacked.values())).device
    if buffer_device is None:
      buffer_device = device

    for i, layer in enumerate(self._shadow_layers):
      saved = _bind_params_(layer, {fqn: stacked[fqn][i] for fqn in stacked})
      _materialize_buffers_(layer, device)
      try:
        layer.init_states(buffer_device=buffer_device)
      finally:
        _restore_params_(layer, saved)

  def stacked_parameters(self) -> list[nn.Parameter]:
    """The `[n_layers, *shape]` parameters, in `param_fqns` order."""
    return [self._parameters[_mangle(fqn)] for fqn in self.param_fqns]  # pylint: disable=protected-access

  def set_fsdp_plan(self, plan: dict[str, ScanFSDPSpec]) -> None:
    """Installs the per-parameter in-body gather plan (see `fsdp.py`)."""
    self._fsdp_plan = plan

  # -- ModuleDict compatibility ------------------------------------------------
  # `Decoder.forward` does `for layer in self.layers.values(): h = layer(h, ...)`.
  # Returning `[self]` makes that loop run once over the whole scanned stack.

  def values(self) -> Iterable[nn.Module]:
    return [self]

  def keys(self) -> Iterable[str]:
    return ["scan"]

  def items(self) -> Iterable[tuple[str, nn.Module]]:
    return [("scan", self)]

  def __len__(self) -> int:
    return self.n_layers

  # -- forward -----------------------------------------------------------------

  def forward(
      self,
      h: torch.Tensor,
      attention_masks: Any = None,
      positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    template = self._template
    fqns = self.param_fqns

    if self.training and not h.requires_grad:
      # A carry that does not require grad makes the backward return zeros for
      # the carry gradient, silently severing the inter-layer chain while
      # keeping the loss bit-exact. Fail loudly instead.
      raise RuntimeError(
          "ScannedLayerStack: the scan init carry does not require grad. The"
          " backward would return zeros_like(carry) and every layer's weight"
          " gradient would reflect only itself, with a bit-exact loss and no"
          " error."
      )

    plan = self._fsdp_plan

    def body(carry, params_slice):
      # One layer's slice of each stacked parameter. Under FSDP the slice is
      # this rank's *shard* of that layer's weight and `gather_slice` issues the
      # all-gather right here, inside the combine_fn -- which is the point: the
      # collective lowers into the `stablehlo.while` body, so only the layer in
      # flight is ever gathered. See `fsdp.py` for why gathering outside
      # (SimpleFSDP's parametrization property) is a 136.9 GB OOM at 70B.
      params = {
          fqn: gather_slice(params_slice[_mangle(fqn)], plan.get(_mangle(fqn)))
          for fqn in fqns
      }
      out = functional_call(
          template, params, (carry, attention_masks, positions)
      )
      # `scan` requires a non-empty per-iteration output: returning `None` (or
      # any empty pytree) trips `all fw_gm output nodes must be torch.fx.Node
      # with 'val' in meta` in the HOP's autograd path. We only want the final
      # carry, so emit the cheapest possible placeholder and discard it.
      return out, out.new_zeros(())

    # `self._parameters`, NOT `getattr`: `getattr` goes through SimpleFSDP's
    # parametrization property, which gathers the whole `[n_layers, ...]` stack
    # eagerly. Read the sharded DTensor and hand `scan` the *local* shard; the
    # body gathers per layer instead.
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
      # symbol and the outer module fails verification — XLA misreports it as
      # "failed to upgrade versioned StableHLO". Running layer 0 outside the
      # scan puts an identically-named copy of that function in the outer
      # module (the symbol is a pure function of kernel name + operand
      # shapes/dtypes), where the call resolves.
      #
      # The peeled layer is checkpointed so that its *backward* recomputes the
      # forward and therefore emits the forward kernel's symbol into the
      # backward module too — which the scanned body's own recompute needs.
      # `torch.utils.checkpoint` is a silent no-op inside the HOP body, but
      # works normally out here.
      head = {mangled: value[0] for mangled, value in xs.items()}
      h, _ = torch.utils.checkpoint.checkpoint(
          body, h, head, use_reentrant=False
      )
      xs = {mangled: value[1:] for mangled, value in xs.items()}

    carry, _ = scan(body, h, xs)
    return carry

  # -- state_dict adapter ------------------------------------------------------
  # On disk we keep the *unrolled* FQNs (`layers.{i}.attention.wo.weight`) so a
  # scanned run and an unrolled run share checkpoints in both directions.

  def stacked_to_unrolled(self) -> dict[str, torch.Tensor]:
    """Returns `{"<i>.<layer-relative fqn>": tensor}` — the ModuleDict layout."""
    out = {}
    for fqn in self.param_fqns:
      stack = self._parameters[_mangle(fqn)]  # pylint: disable=protected-access
      for i in range(self.n_layers):
        out[f"{i}.{fqn}"] = stack[i]
    return out

  @torch.no_grad()
  def load_unrolled(self, unrolled: dict[str, torch.Tensor]) -> None:
    """Loads a `{"<i>.<layer-relative fqn>": tensor}` mapping into the stack."""
    for fqn in self.param_fqns:
      stack = self._parameters[_mangle(fqn)]  # pylint: disable=protected-access
      for i in range(self.n_layers):
        key = f"{i}.{fqn}"
        if key not in unrolled:
          raise KeyError(f"missing {key} in unrolled state dict")
        stack[i].copy_(unrolled[key])


def apply_scan_over_layers(
    model: nn.Module, *, peel_first_layer: bool = False
) -> nn.Module:
  """Swaps `model.layers` (a `ModuleDict` of blocks) for a `ScannedLayerStack`.

  Also installs the real-arithmetic RoPE patch, which is process-global rather
  than scoped to `model` — the swap and the patch are together here because
  every caller that stacks the layers then runs the body, and the body cannot
  be traced without the patch. Idempotent. Returns `model` for chaining.
  """
  layers = getattr(model, "layers", None)
  if isinstance(layers, ScannedLayerStack):
    return model
  if layers is None:
    raise ValueError("model has no `.layers` to scan over.")

  # ComplexRoPE's differentiable `view_as_complex` has no autograd formula
  # under the HOP's functional re-trace; swap in the bit-equivalent real form.
  from torchtitan.experiments.tpu.rope_real_arithmetic import (  # pylint: disable=g-import-not-at-top
      use_real_arithmetic_rope_patch,
  )

  use_real_arithmetic_rope_patch()

  ordered = [layers[str(i)] for i in range(len(layers))]
  model.layers = ScannedLayerStack(ordered, peel_first_layer=peel_first_layer)
  return model


def install_scan_over_layers(
    model: nn.Module, *, parallel_dims: Any, peel_first_layer: bool
) -> "ScannedLayerStack":
  """Stacks the decoder layers and installs the torch_tpu patches a scan needs.

  Shared by the two lanes that can run a scan — the graph_trainer lane's
  `graph_trainer/parallelize_scan.py` and the fullgraph lane's
  `llama3/infra/parallelize.py` — so the ordering constraints below are stated
  once. Must run before anything else rewrites `model.layers`: activation
  checkpointing and FSDP both iterate the `ModuleDict`'s children, and the
  stack has a different child structure.

  Returns the `ScannedLayerStack` now installed at `model.layers`.
  """
  from torchtitan.experiments.tpu.scan import reverse as scan_reverse  # pylint: disable=g-import-not-at-top
  from torchtitan.experiments.tpu.scan.torch_tpu_fix import (  # pylint: disable=g-import-not-at-top
      use_torch_tpu_scan_fix,
  )

  for dim in ("tp", "cp", "pp"):
    if getattr(parallel_dims, f"{dim}_enabled", False):
      raise ValueError(
          f"tpu_config.use_scan does not support {dim.upper()} yet: the"
          " stacked parameters carry a leading layer axis that the existing"
          " parallel plans do not describe."
      )

  # TEMPORARY: torch_tpu's scan lowering needs two local patches before any
  # scan can run on TPU. This is the only call site, and it is reached only
  # when a scan is actually being installed. See torch_tpu_fix.py.
  use_torch_tpu_scan_fix()

  apply_scan_over_layers(model, peel_first_layer=peel_first_layer)

  # `scan`'s backward is a reverse scan that upstream emulates by flipping the
  # saved residual stack, materialising a byte-for-byte duplicate of the largest
  # buffer in the arena. torch_tpu can lower the reverse directly.
  if scan_reverse.enabled():
    scan_reverse.install_scan_reverse_backward()

  return model.layers
