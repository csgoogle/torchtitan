# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Functional optimizer step implementations and state buffer helpers for PyTorch-on-TPU.

Provides stateless functional optimizer kernels (such as `functional_adamw_step`)
and helper utilities for registering and synchronizing model parameter buffers when
running in-graph optimizer execution under `TpuGraphTrainer`.
"""

from typing import Any, Iterable, List, Mapping, Sequence, Tuple
import torch
from torch import nn
import torch.optim.adamw as _adamw_mod


def adamw_donated_input_indices(
    state_fqns: Sequence[str],
    input_subclass_layouts: Mapping[int, Any],
    *,
    tied_fqns: Iterable[str] = (),
) -> list[int]:
    """Graph-input indices of the parameters and AdamW moments updated in-graph.

    Only supported for Adam/AdamW: ownership is decided by the presence of an
    ``exp_avg`` sibling and the donated set is exactly the two moments AdamW
    keeps per parameter. An optimizer with a different state shape donates
    nothing here rather than donating the wrong buffers.

    The in-graph optimizer returns a fresh tensor for every parameter,
    ``exp_avg`` and ``exp_avg_sq`` it owns. Without donation XLA keeps the input
    and the output copies of all three resident at once; donating the inputs
    lets it alias them (``jax.buffer_donor`` -> ``input_output_alias``).

    A parameter is optimizer-owned iff ``_register_optimizer_buffers`` gave it a
    sibling ``_optstate_<leaf>_exp_avg`` buffer; the moments are the
    ``_optstate_*_exp_avg`` / ``_optstate_*_exp_avg_sq`` buffers themselves
    (``_optstate_step`` / ``_optstate_lr`` are not donated). ``state_fqns`` are
    the leading graph inputs in order; a wrapper-subclass state leaf (a DTensor
    parameter) occupies ``input_subclass_layouts[i].num_tensors`` consecutive
    plain graph inputs, all of which are donated.

    ``tied_fqns`` names parameters that appear under more than one FQN. The
    tracer keeps one graph input per FQN, so a tied parameter is the same buffer
    in two argument slots; donating either is ``f(donate(a), a)``, which PJRT
    rejects at execute time. Tied parameters are therefore left un-donated;
    their moments are unique buffers and are still donated.
    """
    names = set(state_fqns)
    tied = set(tied_fqns)

    def _optstate_sibling(fqn: str, kind: str) -> str:
        parent, _, leaf = fqn.rpartition(".")
        return f"{parent}._optstate_{leaf}_{kind}" if parent else f"_optstate_{leaf}_{kind}"

    donated: list[int] = []
    flat_index = 0
    for i, fqn in enumerate(state_fqns):
        leaf = fqn.rpartition(".")[2]
        is_moment = leaf.startswith("_optstate_") and (
            leaf.endswith("_exp_avg") or leaf.endswith("_exp_avg_sq")
        )
        is_owned_param = _optstate_sibling(fqn, "exp_avg") in names and fqn not in tied
        layout = input_subclass_layouts.get(i)
        width = layout.num_tensors if layout is not None else 1
        if is_moment or is_owned_param:
            donated.extend(range(flat_index, flat_index + width))
        flat_index += width
    return donated


def _get_param_by_fqn(model: nn.Module, fqn: str) -> torch.Tensor:
    """Return the underlying parameter Tensor for ``fqn``.

    Directly accesses ``parent._parameters[leaf]`` when present rather than
    calling ``getattr(parent, leaf)``. This avoids triggering property getters
    (such as simple_fsdp's all-gather parametrization), which return a fresh
    transient tensor disconnected from the autograd graph.

    Falls back to ``getattr`` for plain modules without a ``_parameters`` entry.
    """
    parent_path, _, leaf = fqn.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    params = getattr(parent, "_parameters", None)
    if params is not None and leaf in params:
        return params[leaf]
    return getattr(parent, leaf)


def _get_optstate_buffer(model: nn.Module, fqn: str, kind: str) -> torch.Tensor:
    """Return the live ``_optstate_<leaf>_<kind>`` buffer for a param.

    Mirrors the registration convention in ``_register_optimizer_buffers``:
    the buffer lives on the same parent submodule as the param.
    """
    parent_path, _, leaf = fqn.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    return getattr(parent, f"_optstate_{leaf}_{kind}")


def update_model_and_optstate_buffers(
    model: nn.Module,
    param_fqns: Sequence[str],
    new_params: Sequence[torch.Tensor],
    new_exp_avgs: Sequence[torch.Tensor],
    new_exp_avg_sqs: Sequence[torch.Tensor],
) -> None:
    """Updates live model parameters and optimizer state buffers in-place from graph outputs."""
    for fqn, new_p in zip(param_fqns, new_params, strict=True):
        live = _get_param_by_fqn(model, fqn)
        live.data.copy_(new_p.detach() if isinstance(new_p, torch.Tensor) else new_p)
    for fqn, new_ea in zip(param_fqns, new_exp_avgs, strict=True):
        buf = _get_optstate_buffer(model, fqn, "exp_avg")
        buf.copy_(new_ea.detach() if isinstance(new_ea, torch.Tensor) else new_ea)
    for fqn, new_eas in zip(param_fqns, new_exp_avg_sqs, strict=True):
        buf = _get_optstate_buffer(model, fqn, "exp_avg_sq")
        buf.copy_(new_eas.detach() if isinstance(new_eas, torch.Tensor) else new_eas)


def _parent_and_leaf(model: nn.Module, fqn: str) -> Tuple[nn.Module, str]:
    parent_path, _, leaf = fqn.rpartition(".")
    return (model.get_submodule(parent_path) if parent_path else model), leaf


def rebind_model_and_optstate_buffers(
    model: nn.Module,
    param_fqns: Sequence[str],
    new_params: Sequence[torch.Tensor],
    new_exp_avgs: Sequence[torch.Tensor],
    new_exp_avg_sqs: Sequence[torch.Tensor],
    fqn_to_canonical: Mapping[str, str] | None = None,
    optimizers: Iterable[torch.optim.Optimizer] = (),
) -> None:
    """Replaces live model parameters and optimizer state buffers with graph outputs.

    Used when the step program donates the parameter and optimizer-state inputs
    to their outputs (``donated_inputs`` in the TPU backend pass): the input
    buffers are consumed by the execution, so the outputs replace them in the
    module's ``_parameters`` / ``_buffers`` rather than being copied back.
    ``fqn_to_canonical`` (from ``_register_optimizer_buffers``) maps every
    parameter FQN, including aliases of a tied parameter, to the FQN that owns
    the optimizer state; all aliases are rebound to the same new Parameter so
    tying survives the replacement.

    ``optimizers`` are the ``torch.optim.Optimizer`` objects whose
    ``param_groups`` reference the model's parameters. The in-graph step never
    calls them, but checkpointing does (``OptimizersContainer.state_dict`` runs
    ``init_optim_state``, a zero-lr step over ``param_groups``), so their entries
    are swapped to the new Parameters too; otherwise they would keep pointing
    at the consumed buffers.
    """
    aliases: dict[str, list[str]] = {}
    for fqn, canonical in (fqn_to_canonical or {}).items():
        aliases.setdefault(canonical, []).append(fqn)
    replaced: dict[int, nn.Parameter] = {}
    for fqn, new_p in zip(param_fqns, new_params, strict=True):
        live = _get_param_by_fqn(model, fqn)
        new_param = nn.Parameter(new_p.detach(), requires_grad=live.requires_grad)
        replaced[id(live)] = new_param
        for target in aliases.get(fqn, [fqn]):
            parent, leaf = _parent_and_leaf(model, target)
            parent._parameters[leaf] = new_param
    for kind, news in (("exp_avg", new_exp_avgs), ("exp_avg_sq", new_exp_avg_sqs)):
        for fqn, new in zip(param_fqns, news, strict=True):
            parent, leaf = _parent_and_leaf(model, fqn)
            parent._buffers[f"_optstate_{leaf}_{kind}"] = new.detach()
    for opt in optimizers:
        for group in opt.param_groups:
            group["params"] = [replaced.get(id(p), p) for p in group["params"]]
        for old_id, new_param in replaced.items():
            for key in list(opt.state):
                if id(key) == old_id:
                    opt.state[new_param] = opt.state.pop(key)


def functional_adamw_step(
    params: Sequence[torch.Tensor],
    grads: Sequence[torch.Tensor],
    exp_avgs: Sequence[torch.Tensor],
    exp_avg_sqs: Sequence[torch.Tensor],
    steps: Sequence[torch.Tensor],
    lr: torch.Tensor,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    maximize: bool = False,
    foreach: bool = True,
    capturable: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
  """Runs AdamW functionally by cloning state tensors before in-place modification.

    Args:
        params: Trainable parameter tensors.
        grads: Gradient tensors corresponding to params.
        exp_avgs: Exponential moving average of gradient tensors.
        exp_avg_sqs: Exponential moving average of squared gradient tensors.
        steps: Step counter tensors (one per parameter).
        lr: Scalar tensor representing learning rate.
        beta1: Coefficient used for computing running averages of gradient.
        beta2: Coefficient used for computing running averages of squared gradient.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Weight decay (L2 penalty).
        maximize: Maximize the params based on the objective, instead of minimizing.
        foreach: Whether to use the foreach implementation of AdamW.
        capturable: Whether the optimizer step is capturable in CUDA/XLA graphs.

    Returns:
        A tuple of (new_params, new_exp_avgs, new_exp_avg_sqs, new_steps).
    """
  new_params = [p.clone() for p in params]
  new_eas = [ea.clone() for ea in exp_avgs]
  new_eass = [eas.clone() for eas in exp_avg_sqs]
  new_steps = [s.clone() for s in steps]
  grads_cast = [g.to(p.dtype) for g, p in zip(grads, params, strict=True)]

  _adamw_mod.adamw(
      new_params,
      grads_cast,
      new_eas,
      new_eass,
      [],
      new_steps,
      foreach=foreach,
      capturable=capturable,
      differentiable=False,
      fused=False,
      grad_scale=None,
      found_inf=None,
      has_complex=False,
      amsgrad=False,
      beta1=beta1,
      beta2=beta2,
      lr=lr,
      weight_decay=weight_decay,
      eps=eps,
      maximize=maximize,
  )
  return new_params, new_eas, new_eass, new_steps


__all__ = [
    "_get_param_by_fqn",
    "_get_optstate_buffer",
    "adamw_donated_input_indices",
    "functional_adamw_step",
    "update_model_and_optstate_buffers",
    "rebind_model_and_optstate_buffers",
]
