# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TpuGraphTrainer subclass for TPU with optional optimizer-in-graph."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import logging
import os
from typing import Any, Optional

import torch

from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.compiled_optimizer import _build_compiled_optimizer_groups
from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import register as register_tpu_pipeline
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.tools.logging import logger


@dataclass
class TpuGraphTrainerConfig:
    """Subset of job/TPU config carried forward into TpuGraphTrainer."""

    device: str = "tpu"
    attention_backend: str = "sdpa-no-scale"
    moe_enabled: bool = False
    moe_padded_dispatcher: bool = False
    moe_capacity_factor: float = 1.25
    ep_degree: int = 1
    optimizer_in_graph: bool = False
    optimizer_compile: bool = False
    compile_mode: str = "none"
    job_config: Optional[TPUTrainerConfig] = None


def _optimizer_in_graph_enabled(config, tpu_extra) -> bool:
    if tpu_extra is not None and getattr(tpu_extra, "optimizer_in_graph", False):
        return True
    return bool(getattr(config.compile, "optimizer_in_graph", False))


def _optimizer_compile_enabled(config, tpu_extra) -> bool:
    if tpu_extra is not None and getattr(tpu_extra, "optimizer_compile", False):
        return True
    return bool(getattr(config.compile, "optimizer_compile", False))


class TpuGraphTrainer(GraphTrainer):
    """Branched specialization of GraphTrainer for TPU/PJRT execution.

    Core Extensions and Overrides over GraphTrainer:
    1. __init__:
       - Registers TPU pass pipeline (PASS_PIPELINE_REGISTRY["tpu"]).
       - Overrides cross-entropy loss with PallasCrossEntropyLoss when enabled.
       - Hooks TPU-specific metrics and profiling brackets.
       - Initializes compiled optimizer or optimizer-in-graph state.
    2. _make_fx_forward_backward_step (Overridden):
       - Lowers joint fwd+loss+bwd trace applying PASS_PIPELINE_REGISTRY["tpu"] graph passes.
    3. train_step and forward_backward_step (Overridden conditionally):
       - Delegates to super() when standard tracing is used.
       - Branches to _train_step_optimizer_in_graph / _fwd_bwd_adamw_step when optimizer_in_graph=True.

    New TPU/PJRT Methods:
    - _compiled_optimizer_step: Executes compiled_optimizer groups across meshes.
    - _register_optimizer_buffers: Registers persistent optstate buffers for optimizer_in_graph.
    - _train_step_optimizer_in_graph and _fwd_bwd_adamw_step: Joint fwd+loss+bwd+opt execution loop.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GraphTrainer.Config, TPUTrainerConfig):
        pass

    def __init__(self, config, *, tpu_extra: Optional[TpuGraphTrainerConfig] = None) -> None:
        """Override of GraphTrainer.__init__ adding TPU pipeline registration, Pallas loss, and opt-in-graph setup."""
        # Ensure the 'tpu' pass pipeline is registered before GraphTrainer traces/compiles
        register_tpu_pipeline()

        super().__init__(config)

        if isinstance(config, TPUTrainerConfig):
            from torchtitan.experiments.tpu.loss import build_cross_entropy_loss

            self.loss_fn = build_cross_entropy_loss(config)

        if _split_optimizers_by_mesh(self.optimizers):
            self.lr_schedulers = config.lr_scheduler.build(
                optimizers=self.optimizers,
                training_steps=config.training.steps,
            )
            self.metrics_processor.lr_schedulers = self.lr_schedulers

        for opt in self.optimizers.optimizers:
            for group in opt.param_groups:
                group["capturable"] = True

        self._tpu_extra = tpu_extra
        if tpu_extra is not None and tpu_extra.device == "tpu":
            _apply_tpu_moe_post_init(self.model_parts, tpu_extra)

        # Only dump weight/buffer statistics when debug logging is explicitly enabled per review comment.
        if logger.isEnabledFor(logging.DEBUG):
            _dump_weight_stats(self.model_parts)

        self._optimizer_in_graph = _optimizer_in_graph_enabled(config, tpu_extra)
        if self._optimizer_in_graph:
            self._opt_state_registered = False

        self._optimizer_compile = (
            _optimizer_compile_enabled(config, tpu_extra) and not self._optimizer_in_graph
        )
        if self._optimizer_compile:
            self._compiled_optimizers: list[Optional[Any]] = [
                None for _ in self.optimizers.optimizers
            ]
            _container = self.optimizers
            _trainer = self

            def _patched_step(closure: Any = None) -> None:
                assert closure is None, "compiled optimizer step does not support closures"
                _trainer._compiled_optimizer_step()

            _container.step = _patched_step  # type: ignore[assignment]
            print(
                f"[compiled_opt] Separate optimizer-graph compile enabled — patched OptimizersContainer.step "
                f"({len(self.optimizers.optimizers)} inner optimizer(s)).",
                flush=True,
            )

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> None:
        """Override of GraphTrainer.train_step to branch into optimizer_in_graph loop when enabled."""
        if not self._optimizer_in_graph:
            return super().train_step(data_iterator)
        return self._train_step_optimizer_in_graph(data_iterator)

    def _compiled_optimizer_step(self) -> None:
        """TPU/PJRT addition: executes compiled optimizer groups across meshes."""
        for i, opt in enumerate(self.optimizers.optimizers):
            if self._compiled_optimizers[i] is None:
                self._compiled_optimizers[i] = _build_compiled_optimizer_groups(opt)
            for fn, params, state_refs, lr_ref in self._compiled_optimizers[i]:
                grads = [p.grad for p in params]
                if any(g is None for g in grads):
                    opt.step()
                    continue
                exp_avgs = [s["exp_avg"] for s in state_refs]
                exp_avg_sqs = [s["exp_avg_sq"] for s in state_refs]
                steps = [s["step"] for s in state_refs]
                lr_ref.fill_(float(opt.param_groups[lr_ref._group_idx]["lr"]))  # type: ignore[attr-defined]
                outs = fn(params, grads, exp_avgs, exp_avg_sqs, steps, lr_ref)
                n = len(params)
                new_params = outs[:n]
                new_exp_avgs = outs[n : 2 * n]
                new_exp_avg_sqs = outs[2 * n : 3 * n]
                new_steps = outs[3 * n : 4 * n]
                for p, np_ in zip(params, new_params, strict=True):
                    p.data.copy_(np_)
                for ea, nea in zip(exp_avgs, new_exp_avgs, strict=True):
                    ea.copy_(nea)
                for eas, neas in zip(exp_avg_sqs, new_exp_avg_sqs, strict=True):
                    eas.copy_(neas)
                for s, ns in zip(steps, new_steps, strict=True):
                    s.copy_(ns)

    def _register_optimizer_buffers(self) -> None:
        """TPU/PJRT addition: registers persistent optstate buffers when optimizer_in_graph is enabled."""
        if self._opt_state_registered:
            return
        assert len(self.model_parts) == 1, (
            "optimizer_in_graph supports exactly one model part."
        )
        model = self.model_parts[0]
        seen_ids: dict[int, str] = {}
        param_fqns: list[str] = []
        fqn_to_canonical: dict[str, str] = {}
        for name, p in model.named_parameters(remove_duplicate=False):
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen_ids:
                fqn_to_canonical[name] = seen_ids[pid]
                continue
            seen_ids[pid] = name
            fqn_to_canonical[name] = name
            param_fqns.append(name)
            parent_path, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            parent.register_buffer(
                f"_optstate_{leaf}_exp_avg",
                torch.zeros_like(p, requires_grad=False),
                persistent=True,
            )
            parent.register_buffer(
                f"_optstate_{leaf}_exp_avg_sq",
                torch.zeros_like(p, requires_grad=False),
                persistent=True,
            )
        model.register_buffer(
            "_optstate_step",
            torch.zeros((), dtype=torch.float32, device=self.device),
            persistent=True,
        )
        lr0 = float(self.lr_schedulers.schedulers[0].get_last_lr()[0])
        model.register_buffer(
            "_optstate_lr",
            torch.tensor(lr0, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self._optstate_param_fqns = param_fqns
        self._optstate_fqn_to_canonical = fqn_to_canonical
        self._opt_state_registered = True

    def _train_step_optimizer_in_graph(
        self,
        data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]],
    ) -> None:
        """TPU/PJRT addition: runs joint fwd+loss+bwd+opt training loop when optimizer_in_graph is active."""
        from torchtitan.components.loss import IGNORE_INDEX
        from torchtitan.distributed import utils as dist_utils

        self._register_optimizer_buffers()

        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
        parallel_dims = self.parallel_dims

        model = self.model_parts[0]
        model._optstate_lr.fill_(lr)

        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _ in range(self.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()
            microbatches.append((input_dict, labels))
        local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum(
                local_valid_tokens, batch_mesh
            )
        else:
            global_valid_tokens = local_valid_tokens.float()

        accumulated_losses = []
        for input_dict, labels in microbatches:
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            labels = labels.to(self.device)
            loss = self.forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,  # type: ignore[arg-type]
            )
            accumulated_losses.append(loss.detach())

        self.lr_schedulers.step()
        loss = torch.sum(torch.stack(accumulated_losses))

        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            global_avg_loss = dist_utils.dist_sum(loss.detach(), loss_mesh)
            global_max_loss = global_avg_loss
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())

        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            0.0,
            extra_metrics={"lr": lr, "n_tokens_seen": self.ntokens_seen},
        )

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Override of GraphTrainer.forward_backward_step to branch into AdamW step when optimizer_in_graph is set."""
        if not self._optimizer_in_graph:
            return super().forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )
        return self._fwd_bwd_adamw_step(
            input_dict=input_dict,
            labels=labels,
            global_valid_tokens=global_valid_tokens,
        )

    def _make_fx_forward_backward_step(
        self,
        model: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        params: list[torch.Tensor],
        extra_inputs: dict[str, torch.Tensor],
        extra_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Override of GraphTrainer._make_fx_forward_backward_step applying TPU pipeline graph passes."""
        from torchtitan.experiments.graph_trainer.common_utils import (
            maybe_register_blockmask_pytree_node,
        )
        from torchtitan.experiments.graph_trainer.make_fx_tracer import (
            run_traced_train_step,
            trace_train_step,
        )
        from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step
        from torchtitan.experiments.graph_trainer.passes import apply_graph_passes
        from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
            PASS_PIPELINE_REGISTRY,
        )

        if self._traced_step is None:
            if self.config.compile.precompile_artifact_dir:
                self._load_precompiled_fx_trace(model)
            else:
                fwd_bwd_fn = make_fwd_bwd_step(self.loss_fn)
                maybe_register_blockmask_pytree_node()
                with self.train_context():
                    self._traced_step = trace_train_step(fwd_bwd_fn)(
                        model,
                        inputs,
                        labels,
                        global_valid_tokens,
                        extra_inputs,
                        extra_kwargs,
                    )

            if self.config.compile.enable_passes:
                pipeline_fn = PASS_PIPELINE_REGISTRY["tpu"]
                passes = pipeline_fn(self._traced_step, self.config)
                self._traced_step.gm = apply_graph_passes(
                    self._traced_step.gm,
                    self._traced_step.example_inputs,
                    passes,
                    compile_config=self.config.compile,
                )
        with self.train_context():
            outputs = run_traced_train_step(
                self._traced_step,
                model,
                inputs,
                labels,
                global_valid_tokens,
                extra_inputs,
                extra_kwargs,
            )
        loss = outputs[0]
        grads = outputs[1:]

        for param, grad in zip(params, grads, strict=True):
            if param.requires_grad and grad is not None:
                param.grad = grad

        return loss

    def _fwd_bwd_adamw_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """TPU/PJRT addition: lowers and runs joint forward, backward, and AdamW optimizer FX graph step."""
        from torchtitan.experiments.graph_trainer.common_utils import (
            maybe_register_blockmask_pytree_node,
        )
        from torchtitan.experiments.graph_trainer.make_fx_tracer import (
            run_traced_train_step,
            trace_train_step,
        )
        from torchtitan.experiments.graph_trainer.passes import apply_graph_passes
        from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
            PASS_PIPELINE_REGISTRY,
        )

        assert len(self.model_parts) == 1
        model = self.model_parts[0]

        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )

        if self._traced_step is None:
            maybe_register_blockmask_pytree_node()
            param_fqns = list(self._optstate_param_fqns)
            betas = tuple(self.optimizers.optimizers[0].param_groups[0]["betas"])
            wd = float(self.optimizers.optimizers[0].param_groups[0]["weight_decay"])
            eps = float(self.optimizers.optimizers[0].param_groups[0]["eps"])

            fwd_bwd_adamw = _make_fwd_bwd_adamw_fn(
                loss_fn=self.loss_fn,
                param_fqns=param_fqns,
                beta1=float(betas[0]),
                beta2=float(betas[1]),
                eps=eps,
                weight_decay=wd,
            )

            with self.train_context():
                self._traced_step = trace_train_step(fwd_bwd_adamw)(
                    model,
                    inputs,
                    labels,
                    global_valid_tokens,
                    extra_inputs,
                    extra_kwargs,
                )
            pipeline_fn = PASS_PIPELINE_REGISTRY["tpu"]
            passes = pipeline_fn(self._traced_step, self.config)
            self._traced_step.gm = apply_graph_passes(
                self._traced_step.gm,
                self._traced_step.example_inputs,
                passes,
                compile_config=self.config.compile,
            )

        with self.train_context():
            outputs = run_traced_train_step(
                self._traced_step,
                model,
                inputs,
                labels,
                global_valid_tokens,
                extra_inputs,
                extra_kwargs,
            )

        loss = outputs[0]
        n = len(self._optstate_param_fqns)
        new_params = outputs[1 : 1 + n]
        new_exp_avgs = outputs[1 + n : 1 + 2 * n]
        new_exp_avg_sqs = outputs[1 + 2 * n : 1 + 3 * n]
        new_step = outputs[1 + 3 * n]

        for fqn, new_p in zip(self._optstate_param_fqns, new_params, strict=True):
            live = _get_param_by_fqn(model, fqn)
            live.data.copy_(new_p.detach() if isinstance(new_p, torch.Tensor) else new_p)
        for fqn, new_ea in zip(self._optstate_param_fqns, new_exp_avgs, strict=True):
            buf = _get_optstate_buffer(model, fqn, "exp_avg")
            buf.copy_(new_ea.detach() if isinstance(new_ea, torch.Tensor) else new_ea)
        for fqn, new_eas in zip(
            self._optstate_param_fqns, new_exp_avg_sqs, strict=True
        ):
            buf = _get_optstate_buffer(model, fqn, "exp_avg_sq")
            buf.copy_(new_eas.detach() if isinstance(new_eas, torch.Tensor) else new_eas)
        model._optstate_step.copy_(
            new_step.detach() if isinstance(new_step, torch.Tensor) else new_step
        )

        return loss


def _get_param_by_fqn(model, fqn: str) -> torch.Tensor:
    parent_path, _, leaf = fqn.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    params = getattr(parent, "_parameters", None)
    if params is not None and leaf in params:
        return params[leaf]
    return getattr(parent, leaf)


def _get_optstate_buffer(model, param_fqn: str, kind: str) -> torch.Tensor:
    parent_path, _, leaf = param_fqn.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    return getattr(parent, f"_optstate_{leaf}_{kind}")


def _make_fwd_bwd_adamw_fn(
    *,
    loss_fn,
    param_fqns: list[str],
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
):
    from torch.fx.traceback import annotate_fn
    from torchtitan.experiments.graph_trainer.common_utils import _MODULE_FQN

    def fwd_bwd_adamw(
        model, inputs, labels, global_valid_tokens, extra_inputs, extra_kwargs
    ):
        pred = model(inputs, **extra_inputs, **extra_kwargs)
        loss = annotate_fn({_MODULE_FQN: "loss"})(loss_fn)(
            pred, labels, global_valid_tokens
        )

        canonical_params: list[torch.Tensor] = [
            _get_param_by_fqn(model, fqn) for fqn in param_fqns
        ]
        grads = torch.autograd.grad(loss, canonical_params)

        exp_avgs = [
            _get_optstate_buffer(model, fqn, "exp_avg") for fqn in param_fqns
        ]
        exp_avg_sqs = [
            _get_optstate_buffer(model, fqn, "exp_avg_sq") for fqn in param_fqns
        ]
        step_t = model._optstate_step
        lr_t = model._optstate_lr

        import torch.optim.adamw as _adamw_mod

        n = len(canonical_params)
        new_params = [p.clone() for p in canonical_params]
        new_eas = [ea.clone() for ea in exp_avgs]
        new_eass = [eas.clone() for eas in exp_avg_sqs]
        steps_c = [step_t.clone() for _ in range(n)]
        grads_cast = [g.to(p.dtype) for g, p in zip(grads, canonical_params)]
        _adamw_mod.adamw(
            new_params,
            grads_cast,
            new_eas,
            new_eass,
            [],
            steps_c,
            foreach=False,
            capturable=True,
            differentiable=False,
            fused=False,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=lr_t,
            weight_decay=weight_decay,
            eps=eps,
            maximize=False,
        )
        return [loss, *new_params, *new_eas, *new_eass, steps_c[0]]

    return fwd_bwd_adamw


def _split_optimizers_by_mesh(optimizers_container) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        DTensor = None

    def _mesh_key(p):
        if DTensor is None or not isinstance(p, DTensor):
            return None
        try:
            return id(p._spec.mesh)
        except Exception:
            return None

    did_split = False
    for opt in optimizers_container.optimizers:
        new_groups: list[dict] = []
        for grp in opt.param_groups:
            params = list(grp["params"])
            base_kwargs = {k: v for k, v in grp.items() if k != "params"}
            buckets: dict = {}
            for p in params:
                key = _mesh_key(p)
                buckets.setdefault(key, []).append(p)
            if len(buckets) > 1:
                did_split = True
            for bucket_params in buckets.values():
                new_groups.append({"params": bucket_params, **base_kwargs})
        if len(new_groups) == len(opt.param_groups):
            continue
        opt.param_groups = []
        opt.state.clear()
        for grp in new_groups:
            opt.add_param_group(grp)
    return did_split


def _dump_weight_stats(model_parts) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        DTensor = None
    logger.debug("[debug_weights] === post-init weight stats (rank 0) ===")
    for model in model_parts:
        for name, p in model.named_parameters():
            t = p
            if DTensor is not None and isinstance(p, DTensor):
                t = p._local_tensor
            try:
                t_cpu = t.detach().to("cpu").float()
                has_nan = bool(torch.isnan(t_cpu).any().item())
                has_inf = bool(torch.isinf(t_cpu).any().item())
                lo = float(t_cpu.min().item())
                hi = float(t_cpu.max().item())
                me = float(t_cpu.mean().item())
                st = float(t_cpu.std().item())
                logger.debug(
                    f"[debug_weights]   {name}: shape={tuple(t.shape)} "
                    f"min={lo:+.4f} max={hi:+.4f} mean={me:+.5f} std={st:.5f} "
                    f"nan={has_nan} inf={has_inf}"
                )
            except Exception as e:
                logger.debug(f"[debug_weights]   {name}: <error {type(e).__name__}: {e}>")
    for model in model_parts:
        for name, buf in model.named_buffers():
            t = buf
            if DTensor is not None and isinstance(buf, DTensor):
                t = buf._local_tensor
            try:
                t_cpu = t.detach().to("cpu").float()
                has_nan = bool(torch.isnan(t_cpu).any().item())
                lo = float(t_cpu.min().item())
                hi = float(t_cpu.max().item())
                logger.debug(
                    f"[debug_weights]   buf {name}: shape={tuple(t.shape)} "
                    f"min={lo:+.4f} max={hi:+.4f} nan={has_nan}"
                )
            except Exception as e:
                logger.debug(f"[debug_weights]   buf {name}: <error {type(e).__name__}: {e}>")
    logger.debug("[debug_weights] === end ===")




def _apply_tpu_moe_post_init(model_parts, tpu_extra: TpuGraphTrainerConfig) -> None:
    """Apply our own experiments/tpu/workarounds.py patches post-super().__init__."""
    for model in model_parts:
        if tpu_extra.job_config is not None:
            workarounds.apply_patches(model, tpu_extra.job_config)
