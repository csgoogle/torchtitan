# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Register and construct the 'tpu' pass pipeline for torchtitan's GraphTrainer."""

from __future__ import annotations

from collections import Counter
import functools
import logging
import os
import time
from typing import Any, Callable

import torch

from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    remove_parameter_gradient_markers_pass,
)
from torchtitan.experiments.tpu.graph_trainer.remat_schedule import (
    apply_tpu_remat_schedule,
)
from torchtitan.tools.logging import logger


def _import_static_compiler():
    """Lazy import so this module is safe to import across environments."""
    from torch_tpu._internal.compile.compiler import StaticCompiler  # pytype: disable=import-error # noqa: F401
    return StaticCompiler


def tpu_backend_compile_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
) -> torch.fx.GraphModule:
    """Terminal pass in the TPU graph pipeline: lowers gm into a static XLA binary.

    This pass inspects the traced GraphModule (`gm`), logs op and collective statistics
    when debug logging is enabled, filters out non-Tensor slots if necessary, and
    compiles the graph using `StaticCompiler` (`lazy PJRT/XLA lowering`). The forward
    method of the module is then rebound to execute the compiled XLA executable directly,
    eliminating per-node host dispatch overhead during training steps.
    """
    StaticCompiler = _import_static_compiler()

    # Only iterate over nodes and count op/collective statistics when debug logging is enabled.
    if logger.isEnabledFor(logging.DEBUG):
        op_counter: Counter = Counter()
        for n in gm.graph.nodes:
            if n.op == "call_function":
                op_counter[str(n.target)] += 1
        ag = sum(c for t, c in op_counter.items() if "all_gather" in t)
        rs = sum(c for t, c in op_counter.items() if "reduce_scatter" in t)
        wait = sum(c for t, c in op_counter.items() if "wait_tensor" in t)
        ob = sum(c for t, c in op_counter.items() if "optimization_barrier" in t)
        n_nodes = sum(1 for _ in gm.graph.nodes)
        n_call = sum(1 for n in gm.graph.nodes if n.op == "call_function")
        logger.debug(
            f"[tpu_backend_pass] graph in: {n_nodes} nodes "
            f"({n_call} call_function); collectives: {ag} all_gather, "
            f"{rs} reduce_scatter, {wait} wait_tensor; "
            f"optimization_barrier nodes: {ob}"
        )

    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    original_outs = output_node.args[0]
    tensor_outs: list = []
    non_tensor_slots: list[tuple[int, object]] = []
    for i, out in enumerate(original_outs):
        if out is None:
            tensor_outs.append(out)
        elif isinstance(out, torch.fx.Node):
            val = out.meta.get("val")
            if val is None or isinstance(val, torch.Tensor):
                tensor_outs.append(out)
            else:
                non_tensor_slots.append((i, val))
        else:
            non_tensor_slots.append((i, out))
    if non_tensor_slots:
        logger.info(
            f"[tpu_backend_pass] filtering {len(non_tensor_slots)} non-Tensor "
            f"output(s) before TpuBackend lowering: positions "
            f"{[p for p, _ in non_tensor_slots]}"
        )
        output_node.args = (tuple(tensor_outs),)
        gm.graph.lint()
        gm.recompile()

    t0 = time.time()
    logger.info("[tpu_backend_pass] StaticCompiler(gm, example_inputs) ...")
    executable = StaticCompiler()(gm, list(example_inputs))
    elapsed = time.time() - t0
    logger.info(
        f"[tpu_backend_pass] StaticCompiler ready in {elapsed:.1f}s "
        "(PJRT compile finishes lazily on first execute)"
    )

    gm._tpu_executable = executable

    if non_tensor_slots:
        def _compiled_forward(*flat_inputs):
            outs = list(executable(*flat_inputs))
            for pos, val in non_tensor_slots:
                outs.insert(pos, val)
            return tuple(outs)
    else:
        def _compiled_forward(*flat_inputs):
            return executable(*flat_inputs)

    gm.forward = _compiled_forward  # type: ignore[method-assign]
    return gm


def construct_tpu_pass_pipeline(
    traced_result: "TracedResult",
    config: "GraphTrainer.Config",
) -> list[Callable]:
    """Build the pass list for the 'tpu' pipeline."""
    passes: list[Callable] = [remove_parameter_gradient_markers_pass]

    memory_policy = getattr(config.compile, "memory_policy", "none")
    if memory_policy and memory_policy != "none":
        from torchtitan.experiments.graph_trainer.passes import (
            selective_activation_remat_pass,
            tag_with_memory_policy_pass,
        )

        passes.append(
            functools.partial(tag_with_memory_policy_pass, config=config)
        )
        passes.append(selective_activation_remat_pass)
        # This pass pipeline is constructed only by TpuGraphTrainer. Do not
        # gate on use_graph_trainer here: compile_mode="aot_fx_trace" also
        # selects TpuGraphTrainer without setting that legacy selector.
        if config.tpu_config.enable_graph_trainer_remat_schedule:
            passes.append(
                functools.partial(
                    apply_tpu_remat_schedule,
                    num_static_inputs=traced_result.num_static_inputs,
                )
            )
        logger.info(
            f"[tpu_backend_pass] activation checkpointing enabled "
            f"(memory_policy={memory_policy!r}); prepended "
            f"tag_with_memory_policy + selective_activation_remat"
        )

    from torchtitan.experiments.tpu.graph_trainer.passes.insert_combine_barriers import (
        apply_combine_barrier_pass,
    )
    passes.append(apply_combine_barrier_pass)

    passes.append(tpu_backend_compile_pass)
    return passes


# Global registry dict so both construct_tpu_pass_pipeline and PASS_PIPELINE_REGISTRY["tpu"] work
PASS_PIPELINE_REGISTRY: dict[str, Callable] = {
    "tpu": construct_tpu_pass_pipeline
}


def register() -> None:
    """Ensure the 'tpu' pipeline is registered."""
    PASS_PIPELINE_REGISTRY["tpu"] = construct_tpu_pass_pipeline


__all__ = [
    "tpu_backend_compile_pass",
    "construct_tpu_pass_pipeline",
    "register",
    "PASS_PIPELINE_REGISTRY",
]
