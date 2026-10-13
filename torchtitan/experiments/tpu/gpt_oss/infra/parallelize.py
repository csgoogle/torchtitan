"""GPT-OSS data parallelism through the upstream and SimpleFSDP seams."""

import dataclasses

import torch

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import (
    configure_simple_fsdp_shard_dims,
)
from torchtitan.models.gpt_oss.parallelize import parallelize_gptoss as upstream


def parallelize_gptoss(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
    skip_dp=False,
    tpu_compile_mode="layer",
    **kwargs,
):
    """Apply AC and per-block compile before FSDP; keep unsupported modes explicit."""
    if any(
        (
            parallel_dims.tp_enabled,
            parallel_dims.cp_enabled,
            parallel_dims.pp_enabled,
            parallel_dims.ep_enabled,
        )
    ):
        raise NotImplementedError("GPT-OSS TPU baseline supports FSDP/HSDP only")
    if tpu_compile_mode != "layer":
        raise NotImplementedError("GPT-OSS TPU baseline requires per-block compile")
    model = upstream(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=dataclasses.replace(compile_config, enable=False),
        ac_config=ac_config,
        dump_folder=dump_folder,
        skip_dp=True,
    )
    if compile_config.enable and "model" in compile_config.components:
        # The alternating window/global blocks must not acquire symbolic dims
        # when Dynamo sees the second block type.
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit, 2 * len(model.layers)
        )
        for layer in model.layers.values():
            layer.compile(backend=compile_config.backend, fullgraph=True, dynamic=False)
    if skip_dp:
        return model
    if parallelism.use_simple_fsdp:
        configure_simple_fsdp_shard_dims(model, parallel_dims)
        model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)
    else:
        mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        apply_fsdp_to_decoder(
            model,
            parallel_dims.get_mesh(mesh_names),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=False,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            dp_mesh_dims=None,
            enable_symm_mem=parallelism.enable_fsdp_symm_mem,
        )
    return model
