"""GPT-OSS joint forward/backward tracing with the TPU SDPA baseline."""

import dataclasses
from dataclasses import dataclass

from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    apply_simple_fsdp,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    disable_active_parametrization,
)
from torchtitan.experiments.tpu.gpt_oss import model_registry as baseline_registry
from torchtitan.experiments.tpu.graph_trainer.scan_fsdp import (
    configure_simple_fsdp_shard_dims,
)
from torchtitan.models.gpt_oss.model import GptOssModel


class GraphTrainerGptOssModel(GptOssModel):
    @dataclass(kw_only=True, slots=True)
    class Config(GptOssModel.Config):
        pass

    def init_states(self, *, buffer_device=None):
        with disable_active_parametrization():
            super().init_states(buffer_device=buffer_device)


def parallelize_gptoss(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    dump_folder,
    **kwargs,
):
    if any(
        (
            parallel_dims.tp_enabled,
            parallel_dims.cp_enabled,
            parallel_dims.pp_enabled,
            parallel_dims.ep_enabled,
        )
    ):
        raise NotImplementedError(
            "GPT-OSS TPU GraphTrainer supports SimpleFSDP/HSDP only"
        )
    annotate_module_fqns(model)
    configure_simple_fsdp_shard_dims(model, parallel_dims)
    model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)
    return apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )


def model_registry(flavor, *, seq_len=None):
    spec = baseline_registry(flavor, seq_len=seq_len)
    model = GraphTrainerGptOssModel.Config(
        **{
            field.name: getattr(spec.model, field.name)
            for field in dataclasses.fields(spec.model)
        }
    )
    return dataclasses.replace(
        spec,
        model=model,
        parallelize_fn=parallelize_gptoss,
        pipelining_fn=None,
    )
