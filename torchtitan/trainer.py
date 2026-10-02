# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import math
import os as os
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Annotated, Any, cast

import spmd_types as spmd
import torch
import torch.distributed.checkpoint.stateful
import tyro
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.loss import BaseLoss, ChunkedLossWrapper, IGNORE_INDEX
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import ensure_pp_loss_visible, MetricsProcessor
from torchtitan.components.optimizer import (
    configure_moe_routing_metrics,
    get_moe_load_balancing_metrics,
    OptimizersContainer,
)
from torchtitan.components.quantization.utils import has_quantization
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.components.validate import BaseValidator, Validator
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.config.override import apply_overrides, OverrideConfig
from torchtitan.distributed import full_dtensor, ParallelDims, utils as dist_utils
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
    MemoryBudgetAC,
    SelectiveAC,
)
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.profiler import Profiler


@dataclass(slots=True)
class _PendingStepMetrics:
    """Metrics of a logged step whose host readback has not happened yet."""

    step: int
    # [global_avg_loss, global_max_loss, grad_norm] as float32.
    float_metrics: torch.Tensor
    # [global_valid_tokens, global_ntokens_seen] as int64.
    int_metrics: torch.Tensor
    extra_metrics: dict[str, Any]
    # Device event recorded after the non-blocking host copies of the tensors
    # above, or None if they need no wait (or are still on device).
    ready_event: Any | None
    # Throughput counters of this step's log interval, detached from the
    # metrics processor at staging time so that steps run before the readback
    # are attributed to the next interval instead.
    ntokens_since_last_log: int
    data_loading_times: list[float]


class Trainer(torch.distributed.checkpoint.stateful.Stateful, Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """
        Default container for training configuration.
        """

        # NOTE: model_spec is suppressed from tyro CLI parsing and is always
        # set programmatically by the model registry before Trainer construction.
        model_spec: Annotated[ModelSpec | None, tyro.conf.Suppress] = None

        hf_assets_path: str = "./tests/assets/tokenizer"
        """
        Path to HF assets folder. This folder contains local copies of Hugging Face assets,
        including model weights in .safetensors format, the model.safetensor.index.json file
        (fqn to file mapping), the config.json file, generation_config.json, and tokenizer files.
        """

        dump_folder: str = "./outputs"
        """Folder to dump job outputs"""

        profiler: Profiler.Config = field(default_factory=Profiler.Config)
        metrics: MetricsProcessor.Config = field(
            default_factory=MetricsProcessor.Config
        )
        tokenizer: BaseTokenizer.Config = field(
            default_factory=HuggingFaceTokenizer.Config
        )
        dataloader: BaseDataLoader.Config = field(default_factory=BaseDataLoader.Config)
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        activation_checkpoint: ActivationCheckpointingConfig = field(
            default_factory=SelectiveAC.Config
        )
        compile: CompileConfig = field(default_factory=CompileConfig)
        comm: CommConfig = field(default_factory=CommConfig)
        validator: Validator.Config = field(default_factory=Validator.Config)
        debug: DebugConfig = field(default_factory=DebugConfig)
        override: OverrideConfig = field(default_factory=OverrideConfig)
        loss: BaseLoss.Config = field(default_factory=BaseLoss.Config)

        def __post_init__(self):
            if self.debug.batch_invariant:
                raise ValueError(
                    "Batch-invariant mode is not supported in pre-training."
                )

            if (
                self.parallelism.spmd_backend == "spmd_types"
                and self.debug.spmd_typechecking
                and self.parallelism.pipeline_parallel_degree > 1
            ):
                # TODO(sanketpurandare): Enable SPMD typechecking under PP.
                raise ValueError(
                    "SPMD typechecking is not supported with pipeline parallelism. "
                    "Validate the same config without PP "
                    "(--parallelism.pipeline_parallel_degree 1)."
                )

            if (
                self.parallelism.spmd_backend == "spmd_types"
                and self.debug.spmd_typechecking
                and isinstance(self.activation_checkpoint, SelectiveAC.Config)
                and self.model_spec is not None
                and any(self.model_spec.model.traverse(FlexAttention.Config))
            ):
                # TODO(pianpwk): Enable SAC with FlexAttention under SPMD typechecking.
                raise ValueError(
                    "Selective activation checkpointing (SAC) is not supported "
                    "with FlexAttention while SPMD typechecking is enabled. "
                    "Use full activation checkpointing, disable activation "
                    "checkpointing, or switch to a non-Flex attention backend."
                )

            if isinstance(self.activation_checkpoint, MemoryBudgetAC.Config) and not (
                self.compile.enable and "model" in self.compile.components
            ):
                raise ValueError(
                    "Memory budget activation checkpointing requires the model to be "
                    "compiled: set --compile.enable and include 'model' in "
                    "--compile.components."
                )

        def to_dict(self) -> dict[str, Any]:
            d = {}
            for f in dataclasses.fields(self):
                if f.name == "model_spec":
                    assert self.model_spec is not None
                    # ModelSpec contains callables that can't be serialized
                    d["model_spec"] = {
                        "name": self.model_spec.name,
                        "flavor": self.model_spec.flavor,
                    }
                else:
                    val = getattr(self, f.name)
                    if hasattr(val, "to_dict"):
                        d[f.name] = val.to_dict()
                    elif dataclasses.is_dataclass(val):
                        d[f.name] = asdict(val)
                    else:
                        d[f.name] = val
            return d

        def maybe_log(self) -> None:
            if self.debug.print_config:
                logger.info(
                    f"Running with configs: {json.dumps(self.to_dict(), indent=2, ensure_ascii=False)}"
                )

            if self.debug.save_config_file is not None:
                config_file = os.path.join(
                    self.dump_folder, self.debug.save_config_file
                )
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        os.makedirs(os.path.dirname(config_file), exist_ok=True)
                        with open(config_file, "w") as f:
                            json.dump(self.to_dict(), f, indent=2)
                    logger.info(f"Saved job configs to {config_file}")
                else:
                    logger.warning(
                        "Job configs logging is disabled due to torch.distributed not initialized."
                    )

    # core configs
    config: Config
    parallel_dims: ParallelDims

    # swappable training components
    tokenizer: BaseTokenizer
    dataloader: BaseDataLoader
    model_config: BaseModel.Config
    # TODO: we should make this list[BaseModel / Decoder] but this will affect many components.
    # will do this in a separate PR
    model_parts: list[torch.nn.Module]
    loss_fn: BaseLoss
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    validator: BaseValidator
    metrics_processor: MetricsProcessor
    checkpointer: CheckpointManager

    # runtime utilities
    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: dist_utils.SpmdContext
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    # additional training states
    step: int
    ntokens_seen: int
    # Metrics of the last logged step, awaiting host readback.
    _pending_metrics: _PendingStepMetrics | None = None

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: Config):
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        assert (
            config.model_spec is not None
        ), "model_spec must be set before creating Trainer"
        model_spec = config.model_spec

        device_type, device_module = utils.device_type, utils.device_module
        # pyrefly: ignore [read-only]
        self.device = utils.get_local_device()
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        self.parallel_dims = parallel_dims = self.init_distributed()

        # validate dense activation sequence length evenness
        seq_len_divisor = (
            parallel_dims.tp if config.parallelism.enable_sequence_parallel else 1
        ) * (2 * parallel_dims.cp if parallel_dims.cp > 1 else 1)
        if config.training.seq_len % seq_len_divisor != 0:
            raise ValueError(
                f"Training sequence length ({config.training.seq_len}) must be "
                f"divisible by {seq_len_divisor} for the configured "
                "sequence/context parallelism."
            )

        # TODO(pianpwk): Transitional until the local-SPMD and full-DTensor
        # backends share one runtime mesh/type mechanism.
        dist_utils.set_spmd_backend(config.parallelism.spmd_backend)

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0

        # verify batch sizes
        global_batch_size = config.training.global_batch_size
        local_global_batch_size = config.training.local_batch_size * batch_degree
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = local_global_batch_size
        assert global_batch_size > 0
        assert global_batch_size % local_global_batch_size == 0, (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({config.training.local_batch_size} * {batch_degree}) != 0)"
        )

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // local_global_batch_size
        assert self.gradient_accumulation_steps > 0

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.training.gc_freq, debug=config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )

        # build model (using meta init)
        model_config = model_spec.model
        # set the model args from training job configs
        model_config.update_from_config(
            config=config,
            global_batch_size=global_batch_size,
            local_global_batch_size=local_global_batch_size,
        )
        self.model_config = model_config

        # Apply overrides to the full config tree, before any component is
        # built. The model config is reached via ModelSpec.traverse. Model
        # overrides must run after update_from_config above (it sets sharding
        # config on the pre-override modules); all other components (optimizer,
        # loss, dataloader, …) are built later in __init__.
        if config.override.imports:
            apply_overrides(config.override, config)

        logger.info(f"Building {model_spec.name} {model_spec.flavor}")

        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        # Verify all submodules satisfy the Module protocol
        # TODO: move this to module validate().
        # This is current put here to verify module build and
        # converter, which should guanrantee Module protocol.
        # On the other hand, some parallelism wrappers don't
        # have this guanrantee, e.g., fully_shard.
        model.verify_module_protocol()

        # metrics logging
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            config_dict=config.to_dict(),
            has_quantization=has_quantization(model_config),
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_config.get_nparams_and_flops(model, config.training.seq_len)

        logger.info(
            f"{color.blue}Model {model_spec.name} {model_spec.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        buffer_device: torch.device | None
        if config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = torch.device(device_type)
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = config.loss.build(
            compile_config=config.compile,
        )

        # apply parallelisms and initialization
        with sl.log_trace_span("model_parallelism_init"):
            if parallel_dims.pp_enabled:
                if not model_spec.pipelining_fn:
                    raise RuntimeError(
                        f"Pipeline Parallel is enabled but {model_spec.name} "
                        f"does not support pipelining"
                    )

                # apply both Pipeline Parallel and SPMD-style scaling techniques
                (
                    self.pp_schedule,
                    self.model_parts,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                ) = model_spec.pipelining_fn(
                    model,
                    parallel_dims=parallel_dims,
                    training=config.training,
                    parallelism=config.parallelism,
                    compile_config=config.compile,
                    ac_config=config.activation_checkpoint,
                    dump_folder=config.dump_folder,
                    device=self.device,
                    model_config=model_config,
                    parallelize_fn=model_spec.parallelize_fn,
                    loss_fn=self.loss_fn,
                )
                # when PP is enabled, `model` obj is no longer used after this point,
                # model_parts is used instead
                del model

                for m in self.model_parts:
                    m.to_empty(device=init_device)
                    with torch.no_grad():
                        # TODO: Change this back to init_weights once
                        # autoparallel contains the wrap_init_states
                        cast(BaseModel, m).init_weights(buffer_device=buffer_device)
                    m.train()

                # confirm that user will be able to view loss metrics on the console
                ensure_pp_loss_visible(
                    parallel_dims=parallel_dims,
                    pp_schedule=config.parallelism.pipeline_parallel_schedule,
                    color=color,
                )
            else:
                if not config.checkpoint.create_seed_checkpoint:
                    # Skip parallelize_fn for seed checkpoints — nothing from
                    # it is needed (AC, compile, nD parallelism, mixed precision, etc.).
                    model = model_spec.parallelize_fn(
                        model,
                        parallel_dims=parallel_dims,
                        training=config.training,
                        parallelism=config.parallelism,
                        compile_config=config.compile,
                        ac_config=config.activation_checkpoint,
                        dump_folder=config.dump_folder,
                    )

                model.to_empty(device=init_device)
                with torch.no_grad():
                    # TODO: Change this back to init_weights once
                    # autoparallel contains the wrap_init_states
                    cast(BaseModel, model).init_weights(buffer_device=buffer_device)
                model.train()

                self.model_parts = [model]

        self._collect_moe_routing_metrics = configure_moe_routing_metrics(
            self.model_parts, config.metrics.enable_moe_routing
        )

        # Set lm_head reference for ChunkedLossWrapper after model construction.
        # Non-PP: single model part always has lm_head.
        # PP: only the last stage has lm_head; non-last stages skip this.
        if isinstance(self.loss_fn, ChunkedLossWrapper):
            if parallel_dims.pp_enabled:
                if self.pp_has_last_stage:
                    lm_head = self.model_parts[-1].lm_head
                    assert (
                        lm_head is not None
                    ), "Last PP stage must have lm_head for ChunkedLossWrapper"
                    self.loss_fn.set_lm_head(
                        lm_head  # pyrefly: ignore[bad-argument-type]
                    )
                    self.model_parts[
                        -1
                    ]._skip_lm_head = True  # pyrefly: ignore[bad-argument-type]
            else:
                assert len(self.model_parts) == 1
                lm_head = self.model_parts[0].lm_head
                assert (
                    lm_head is not None
                ), "Model must have lm_head for ChunkedLossWrapper"
                self.loss_fn.set_lm_head(lm_head)  # pyrefly: ignore[bad-argument-type]
                self.model_parts[
                    0
                ]._skip_lm_head = True  # pyrefly: ignore[bad-argument-type]

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, parallel_dims
            )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0

        # build tokenizer
        self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

        # build dataloader
        self.dataloader = config.dataloader.build(
            dp_world_size=batch_degree,
            dp_rank=batch_rank,
            tokenizer=self.tokenizer,
            seq_len=config.training.seq_len,
            local_batch_size=config.training.local_batch_size,
            snapshot_every_n_steps=(
                config.checkpoint.interval * self.gradient_accumulation_steps
                if config.checkpoint.enable
                else None
            ),
        )

        # build checkpointer
        self.checkpointer = config.checkpoint.build(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=(
                model_spec.state_dict_adapter(model_config, config.hf_assets_path)
                if model_spec.state_dict_adapter
                else None
            ),
            base_folder=config.dump_folder,
        )

        self.train_context = dist_utils.get_spmd_context(
            parallel_dims=parallel_dims,
            spmd_typechecking=(
                config.parallelism.spmd_backend == "spmd_types"
                and config.debug.spmd_typechecking
            ),
        )

        # Build validator if validation is configured
        if config.validator.enable:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                parallelism=config.parallelism,
                dp_world_size=batch_degree,
                dp_rank=batch_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.seq_len,
                local_batch_size=config.training.local_batch_size,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {config.training.seq_len}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.lr_scheduler.warmup_steps})"
        )

    @sl.log_trace_span("torch_distributed_init")
    def init_distributed(self) -> ParallelDims:
        config = self.config
        world_size = dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=config.dump_folder,
        )

        return ParallelDims.from_config(config.parallelism, world_size)

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator.

        Note: Tensors are yielded on CPU. The caller is responsible for moving
        them to GPU when needed. This allows for more efficient memory usage
        when doing gradient accumulation.
        """
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            ntokens_batch = labels.numel()
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Tensors stay on CPU; moved to GPU per-microbatch during training
            yield input_dict, labels

    @sl.log_trace_span("post_dataloading_process")
    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Post-processing hook after data loading and before model forward pass.

        This method processes the raw data from the dataloader and prepares it for
        the model's forward pass. It separates the main input tensor from auxiliary
        inputs and constructs additional keyword arguments (e.g., attention masks).

        This method can be overridden in subclasses to customize data processing
        for different training strategies (e.g., converting tensors to DTensors,
        applying custom transformations, etc.).

        Args:
            input_dict: Dictionary containing tensors from the dataloader. Must
                contain an "input" key with the main input tensor. May contain
                additional keys for auxiliary inputs (e.g., position ids).
            labels: Target labels for the batch.

        Returns:
            A tuple of (inputs, labels, extra_kwargs) where:
                - inputs: Main input tensor extracted from input_dict["input"].
                - labels: Target labels (unchanged from input parameter).
                - extra_kwargs: Additional keyword arguments for the model forward
                    (e.g. positions, attention_masks), forwarded to every
                    pipeline-parallel stage.
        """
        inputs = input_dict["input"]
        # Everything else becomes a model-forward kwarg, forwarded to all PP
        # stages by the schedule. positions is read here so we can build masks.
        extra_kwargs: dict[str, Any] = {
            k: v for k, v in input_dict.items() if k != "input"
        }

        positions = extra_kwargs.get("positions", None)

        # positions and attention_masks are optional (Decoder.forward defaults
        # both to None). Build attention masks only for the masked backends
        # (Flex/Varlen), which is where get_attention_masks is defined. A
        # maskless backend (e.g. the SDPA config used by the graph_trainer
        # tests) still receives positions for RoPE but no masks — it relies on
        # is_causal instead.
        if isinstance(self.model_config, Decoder.Config) and positions is not None:
            inner_attention = getattr(
                self.model_config.first_attention, "inner_attention", None
            )
            if isinstance(
                inner_attention, (FlexAttention.Config, VarlenAttention.Config)
            ):
                model = cast(Decoder, self.model_parts[0])
                extra_kwargs["attention_masks"] = model.get_attention_masks(
                    positions=positions,
                )

        if self.parallel_dims.cp_enabled:
            inputs, labels, extra_kwargs = prepare_context_parallel_input(
                inputs,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.device,
                self.config.parallelism.context_parallel_load_balancer,
            )

        # Accumulate after CP sharding so labels.numel() reflects the actual
        # unique tokens this rank processes (not the full pre-split sequence).
        self.ntokens_seen += labels.numel()

        if self.config.parallelism.spmd_backend == "full_dtensor":
            inputs, labels, extra_kwargs = full_dtensor.parallelize_inputs(
                self.parallel_dims, inputs, labels, extra_kwargs
            )
        elif self.config.parallelism.spmd_backend == "spmd_types":
            inputs, labels, extra_kwargs = annotate_input_spmd_types(
                self.parallel_dims,
                inputs,
                labels,
                extra_kwargs,
            )

        return inputs, labels, extra_kwargs

    @sl.log_trace_span("fwd_bwd")
    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: float | torch.Tensor,
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs, labels, extra_kwargs = self.post_dataloading_process(input_dict, labels)

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            loss_kwargs = {"global_valid_tokens": global_valid_tokens}
            with self.train_context():
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs,
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        loss_kwargs=loss_kwargs,
                        return_outputs=False,
                    )
                else:
                    self.pp_schedule.step(
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        loss_kwargs=loss_kwargs,
                        return_outputs=False,
                    )

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            if self.pp_has_last_stage:
                assert losses is not None
                # All loss classes scale by global_valid_tokens internally
                loss = torch.sum(torch.stack(losses)).to(self.device)
            else:
                loss = torch.tensor([-1.0], device=self.device)
        else:
            # Non-PP forward / backward
            assert len(model_parts) == 1
            with self.train_context():
                pred = model_parts[0](inputs, **extra_kwargs)
                loss, _ = self.loss_fn(pred, labels, global_valid_tokens)
                del pred
                with spmd.no_typecheck():
                    # this propagates types through BWD, causing unnecessary conflicts
                    # between torch_function and internals (e.g. AC). FWD is sufficient.
                    loss.backward()

        # The returned loss here is local SUM loss / global_valid_tokens
        return loss

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        # Save per-optimizer-group learning rates for logging
        lr_metrics = self.lr_schedulers.get_metrics()

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        # Collect all microbatches on CPU and count total valid tokens
        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _microbatch in range(self.gradient_accumulation_steps):
            with sl.log_trace_span("fetching_batch"):
                input_dict, labels = next(data_iterator)
                local_valid_tokens += (labels != IGNORE_INDEX).sum()
                microbatches.append((input_dict, labels))
        sl.log_trace_scalar({"local_valid_tokens": int(local_valid_tokens)})

        # All-reduce to get global token count across DP ranks.
        # Keep the result on device (no `.item()`) so the host does not block on
        # a cross-slice all-reduce before dispatching forward/backward. The
        # reduction is done in int64 for exactness (the global count can exceed
        # float32's 2**24 integer range) and cast to float32 afterwards.
        local_valid_tokens_dev = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens_i64 = dist_utils.dist_sum_tensor(
                local_valid_tokens_dev, batch_mesh
            )
        else:
            global_valid_tokens_i64 = local_valid_tokens_dev
        global_valid_tokens_t = global_valid_tokens_i64.to(torch.float32)

        # Precompiled aot_fx_trace artifacts were traced with a Python float, so
        # keep passing a float in that case to match the artifact signature.
        global_valid_tokens: float | torch.Tensor
        if getattr(self.config.compile, "precompile_artifact_dir", None):
            global_valid_tokens = float(global_valid_tokens_t.item())
        else:
            global_valid_tokens = global_valid_tokens_t

        # Process each microbatch: move to GPU, forward/backward, then free
        accumulated_losses = []
        for input_dict, labels in microbatches:
            # Move tensors to GPU
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            labels = labels.to(self.device)

            loss = self.forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )
            accumulated_losses.append(loss.detach())

        with sl.log_trace_span("optim"):
            grad_norm = self._clip_grad_norm()
            self.checkpointer.maybe_wait_for_staging()
            self.optimizers.step()
            self.lr_schedulers.step()

        # Reduce the data collected over gradient accumulation steps.
        loss = torch.sum(torch.stack(accumulated_losses))

        # With `metrics.defer_readback`, the previous logged step's metrics are
        # read back only now, after this step's work has been dispatched. The
        # host then waits on work that is a step old instead of draining the
        # device queue, so it can keep running ahead of the device.
        self._flush_pending_metrics()

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        with sl.log_trace_span("collect_dist_metrics"):
            # Enqueue all metric collectives before any host sync so they run
            # back-to-back on device, then pay for a single D2H transfer.
            if parallel_dims.dp_cp_enabled:
                loss = loss.detach()
                loss_mesh = parallel_dims.get_optional_mesh("loss")

                # For global_avg_loss, we want the average loss across all ranks:
                # loss = local_loss_sum / global_valid_tokens
                # global_avg_loss = sum(local_loss_sum) / global_valid_tokens
                #                 = sum(loss)
                #
                # For global_max_loss, we want the max of local average losses across ranks:
                # local_avg_loss = local_loss_sum / local_valid_tokens
                #                = (loss * global_valid_tokens) / local_valid_tokens
                # global_max_loss = max(local_avg_loss)
                local_avg_loss = (
                    loss * global_valid_tokens_t / local_valid_tokens_dev
                )
                global_avg_loss_t = dist_utils.dist_sum_tensor(loss, loss_mesh)
                global_max_loss_t = dist_utils.dist_max_tensor(
                    local_avg_loss, loss_mesh
                )
                global_ntokens_seen_t = dist_utils.dist_sum_tensor(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                )
            else:
                loss = loss.detach()
                global_avg_loss_t = global_max_loss_t = loss
                global_ntokens_seen_t = torch.tensor(
                    self.ntokens_seen, dtype=torch.int64, device=self.device
                )

            def _to_plain_scalar(t: torch.Tensor) -> torch.Tensor:
                # Matches what `.item()` does on a DTensor (materialize the
                # full value) so it can be stacked with plain tensors.
                if isinstance(t, DTensor):
                    t = t.full_tensor()
                return t.reshape(())

            # Coalesced device-to-host transfers: one for float metrics and one
            # for int64 counters (kept in int64 so large counts stay exact).
            float_metrics = torch.stack(
                [
                    _to_plain_scalar(global_avg_loss_t).float(),
                    _to_plain_scalar(global_max_loss_t).float(),
                    _to_plain_scalar(grad_norm.detach()).float(),
                ]
            )
            int_metrics = torch.stack(
                [
                    _to_plain_scalar(global_valid_tokens_i64).long(),
                    _to_plain_scalar(global_ntokens_seen_t).long(),
                ]
            )

        # The MoE routing metrics read tensors stashed on the modules by the
        # latest forward, so they must be collected before the next forward
        # overwrites them. Note that, when enabled, this does its own blocking
        # device-to-host copy of the per-expert counts.
        extra_metrics: dict[str, Any] = dict(lr_metrics)
        extra_metrics.update(get_moe_load_balancing_metrics(self.model_parts))

        defer = self.metrics_processor.config.defer_readback
        ready_event = None
        if defer:
            # Start the copies now; they complete once this step's metric
            # collectives do, without making the host wait for them.
            (float_metrics, int_metrics), ready_event = self._copy_to_host_async(
                [float_metrics, int_metrics]
            )
        # Detach this interval's throughput counters so that steps which run
        # before a deferred readback count toward the next log interval.
        metrics_processor = self.metrics_processor
        ntokens_since_last_log = metrics_processor.ntokens_since_last_log
        data_loading_times = list(metrics_processor.data_loading_times)
        metrics_processor.ntokens_since_last_log = 0
        metrics_processor.data_loading_times.clear()
        self._pending_metrics = _PendingStepMetrics(
            step=self.step,
            float_metrics=float_metrics,
            int_metrics=int_metrics,
            extra_metrics=extra_metrics,
            ready_event=ready_event,
            ntokens_since_last_log=ntokens_since_last_log,
            data_loading_times=data_loading_times,
        )
        if not defer:
            self._flush_pending_metrics()

    def _copy_to_host_async(
        self, tensors: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], Any]:
        """Starts non-blocking device-to-host copies of `tensors`.

        Returns the host tensors and a device event recorded after the copies.
        The host tensors must not be read before `event.synchronize()`. The
        event is None when the tensors already live on the host.
        """
        if self.device.type == "cpu":
            return tensors, None
        host_tensors = [t.to("cpu", non_blocking=True) for t in tensors]
        event = torch.get_device_module(self.device.type).Event()
        event.record()
        return host_tensors, event

    def _flush_pending_metrics(self) -> None:
        """Reads back and logs the metrics staged by the last logged step."""
        pending = self._pending_metrics
        if pending is None:
            return
        self._pending_metrics = None

        with sl.log_trace_span("metrics_readback"):
            if pending.ready_event is not None:
                pending.ready_event.synchronize()
            global_avg_loss, global_max_loss, grad_norm_val = (
                pending.float_metrics.cpu().tolist()
            )
            global_valid_tokens_val, global_ntokens_seen = (
                pending.int_metrics.cpu().tolist()
            )
        sl.log_trace_scalar({"global_valid_tokens": int(global_valid_tokens_val)})

        # Crash on invalid loss. global_avg_loss is a SUM reduction, so a infinite
        # loss on any rank propagates here. This reuses the D2H copy already done
        # for logging, so it adds no extra sync. With `metrics.defer_readback`
        # this fires one step after the step that produced the loss.
        # TODO: make this step work even logging is off.
        if not math.isfinite(global_avg_loss):
            raise RuntimeError(
                f"Loss is not finite (global_avg_loss={global_avg_loss}) at "
                f"step {pending.step}. Stopping training."
            )

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            **pending.extra_metrics,
        }
        # `log()` computes throughput from, and then resets, the processor's
        # counters. Log with the pending step's own counters and keep whatever
        # later steps have accumulated since for the next interval.
        metrics_processor = self.metrics_processor
        later_ntokens = metrics_processor.ntokens_since_last_log
        later_data_loading_times = list(metrics_processor.data_loading_times)
        metrics_processor.ntokens_since_last_log = pending.ntokens_since_last_log
        metrics_processor.data_loading_times.clear()
        metrics_processor.data_loading_times.extend(pending.data_loading_times)
        metrics_processor.log(
            pending.step,
            global_avg_loss,
            global_max_loss,
            grad_norm_val,
            extra_metrics=extra_metrics,
        )
        metrics_processor.ntokens_since_last_log = later_ntokens
        metrics_processor.data_loading_times.clear()
        metrics_processor.data_loading_times.extend(later_data_loading_times)

    def _clip_grad_norm(self) -> torch.Tensor:
        return dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )

    @record
    def train(self):
        config = self.config

        sl.log_trace_instant("training_start")

        self.checkpointer.load(step=config.checkpoint.load_step)

        # Capture loaded step for relative_step calculation.
        # After checkpoint load: self.step = restored step (e.g. 100), or 0 if fresh.
        loaded_step = self.step

        logger.info(f"Training starts at step {self.step + 1}")

        with config.profiler.build(
            global_step=self.step,
            base_folder=config.dump_folder,
        ) as profiler:
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                sl.set_step(self.step, relative_step=self.step - loaded_step)

                with sl.log_trace_span("step"):
                    self.gc_handler.run(self.step)

                    try:
                        self.train_step(data_iterator)
                    except DataloaderExhaustedError:
                        logger.warning("Ran out of data; last step was canceled.")
                        break

                    last_step = self.step == config.training.steps
                    # Deferred metrics carry the non-finite loss check; run it
                    # before persisting state that may be corrupted.
                    if self.checkpointer._should_save(self.step, last_step):
                        self._flush_pending_metrics()
                    self.checkpointer.save(self.step, last_step=last_step)

                    # Run validation if validator is available
                    if self.config.validator.enable and self.validator.should_validate(
                        self.step
                    ):
                        self.validator.validate(self.model_parts, self.step)

                    # signal the profiler that the next profiling step has started
                    profiler.step()

                    # Reduce timeout after the first train step of THIS process
                    # (assuming lazy init and compilation are finished). Use the
                    # relative step so this fires on resumed runs too.
                    if (
                        self.step - loaded_step == 1
                        and self.device.type != "tpu"
                        and self.device.type != "cpu"
                    ):
                        # TODO(alekseyv): TPU doesn't support set_pg_timeouts yet.
                        dist_utils.set_pg_timeouts(
                            timeout=timedelta(
                                seconds=config.comm.train_timeout_seconds
                            ),
                            parallel_dims=self.parallel_dims,
                        )

            # Log the last deferred step, if any.
            self._flush_pending_metrics()

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def should_continue_training(self) -> bool:
        return self.step < self.config.training.steps

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]

    def close(self) -> None:
        if hasattr(self, "checkpointer") and self.checkpointer:
            self.checkpointer.close()
        if hasattr(self, "metrics_processor") and self.metrics_processor:
            self.metrics_processor.close()
