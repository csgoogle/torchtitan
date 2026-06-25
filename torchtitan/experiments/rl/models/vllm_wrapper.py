# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Base wrapper for TorchTitan models to work with vLLM V1 engine.

This module provides TorchTitanVLLMModel: Core model class that adapts
TorchTitan models for vLLM.
"""

import dataclasses

import torch
import torch._dynamo
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)

from torchtitan.config import ParallelismConfig
from torchtitan.config.configs import CompileConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.tools.utils import get_device_type
from torchtitan.experiments.rl.models.attention import VLLMAttentionWrapper
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.module import Module
# vllm.compilation.codegen depends on GPU/CUDA-specific Triton compilation tools 
# which are completely absent on TPU or CPU-only vLLM container environments.
# Wrapping the import in a try-except block guarantees multi-architecture portability.
try:
    from vllm.compilation import codegen as _codegen
except ImportError:
    _codegen = None
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import torch_utils as _torch_utils


logger = init_logger(__name__)

# NOTE: Monkeypatch vLLM's weak_ref_tensor to handle DTensor
# This is because piecewise CUDA-graph capture calls weak_ref_tensor()
# on every subgraphoutput (see vllm/compilation/cuda_graph.py).
# When TP is active some of those outputs are DTensors which fail with
# ("The specified pointer resides on host memory").  to_local
# converts the DTensor to a plain tensor. which succeeds with this
# cudagraph implementation.
_original_weak_ref_tensor = _torch_utils.weak_ref_tensor


def _dtensor_safe_weak_ref_tensor(tensor):
    if isinstance(tensor, DTensor):
        tensor = tensor._local_tensor
    return _original_weak_ref_tensor(tensor)


_torch_utils.weak_ref_tensor = _dtensor_safe_weak_ref_tensor


# We bypass GPU-specific codegen monkeypatching in two key Ray scenarios:
# 1. On Ray TPU Workers: They run a TPU/CPU-focused build of vLLM where Triton 
#    and GPU codegen tools are absent, causing the _codegen import to fall back to None.
# 2. On Ray CPU Head Node / CPU Actors: Non-accelerator VM environments do not 
#    load GPU compilation dependencies.
if get_device_type() == "tpu" or _codegen is None:
    pass
else:
    # NOTE: Monkeypatch vLLM's _node_ref to handle DTensor placement types
    # whose repr() uses unqualified class names not available in the generated
    # code's exec namespace (which only has `import torch`).
    _original_node_ref = _codegen._node_ref
    
    
    # TODO: Followup with core vLLM fix
    # https://github.com/pytorch/torchtitan/issues/3067
    def _patched_node_ref(arg):
        try:
            from torch.distributed.tensor.placement_types import Partial, Placement
    
            if isinstance(arg, Placement):
                cls = type(arg)
                # Partial.__repr__ leaves reduce_op unquoted (e.g. "Partial(sum)")
                # which would resolve to the builtin sum, not the string "sum".
                if isinstance(arg, Partial):
                    return f"{cls.__module__}.{cls.__name__}({arg.reduce_op!r})"
                return f"{cls.__module__}.{repr(arg)}"
        except ImportError:
            pass
        return _original_node_ref(arg)
    
    
    _codegen._node_ref = _patched_node_ref


def create_torchtitan_config_from_vllm_config(
    vllm_config: VllmConfig,
) -> tuple[ParallelDims, ParallelismConfig]:
    """
    Create ParallelDims and ParallelismConfig from vLLM configuration.

    Maps vLLM parallelism settings to TorchTitan's config objects so that
    TorchTitan's parallelize functions can be called with the correct kwargs.

    This is needed because vLLM doesn't separate model creation and parallelism
    application — it requires parallelization inside the model constructor
    (TorchTitanVLLMModelWrapper.__init__).

    Args:
        vllm_config: vLLM configuration object

    Returns:
        Tuple of (ParallelDims, ParallelismConfig) mapped from vLLM config

    Note:
        vLLM doesn't use FSDP sharding (dp_shard=1) or expert parallelism (ep=1)
        in inference. These are set to default values.
    """
    if get_device_type() == "tpu":
        if vllm_config.parallel_config.tensor_parallel_size > 1:
            world_size = dist.get_world_size()
        else:
            # vLLM engine is forced to run on a single device per rank for TP = 1
            logger.info(
                "TPU hack for vllm with tp=1"
                "We set world_size = 1, so"
                "vLLM engine is forced to run on a single device per rank."
            )
            world_size = 1
    else:
        world_size = dist.get_world_size()
    parallel_config = vllm_config.parallel_config

    parallel_dims = ParallelDims(
        dp_replicate=parallel_config.data_parallel_size,
        dp_shard=1,
        cp=parallel_config.decode_context_parallel_size,
        tp=parallel_config.tensor_parallel_size,
        pp=parallel_config.pipeline_parallel_size,
        ep=1,
        world_size=world_size,
    )

    parallelism = ParallelismConfig(
        data_parallel_replicate_degree=parallel_config.data_parallel_size,
        data_parallel_shard_degree=1,
        context_parallel_degree=parallel_config.decode_context_parallel_size,
        tensor_parallel_degree=parallel_config.tensor_parallel_size,
        pipeline_parallel_degree=parallel_config.pipeline_parallel_size,
        expert_parallel_degree=1,
        disable_loss_parallel=True,  # vLLM handles sampling and expects plain tensor logits.
        enable_sequence_parallel=False,
    )

    logger.info(
        f"Created TorchTitan config from vLLM: "
        f"DP={parallel_dims.dp_replicate}, TP={parallel_dims.tp}, "
        f"CP={parallel_dims.cp}, PP={parallel_dims.pp}"
    )

    return parallel_dims, parallelism


# TPU hack for removing the decorator for TorchTitanVLLMModelWrapper class below
def apply_if(condition, decorator):
    """Applies a decorator only if the condition is True."""
    def meta_decorator(target):
        if condition:
            return decorator(target)
        return target
    return meta_decorator


# @support_torch_compile(
#     dynamic_arg_dims={
#         "input_ids": 0,
#         "positions": 0,
#     }
# )
@apply_if(
    condition=get_device_type() != "tpu",  # remove decorator if on TPU
    decorator=support_torch_compile(       # The original decorator to apply if True
        dynamic_arg_dims={
            "input_ids": 0,
            "positions": 0,
        }
    )
)
class TorchTitanVLLMModelWrapper(Module):
    """
    Generic vLLM-compatible model wrapper for TorchTitan models. Implemented
    required interface required by vLLM Engine.
    Doc: https://docs.vllm.ai/en/latest/contributing/model/basic/
    Reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py

    The wrapper handles:
    - Direct usage of TorchTitan model args (no HF config mapping needed)
    - Attention replacement with vLLM paged attention
    - Parallelism setup and DTensor conversion between torchtitan and vLLM
    - Weight loading from HF checkpoints
    - vLLM forward/compute_logits interface
    """

    is_text_generation_model = True  # Required for vLLM runner validation
    supports_pp = False  # Pipeline parallelism not supported yet
    supports_multimodal = False

    def __init__(
        self,
        *,
        model_spec: ModelSpec,
        vllm_config: VllmConfig,
        prefix: str = "",
        compile_config: CompileConfig,
    ):
        super().__init__()

        assert vllm_config is not None, "vllm_config is required"

        # Store components from model_spec
        self.state_dict_adapter = model_spec.state_dict_adapter
        self.parallelize_fn = model_spec.parallelize_fn

        # Replace inner_attention with VLLMAttentionWrapper in config
        model_config = model_spec.model
        attn_config = model_config.layers[0].attention
        n_heads = attn_config.n_heads
        n_kv_heads = attn_config.n_kv_heads or n_heads
        head_dim = (
            attn_config.head_dim
            if attn_config.head_dim is not None
            else model_config.dim // n_heads
        )
        vllm_backend = VLLMAttentionWrapper.Config(
            hidden_size=model_config.dim,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        new_layers = [
            dataclasses.replace(
                layer_cfg,
                attention=dataclasses.replace(
                    layer_cfg.attention, inner_attention=vllm_backend
                ),
            )
            for layer_cfg in model_config.layers
        ]
        self.config = dataclasses.replace(model_config, layers=new_layers)
        logger.debug(f"Creating model with config: {self.config.to_dict()}")

        # Create ParallelDims and configs from vLLM config at runtime.
        self.parallel_dims, parallelism = create_torchtitan_config_from_vllm_config(
            vllm_config
        )
        
        if get_device_type() == "tpu" and self.parallel_dims.tp_enabled:
            # Initialize distributed environment and mesh if not already done
            import torch.distributed as dist
            if not dist.is_initialized():
                from torchtitan.experiments.tpu import utils as tpu_utils
                import os
                # This should have been done by vLLM TPUWorker but just in case
                dist.init_process_group(
                    backend="tpu_dist",
                    rank=int(os.environ.get("RANK", 0)),
                    world_size=int(os.environ.get("WORLD_SIZE", 1))
                )
            
            # For TP>1, the mesh must be built to allow get_mesh("tp") calls
            self.parallel_dims.build_mesh()

        # Fill sharding configs on the config BEFORE build so every sub-module
        # is constructed with its ShardingConfig attached (required by the
        # declarative model.parallelize() API).
        # TODO: Refactor update_from_config to accept ParallelismConfig
        # directly instead of requiring a trainer_config wrapper.
        from types import SimpleNamespace

        from torchtitan.config import DebugConfig, TrainingConfig

        self.config.update_from_config(
            trainer_config=SimpleNamespace(
                training=TrainingConfig(),
                parallelism=parallelism,
                debug=DebugConfig(),
            )
        )

        if get_device_type() == "tpu" and self.parallel_dims.tp_enabled:
            # For TPU vllm with TP, we need to init model in meta and then apply sharding
            # otherwise it will lead to hang at Pre-built RPA kernels
            with torch.device("meta"):
                self.model = self.config.build()
        else:
            # TODO: Check if it's possible to apply meta init
            self.model = self.config.build()

        # RoPE config from model for cache extension
        self.rope_config = self.config.rope

        # Apply parallelism using the model's own parallelize function.
        # AC is disabled; skip_dp=True skips FSDP. compile_config is passed
        # through so apply_compile runs per-layer after TP.
        from torchtitan.config import ActivationCheckpointConfig

        # With TP, collectives may return AsyncCollectiveTensor (overlap
        # path) or plain Tensor (sync path) depending on timing.  Dynamo
        # specializes on tensor type, so each switch triggers a
        # recompile.  Because of this, the default recompile_limit (8) is
        # too low; exceeding it fails under
        # fullgraph=True so set to 10 for now
        if compile_config.enable:
            torch._dynamo.config.recompile_limit = 100

        self.model = self.parallelize_fn(
            model=self.model,
            parallel_dims=self.parallel_dims,
            training=TrainingConfig(),
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ActivationCheckpointConfig(mode="none"),
            dump_folder="",
            skip_dp=True,
        )

        if get_device_type() == "tpu" and self.parallel_dims.tp_enabled:
            # put the model into device if it was at meta
            device = vllm_config.device_config.device            
            self.model = self.model.to_empty(device=device)
            # Skip init_states to prevent thousands of eager TPU compilations!
            # We will copy the weights from the trainer anyway.
            # But we MUST initialize the RoPE cache, otherwise it will output garbage!
            extended_rope = self.rope_config.build()
            self.model.freqs_cis = extended_rope.cache.to(device=device, dtype=self.model.tok_embeddings.weight.dtype)

        # Pre-extend RoPE cache to cover vLLM's max model length (profiling
        # may use up to 2x max_seq_len, so use max_model_len which already
        # accounts for this). This avoids data-dependent control flow in
        # forward() which is incompatible with torch.compile.
        max_model_len = vllm_config.model_config.max_model_len
        if self.model.freqs_cis.shape[0] < max_model_len:
            self.model.freqs_cis = self._extend_rope_cache(
                self.model.freqs_cis, max_model_len
            )

        # Initial load model weights from HuggingFace checkpoint path
        if get_device_type() == "tpu":
            # On TPU with collocated GRPO, the vLLM engine is instantiated with
            # `load_format="dummy"` and a fake model path to avoid duplicate memory
            # allocation and disk I/O. Therefore, we must skip loading HF weights here,
            # as the dummy path contains no weights. Instead, the real weights are synced 
            # directly from the TorchTitan FSDP policy model via `VLLMSampler.sync_weights()`.
            pass
        else:
            self._initial_load_weights(checkpoint_path=vllm_config.model_config.model)

    def _extend_rope_cache(
        self, rope_cache: torch.Tensor, required_len: int
    ) -> torch.Tensor:
        """
        Build an extended RoPE cache of at least ``required_len`` positions.

        Args:
            rope_cache: Current RoPE cache tensor
            required_len: Minimum number of positions the cache must cover

        Returns:
            Extended RoPE cache tensor
        """
        # Handle DTensor case
        is_dtensor = isinstance(rope_cache, DTensor)
        if is_dtensor:
            device_mesh = rope_cache.device_mesh
            local_rope_cache = rope_cache.to_local()
            device = local_rope_cache.device
            dtype = local_rope_cache.dtype
        else:
            device = rope_cache.device
            dtype = rope_cache.dtype

        # Build a new RoPE module with extended max_seq_len
        extended_rope_config = dataclasses.replace(
            self.rope_config, max_seq_len=required_len
        )
        extended_rope = extended_rope_config.build()
        extended_cache = extended_rope.cache.to(device=device, dtype=dtype)

        # Convert back to DTensor if needed
        if is_dtensor:
            rope_cache = DTensor.from_local(
                extended_cache,
                device_mesh=device_mesh,
                placements=[Replicate()],
            )
        else:
            rope_cache = extended_cache

        return rope_cache

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """vLLM required API.
        Convert input token IDs to embeddings."""
        return self.model.tok_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """vLLM required API.
        Convert input token IDs to embeddings (deprecated vLLM interface)."""
        return self.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        vLLM required API.
        Forward pass with vLLM interface.

        Args:
            input_ids: Token IDs [total_tokens] (1D varlen format)
            positions: Position indices [total_tokens] (1D varlen format)
            inputs_embeds: Pre-computed embeddings (optional)
            **kwargs: Additional vLLM kwargs

        Returns:
            hidden_states: Final hidden states [total_tokens, hidden_size]
        """
        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds not yet supported")

        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Convert vLLM interface to TorchTitan interface
        # vLLM: [total_tokens] → TorchTitan: [batch_size, seq_len]
        tokens_2d = input_ids.unsqueeze(0)
        
        # When TP is enabled, model expects DTensors
        if get_device_type() == "tpu" and self.parallel_dims.tp_enabled:
            tokens_2d = DTensor.from_local(
                tokens_2d,
                device_mesh=self.parallel_dims.get_mesh("tp"),
                placements=[Replicate()],
            )

        # Get embeddings
        h = self.model.tok_embeddings(tokens_2d)

        rope_cache = self.model.freqs_cis
        positions = positions.unsqueeze(0)
        
        if get_device_type() == "tpu" and self.parallel_dims.tp_enabled:
            positions = DTensor.from_local(
                positions,
                device_mesh=self.parallel_dims.get_mesh("tp"),
                placements=[Replicate()],
            )

        # Pass through transformer layers
        for layer in self.model.layers.values():
            h = layer(h, rope_cache, attention_masks=None, positions=positions)
        h = self.model.norm(h)
        # When parallelism is applied, get full tensor before return to vLLM Engine
        if isinstance(h, DTensor):
            h = h.full_tensor()

        # Convert to vLLM format: [total_tokens, hidden_size]
        if h.dim() == 3:
            hidden_size = h.size(-1)
            h = h.view(-1, hidden_size)
        return h

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        """vLLM required API.
        Compute logits from hidden states."""

        # When TP is applied, we return the full tensor (plain tensor) to vLLM engine
        # at the end of TorchTitanVLLMModelWrapper.forward().
        # We need to wrap the input from vLLM engine back to DTensor with Replicate() placement.
        if self.parallel_dims.tp_enabled:
            hidden_states = DTensor.from_local(
                hidden_states,
                device_mesh=self.parallel_dims.get_mesh("tp"),
                placements=[
                    Replicate(),
                ],
            )

        logits = self.model.lm_head(hidden_states)

        # Full DTensor path returns logits as DTensor; vLLM expects plain tensors.
        if isinstance(logits, DTensor):
            logits = logits.to_local()

        return logits

    def load_weights_from_state_dict(self, state_dict):
        """
        Load model weights from a state dict.

        Expects DTensor-wrapped tensors matching the model's placements.
        The caller is responsible for reconstructing DTensors from plain
        local tensors before calling this method.
        """
        set_model_state_dict(
            model=self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )

        return state_dict.keys()

    def _initial_load_weights(self, checkpoint_path):
        """
        Helper function to load torchtitan model weights from HF checkpoint when initialize this model.

        Args:
            checkpoint_path: Path to the HuggingFace checkpoint directory
        """
        # Create adapter instance
        adapter = self.state_dict_adapter(
            model_config=self.config,
            hf_assets_path=None,
        )

        # Get HF storage reader from adapter
        storage_reader = adapter.get_hf_storage_reader(checkpoint_path)

        # Load HF state dict using DCP
        hf_state_dict = adapter.to_hf(self.model.state_dict())
        dcp.load(hf_state_dict, storage_reader=storage_reader)

        # Convert HF state dict to TorchTitan format
        torchtitan_state_dict = adapter.from_hf(hf_state_dict)

        model_state_dict = {k: v for k, v in self.model.state_dict().items()}

        # Convert to DTensor if target is DTensor (when the target model is sharded)
        # This only happens when initial loading from HF full state dict
        for name, tensor in torchtitan_state_dict.items():
            if name in model_state_dict and isinstance(model_state_dict[name], DTensor):
                if isinstance(tensor, DTensor):
                    continue
                target_dtensor = model_state_dict[name]
                device_mesh = target_dtensor.device_mesh
                torchtitan_state_dict[name] = DTensor.from_local(
                    tensor.to(device_mesh.device_type),
                    device_mesh=device_mesh,
                    placements=[Replicate()],
                )

        return self.load_weights_from_state_dict(torchtitan_state_dict)

    def load_weights(self, weights_iter):
        """
        vLLM required API.

        This is a no-op method since model weights are already loaded during initialization.
        Returns the names of all parameters that have been loaded so vLLM's safety check passes.

        Args:
            weights_iter: Iterator of (name, tensor) pairs from HF checkpoint

        Returns:
            Set of loaded parameter names
        """

        loaded_param_names = set()
        for name, _ in self.model.named_parameters():
            loaded_param_names.add("model." + name)

        logger.info(
            f"Weights already loaded during model initialization. \
            Returning {len(loaded_param_names)} loaded parameter names to satisfy vLLM safety check."
        )

        # Return the names of all loaded parameters so vLLM knows they were handled
        return loaded_param_names
