# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import importlib
import math
import os as os
from dataclasses import dataclass, field, fields, MISSING
from fractions import Fraction
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import init
from torch.nn.attention.flex_attention import and_masks
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from . import _meta_init, _structure
from transformers.integrations.flex_attention import flex_attention_forward
from transformers.modeling_utils import AttentionInterface, PreTrainedModel

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common.attention import (
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
)
from torchtitan.models.utils import quadratic_attention_flops_per_token
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict
from torchtitan.tools.logging import logger


class HFFlexKernel(Module):
    """Flex-attention kernel wrapped as a titan Module for declarative TP.

    Runs the flex HOP over q/k/v. Under TP the Module protocol wraps this
    forward with ``local_map`` (driven by the ``ShardingConfig`` set in
    hf_sharding.py): q/k/v arrive head-sharded as DTensors, are converted to
    local tensors so the document ``mask_mod`` -- which closes over a plain
    ``positions`` tensor -- sees plain tensors, and the output is wrapped back
    head-sharded. Expressing the sharding declaratively
    (``ShardingConfig``/``LocalMapConfig``) keeps it consistent with Titan's own
    attention and lets it ride the ``spmd_types`` backend switch, instead of a
    hand-rolled ``local_map`` call.

    The HF attention module and the BlockMask ride as passthrough keyword args
    (non-tensors, so ``local_map`` leaves them untouched). CP is not handled
    here (guarded in ``parallelize_hf_transformers``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: "HFFlexKernel.Config") -> None:
        super().__init__()

    def forward(self, query, key, value, *, module, block_mask=None, **kwargs):
        # flex_attention_forward returns (output, lse); output is already
        # transposed to (b, seq, heads, dim). Return the single tensor so the
        # local_map out_placements is a 1-tuple.
        out, _ = flex_attention_forward(module, query, key, value, block_mask, **kwargs)
        return out


def _flex_attention_torchtitan(module, query, key, value, attention_mask, **kwargs):
    """HF ``AttentionInterface`` shim for flex attention.

    Delegates to the per-attention-module ``HFFlexKernel`` when present (attached
    under TP/EP in hf_sharding.py) so the Module protocol applies the declarative
    ``local_map``. When no kernel is attached (e.g. FSDP-only, where the sharding
    pass does not run), q/k/v are plain tensors and flex runs directly -- no
    mapping needed. CP is not handled here (see the guard in
    ``parallelize_hf_transformers``).
    """
    kernel = getattr(module, "_titan_flex_kernel", None)
    if kernel is None:
        return flex_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )
    out = kernel(query, key, value, module=module, block_mask=attention_mask, **kwargs)
    return out, None


class SliceableModuleDict(ModuleDict):
    """
    A ModuleDict that supports slicing like ModuleList.
    Keys are expected to be string representations of integers (e.g., "0", "1", "2").
    """

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Handle slicing: convert slice to list of keys
            keys = sorted(
                self.keys(), key=lambda x: int(x) if x.isdigit() else float("inf")
            )
            sliced_keys = keys[key]
            # Return a new SliceableModuleDict with the sliced items
            return SliceableModuleDict({k: self[k] for k in sliced_keys})
        return super().__getitem__(key)

    def __iter__(self):
        # Iterate over values in sorted order by key (as integers)
        keys = sorted(
            self.keys(), key=lambda x: int(x) if x.isdigit() else float("inf")
        )
        for key in keys:
            yield self[key]

    def __len__(self):
        return len(self._modules)

    def init_states(self, **_kwargs) -> None:
        """No-op: HFTransformerModel handles initialization via HF mechanisms."""
        pass


# Define all possible mappings organized by argument type
_TT_TO_HF_MAPPINGS = {
    "dense": {
        # TorchTitan dense model mappings (always available)
        "dim": "hidden_size",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "n_kv_heads": "num_key_value_heads",
        "vocab_size": "vocab_size",
        "norm_eps": "rms_norm_eps",
        "max_seq_len": "max_position_embeddings",
        "eos_id": "eos_token_id",
    },
    # MoE attrs use the same names in TorchTitan and HuggingFace, no remapping needed
    "moe": {},
}

# Declarative list of TorchTitan-only attributes (no HF equivalent)
_TT_SPECIFIC_ATTRIBUTES = [
    "multiple_of",
    "ffn_dim_multiplier",
    "depth_init",
    "attn_mask_type",
]

# NOTE: This backend instantiates model classes from the installed
# ``transformers`` package. When a model repo ships an older remote config
# implementation, mixing that remote config with the newer local model code can
# drop compatibility attrs the local model expects. Prefer the local
# ``transformers`` config/model pair for denylisted model types.
_REMOTE_CONFIG_DENYLIST = frozenset({"deepseek_v2", "deepseek_v3"})


# TorchTitan initializes an untied token embedding at unit std.
_TITAN_EMBED_INIT_STD = 1.0

_ATTN_IMPLEMENTATION = "flex_torchtitan"


def _uses_dsa(config) -> bool:
    """True if the model uses DeepSeek-style sparse attention (DSA).

    DSA models (e.g. GLM-5, model_type 'glm_moe_dsa') run an auxiliary
    "indexer" sub-attention that scores all keys and selects the top-k per
    query, expressing the selection as a dense additive mask. The indexer and
    the main attention both consume the incoming attention mask as a plain
    tensor -- the modeling code calls ``.dim()`` on it, adds it to the score
    matrix, and combines it with the top-k selection into a dense additive mask
    that is then handed to the attention interface. A flex ``BlockMask`` has no
    ``.dim()`` and cannot be added elementwise, so DSA needs a dense 4D tensor
    mask instead of a BlockMask (flex still runs, consuming the dense mask as a
    ``score_mask``). Detected by the DSA-specific ``index_topk`` config attr.
    """
    return getattr(config, "index_topk", None) is not None


class HFTransformerModel(BaseModel):
    # TODO(#ISSUE): Remove after fixing PP backward to skip non-tensor inputs.
    _skip_lm_head: bool = False

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config, PretrainedConfig):
        """Configuration that bridges TorchTitan and HuggingFace Transformers.

        Uses properties to provide TorchTitan-style access while maintaining
        HuggingFace compatibility.
        """

        # Redeclare PretrainedConfig dataclass fields as init=False so
        # ``Configurable.__init_subclass__`` kw_only check passes.  These
        # fields are set dynamically in ``__init__`` via
        # ``PretrainedConfig.__init__``, not through the generated __init__.
        transformers_version: str = field(init=False, default="")
        architectures: list | None = field(init=False, default=None)
        output_hidden_states: bool = field(init=False, default=False)
        return_dict: bool = field(init=False, default=True)
        dtype: str | None = field(init=False, default=None)
        chunk_size_feed_forward: int = field(init=False, default=0)
        is_encoder_decoder: bool = field(init=False, default=False)
        id2label: dict | None = field(init=False, default=None)
        label2id: dict | None = field(init=False, default=None)
        problem_type: str | None = field(init=False, default=None)

        def __init__(
            self,
            model_config,
            **kwargs,
        ):
            # Explicitly call PretrainedConfig.__init__ (not via MRO, since
            # Configurable.Config's generated __init__ doesn't chain to it)
            PretrainedConfig.__init__(
                self, attn_implementation=_ATTN_IMPLEMENTATION, **kwargs
            )
            # Set param_init and sharding_config before Module.Config.build()
            # accesses them. PretrainedConfig.__getattribute__ doesn't
            # recognize slots inherited from Module.Config.
            self.param_init = (
                None  # noqa: this sets Config.param_init, not Module._param_init
            )
            self.sharding_config = None

            assert model_config is not None, "model_config is required"

            from torchtitan.experiments.transformers_modeling_backend import (
                TitanMoeModelConfig,
            )

            self.is_moe = isinstance(model_config, TitanMoeModelConfig)

            # Create getter/setter dynamically for TT <-> HF attribute mappings
            self._create_getter_setter_dynamically(is_moe=self.is_moe)

            self._titan_injected_model_args = {}
            self._configure_hf_attention()

            self._initialize_attributes(model_config)

        def build(self, **kwargs):
            """Override build() to use _replace() instead of dataclasses.replace().

            dataclasses.replace() re-invokes __init__, which is incompatible
            with the custom __init__ here (expects model_config).
            """
            clone = self._replace()
            # HF modules compute real buffers in __init__ (causal masks, rope
            # tables, embedding normalizers) and a plain meta build throws all
            # of them away. See _meta_init.py.
            with _meta_init.params_meta_buffers_real():
                instance = self._owner(config=clone, **kwargs)
            if self.param_init is not None:
                instance._param_init = self.param_init
            return instance

        def _replace(self, **overrides):
            """Override to use ``copy.copy()`` instead of ``dataclasses.replace()``.

            ``dataclasses.replace()`` re-invokes ``__init__``, which is
            incompatible with the custom ``__init__`` here (it expects
            ``model_config`` and calls ``PretrainedConfig.__init__``).
            A shallow copy preserves all dynamically-set HF attributes.
            """
            clone = copy.copy(self)
            for f in fields(self):
                if f.init:
                    continue
                if f.name in overrides:
                    setattr(clone, f.name, overrides[f.name])
                elif hasattr(self, f.name):
                    setattr(clone, f.name, getattr(self, f.name))
                else:
                    raise TypeError(
                        f"{type(self).__name__} field '{f.name}' "
                        f"(init=False) was not provided via build()"
                    )
            return clone

        def _initialize_attributes(self, model_config):
            """Initialize all model attributes from the config.

            Only stores explicitly-set (non-default) fields in
            ``_titan_injected_model_args`` so that ``update_from_config``
            only overrides HF config values the user intentionally set
            in the flavor, preserving model-specific HF attrs like
            ``qk_head_dim`` or ``n_routed_experts``.
            """
            # Determine which fields were explicitly set (not defaults)
            explicit_overrides = {}
            for f in fields(model_config):
                value = getattr(model_config, f.name)
                default = f.default
                if default is MISSING:
                    # No default — always explicit
                    explicit_overrides[f.name] = value
                elif value != default:
                    explicit_overrides[f.name] = value

            # Set mapped attributes (TorchTitan -> HuggingFace)
            for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
                if hasattr(model_config, titan_name):
                    setattr(self, hf_name, getattr(model_config, titan_name))

            # Set all remaining attributes directly (TorchTitan-only + MoE)
            for attr_name, value in vars(model_config).items():
                if (
                    not attr_name.startswith("_")
                    and attr_name not in self._tt_to_hf_attribute_map
                ):
                    setattr(self, attr_name, value)

            # Store only EXPLICIT overrides for re-application after HF
            # config load. Mapped attrs use HF names; others use titan names.
            for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
                if titan_name in explicit_overrides:
                    self._titan_injected_model_args[hf_name] = explicit_overrides[
                        titan_name
                    ]
            for attr_name, value in explicit_overrides.items():
                if attr_name not in self._tt_to_hf_attribute_map:
                    self._titan_injected_model_args[attr_name] = value

        def _configure_hf_attention(self):
            """Configure HuggingFace attention to route through flex attention.

            Routes attention through the flex HOP so a causal or document/packing
            BlockMask can be applied -- is_causal alone cannot express
            cross-sample (packed)
            masking. The titan-built BlockMask (see ``get_attention_masks``)
            rides HF's normal ``attention_mask`` argument (HF returns an
            already-4D/BlockMask mask as-is), so no custom mask plumbing is
            needed. The custom impl name only exists to bypass HF's per-model
            ``_supports_flex_attn`` gate.
            """
            AttentionInterface._global_mapping[
                _ATTN_IMPLEMENTATION
            ] = _flex_attention_torchtitan
            self._titan_injected_model_args[
                "attn_implementation"
            ] = _ATTN_IMPLEMENTATION
            self.attn_implementation = _ATTN_IMPLEMENTATION
            # HF selects the attention function from ``config._attn_implementation``.
            # PretrainedConfig has no ``attn_implementation`` property in this
            # version, so the line above only sets a dead plain attribute -- set the
            # underscore field directly (it is preserved through update_from_config,
            # which skips underscore keys when copying the loaded HF config).
            self._attn_implementation = _ATTN_IMPLEMENTATION

        def _create_getter_setter_dynamically(self, is_moe: bool):
            """
            Create properties dynamically based on tt and hf attribute mappings.
            For example, creates a property 'dim' that reads/writes to 'hidden_size'.
            """

            def _create_property(hf_name: str) -> property:
                def getter(self):
                    return getattr(self, hf_name)

                def setter(self, value):
                    setattr(self, hf_name, value)

                return property(getter, setter)

            # Setup attribute mappings
            self._tt_to_hf_attribute_map = dict(_TT_TO_HF_MAPPINGS["dense"])
            if is_moe:
                self._tt_to_hf_attribute_map.update(_TT_TO_HF_MAPPINGS["moe"])

            for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
                # Identity mappings (e.g. vocab_size -> vocab_size) need no
                # property: the HF attribute is already reachable under the same
                # name. Creating one would make the setter ``setattr(self,
                # hf_name)`` recurse into itself. The map entry is still used by
                # _initialize_attributes to copy the value from the titan config.
                if titan_name == hf_name:
                    continue
                # Create getter/setter for attribute that don't already exist
                if not hasattr(self.__class__, titan_name):
                    setattr(self.__class__, titan_name, _create_property(hf_name))

        def __repr__(self) -> str:
            args_lines = [
                f"{k}={getattr(self, k)!r}"
                for k in sorted(self._titan_injected_model_args.keys())
                if hasattr(self, k)
            ]
            args_str = "\n".join(args_lines)
            return f"{self.__class__.__name__}(\n{args_str}\n)"

        def update_from_config(
            self,
            *,
            config=None,
            **kwargs,
        ):
            training = config.training
            parallelism = config.parallelism
            debug = config.debug
            # Extract HF model ID from the extended config
            hf_model_id = getattr(config, "hf_model", "")
            config_dict, _ = PretrainedConfig.get_config_dict(hf_model_id)
            trust_remote_code = (
                config_dict.get("model_type", "") not in _REMOTE_CONFIG_DENYLIST
            )
            # Load HF config (overwrites our HF attributes)
            # is_moe was decided from the *flavor* in __init__, before
            # --hf-model had been applied, so a MoE checkpoint selected on the
            # command line arrived with a dense flavor and the MoE replacement
            # never ran. hf_model_id is final here: ask the checkpoint.
            hf_model_config = AutoConfig.from_pretrained(
                hf_model_id,
                attn_implementation=self.attn_implementation,
                trust_remote_code=trust_remote_code,
            )

            if not getattr(self, "is_moe", False):
                for _attr in ("num_experts", "n_routed_experts", "num_local_experts"):
                    for _holder in (
                        hf_model_config,
                        getattr(hf_model_config, "text_config", None),
                    ):
                        _v = getattr(_holder, _attr, None) if _holder is not None else None
                        if isinstance(_v, int) and _v > 1:
                            logger.warning(
                                f"{hf_model_id} declares {_v} experts but the "
                                "selected flavor is dense, so the MoE replacement "
                                "will not run and HF's own MoE executes unreplaced. "
                                "Select a *_moe config for this model."
                            )
                            break
                    else:
                        continue
                    break

            # For composite (VL) models, use the nested text config so we
            # build the text-only CausalLM instead of the conditional model.
            if hasattr(hf_model_config, "text_config"):
                hf_model_config = hf_model_config.text_config
                hf_model_config.attn_implementation = self.attn_implementation
                # Ensure the text config has architectures for model class lookup
                if not getattr(hf_model_config, "architectures", None):
                    from transformers.models.auto.modeling_auto import (
                        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
                    )

                    model_type = getattr(hf_model_config, "model_type", "")
                    cls_name = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(model_type)
                    if cls_name:
                        hf_model_config.architectures = [cls_name]

            # Explicitly update attributes based on mappings
            for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
                if hasattr(hf_model_config, hf_name):
                    setattr(self, titan_name, getattr(hf_model_config, hf_name))

            # Copy all HF config attributes including computed ones.
            # to_dict() misses attrs computed in __init__ (e.g., DeepSeek V3's
            # qk_head_dim), so we copy from vars() which has them.
            for key, value in vars(hf_model_config).items():
                if not key.startswith("_"):
                    setattr(self, key, value)

            # Copy attribute_map for models that alias config names
            # (e.g., DeepSeek V3 maps num_local_experts → n_routed_experts)
            if hf_model_config.attribute_map:
                self.attribute_map.update(hf_model_config.attribute_map)

            # Re-apply explicitly-set flavor overrides (not defaults)
            for key, value in self._titan_injected_model_args.items():
                setattr(self, key, value)
                # Sync expert count aliases for models that use different naming
                # (e.g., DeepSeek V3 uses n_routed_experts, GLM uses num_local_experts)
                if key == "num_experts" and hasattr(self, "n_routed_experts"):
                    self.n_routed_experts = value

            self.max_seq_len = training.max_context_length

            if hasattr(config.loss, "global_vocab_size"):
                config.loss.global_vocab_size = self.vocab_size

            self.deterministic = debug.deterministic

            # Configure HF-specific settings to match TorchTitan settings
            # TODO: false ?
            self.attention_bias = False
            self.mlp_bias = False
            self.use_cache = False
            # Keep the model's own initializer_range. Forcing 1.0 here also
            # reaches HF's stock _init_weights, which uses this field as the
            # std for every Linear/Conv1D/Embedding -- and the stock path runs
            # whenever the Llama-like patch does not match (GPT-2, BERT, XLM-R
            # are not named <Prefix>DecoderLayer). Those models were built at
            # std=1.0 instead of 0.02, which lands cross entropy near 120 where
            # ln(vocab) is 10.8, with nothing failing. TorchTitan's own init
            # states its 1.0 directly, at the one place it means it.
            if not hasattr(self, "initializer_range"):
                self.initializer_range = 0.02

            # When dim is explicitly overridden (e.g. debugmodel), derive the
            # dependent sizes from it. Otherwise keep what AutoConfig loaded from
            # the HF config -- models like Qwen3 decouple head_dim and
            # intermediate_size from hidden_size/num_heads, so deriving them here
            # would silently build the wrong architecture.
            dim_overridden = self._titan_injected_model_args.get("dim") is not None
            if (
                dim_overridden
                and not getattr(self, "is_moe", False)
                and not hasattr(self, "inter_dim")
            ):
                ffn_hidden_size = 4 * self.dim
                ffn_hidden_size = int(2 * ffn_hidden_size / 3)
                if self.ffn_dim_multiplier is not None:
                    ffn_hidden_size = int(self.ffn_dim_multiplier * ffn_hidden_size)
                self.intermediate_size = self.multiple_of * (
                    (ffn_hidden_size + self.multiple_of - 1) // self.multiple_of
                )

            # MLA models (DeepSeek V3, GLM-5) set head_dim = qk_rope_head_dim
            # in the HF config for RoPE; don't clobber it with the standard
            # computation. Also force num_key_value_heads = num_attention_heads
            # because MLA has no GQA -- the KV LoRA path always produces
            # num_attention_heads heads.
            if hasattr(self, "qk_rope_head_dim"):
                self.num_key_value_heads = self.num_attention_heads
                # Ensure head_dim is set for MLA models; remote configs
                # may not compute it in __post_init__.
                if not getattr(self, "head_dim", None):
                    self.head_dim = self.qk_rope_head_dim
            elif dim_overridden:
                # dim explicitly overridden: derive head_dim from it.
                self.head_dim = self.dim // self.num_attention_heads
            elif not getattr(self, "head_dim", None):
                # HF config did not provide head_dim; use the standard
                # derivation. A composite config (CLIP, BLIP, SegFormer) states
                # neither dim nor num_attention_heads at the top level, which
                # makes that a TypeError rather than a wrong number -- read the
                # towers' own sizes instead.
                if self.dim is not None and getattr(self, "num_attention_heads", None):
                    self.head_dim = self.dim // self.num_attention_heads
                else:
                    self.head_dim = _structure.derive_head_dim(self)

            # Ensure expert groups are consistent with (possibly overridden)
            # num_experts for models with group-level routing (DeepSeek V3).
            # Each group needs >= 2 experts for the in-group topk(2).
            if hasattr(self, "n_group") and hasattr(self, "n_routed_experts"):
                while self.n_group > 1 and self.n_routed_experts // self.n_group < 2:
                    self.n_group //= 2
                if hasattr(self, "topk_group"):
                    self.topk_group = min(self.topk_group, self.n_group)

            return self

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            assert isinstance(model, HFTransformerModel)
            named_parameters = list(model.named_parameters())
            nparams = sum(param.numel() for _, param in named_parameters)
            parameter_weights = {
                id(param): Fraction(1) for _, param in named_parameters
            }

            lm_head = getattr(model, "lm_head", None)
            lm_head_parameter_ids = (
                {id(param) for param in lm_head.parameters()}
                if isinstance(lm_head, nn.Module)
                else set()
            )
            for module in model.modules():
                if isinstance(module, nn.Embedding):
                    for param in module.parameters(recurse=False):
                        if id(param) not in lm_head_parameter_ids:
                            parameter_weights[id(param)] = Fraction(0)

            for layer in model.layers.values():
                moe_config = getattr(layer, "_native_moe_config", None)
                if moe_config is None:
                    continue
                active_expert_ratio = Fraction(
                    moe_config.router.top_k, moe_config.num_experts
                )
                if getattr(layer, "_layer_level_moe", False):
                    moe_block = layer
                else:
                    moe_attr = _structure.moe_attr_name(layer)
                    if moe_attr is None:
                        logger.warning(
                            "HF MFU calculation could not identify the MoE block; "
                            "counting all expert parameters as active."
                        )
                        continue
                    moe_block = getattr(layer, moe_attr)
                for param in moe_block.experts.parameters():
                    parameter_weights[id(param)] = active_expert_ratio

            nparams_for_matmul = sum(
                param.numel() * parameter_weights[id(param)]
                for _, param in named_parameters
            )
            assert nparams_for_matmul.denominator == 1
            active_nparams = nparams_for_matmul.numerator

            if all(
                hasattr(self, name)
                for name in ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
            ):
                qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
                v_head_dim = self.v_head_dim
            else:
                qk_head_dim = self.head_dim
                v_head_dim = self.head_dim

            layer_types = getattr(self, "layer_types", None)
            if layer_types is None:
                default_layer_type = (
                    "sliding_attention"
                    if getattr(self, "sliding_window", None) is not None
                    else "full_attention"
                )
                layer_types = [default_layer_type] * len(model.layers)
            else:
                # Debug flavors can override num_hidden_layers while retaining the full
                # architecture's layer_types list. Count only the layers that were
                # built.
                layer_types = list(layer_types[: len(model.layers)])
                if len(layer_types) < len(model.layers):
                    logger.warning(
                        "HF config defines fewer layer_types than built layers; "
                        "approximating the remaining layers as full attention."
                    )
                    layer_types.extend(
                        ["full_attention"] * (len(model.layers) - len(layer_types))
                    )

            attention_op_flops = 0
            unsupported_layer_types = set()
            for layer_type in layer_types:
                if layer_type in ("attention", "full_attention"):
                    sliding_window_size = None
                elif layer_type == "sliding_attention":
                    sliding_window_size = getattr(self, "sliding_window", None)
                elif layer_type == "chunked_attention":
                    sliding_window_size = getattr(self, "attention_chunk_size", None)
                else:
                    unsupported_layer_types.add(str(layer_type))
                    sliding_window_size = None

                layer_qk_head_dim = qk_head_dim
                layer_v_head_dim = v_head_dim
                if getattr(
                    self, "model_type", None
                ) == "gemma4_text" and layer_type in (
                    "attention",
                    "full_attention",
                ):
                    global_head_dim = getattr(self, "global_head_dim", None)
                    if global_head_dim is not None:
                        layer_qk_head_dim = global_head_dim
                        layer_v_head_dim = global_head_dim

                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=self.n_heads,
                    qk_head_dim=layer_qk_head_dim,
                    v_head_dim=layer_v_head_dim,
                    seq_len=seq_len,
                    sliding_window_size=sliding_window_size,
                )

            if unsupported_layer_types:
                logger.warning(
                    "HF MFU calculation does not recognize layer types "
                    f"{sorted(unsupported_layer_types)}; "
                    "approximating them as full attention."
                )

            num_flops_per_token = 6 * active_nparams + attention_op_flops
            logger.info(
                f"Total parameter count: {nparams:,}, "
                f"active parameters: {active_nparams:,}"
            )
            return nparams, num_flops_per_token

    @staticmethod
    def _set_attn_everywhere(config, attn: str) -> None:
        """Set the attention implementation on a config and its sub-configs.

        A model that builds a nested backbone from its own sub-config
        (grounding-dino's SwinBackbone) never sees a change made only at the
        top level.
        """
        config._attn_implementation = attn
        config.attn_implementation = attn
        for name in dir(config):
            if not name.endswith("_config"):
                continue
            sub = getattr(config, name, None)
            if sub is not None and hasattr(sub, "_attn_implementation"):
                sub._attn_implementation = attn
                sub.attn_implementation = attn

    @staticmethod
    def _build_with_supported_attention(model_cls, config):
        """Build the model, backing off to an attention path it accepts.

        Registering an implementation with AttentionInterface only reaches
        architectures that dispatch through it. The BERT family indexes a
        fixed class dict instead -- BERT_SELF_ATTENTION_CLASSES[
        config._attn_implementation] -- so a name it does not know is a
        KeyError in __init__, and Electra and the T5 family raise ValueError
        for the same reason. Attention that is slower is not attention that is
        wrong.
        """
        wanted = getattr(config, "_attn_implementation", None)
        attempts = [wanted] + [a for a in ("sdpa", "eager") if a != wanted]
        last = None
        for attempt in attempts:
            if attempt != wanted:
                HFTransformerModel._set_attn_everywhere(config, attempt)
            try:
                model = model_cls(config=config)
            except (ValueError, KeyError) as e:
                text = str(e)
                if attempt is not None and attempt not in text and (
                    "attention implementation" not in text
                ):
                    raise
                last = e
                continue
            if attempt != wanted:
                logger.warning(
                    f"{model_cls.__name__} does not accept "
                    f"attn_implementation={wanted!r}; built with {attempt!r}."
                )
            return model
        raise last

    def __init__(self, config: Config):
        super().__init__()

        # Try to import the model class dynamically from the transformers library if not found in globals
        model_class_name = config.architectures[0]
        model_cls = globals().get(model_class_name, None)
        if model_cls is None:
            try:
                transformers_mod = importlib.import_module("transformers")
                model_cls = getattr(transformers_mod, model_class_name)
            except (ImportError, AttributeError):
                model_cls = None

        if model_cls is None:
            # Fallback: resolve via model_type → Auto mapping.
            # Handles cases where the config's architecture name doesn't match
            # the actual class name.
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
            )

            model_type = getattr(config, "model_type", "")
            resolved_name = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(model_type)
            if resolved_name:
                transformers_mod = importlib.import_module("transformers")
                model_cls = getattr(transformers_mod, resolved_name, None)
                if model_cls is not None:
                    model_class_name = resolved_name

            if model_cls is None:
                raise ImportError(
                    f"Could not find model class '{model_class_name}' in globals or transformers. "
                    f"Make sure the class is available."
                )

        # Attempt to patch model weight initialization based on architecture type
        try:
            model_name_prefix = model_class_name.replace("ForCausalLM", "")
            model_module = importlib.import_module(model_cls.__module__)

            # Some models name their text-stack classes with a "Text" infix
            # (e.g. Gemma4 -> Gemma4TextAttention / Gemma4TextDecoderLayer,
            # while the causal-LM class stays Gemma4ForCausalLM). Prefer the
            # plain prefix, but fall back to "<prefix>Text" when the plain
            # prefix does not resolve the attention/decoder classes -- otherwise
            # the init patch is silently skipped and HF's default init (with the
            # backend's initializer_range=1.0) blows up the loss.
            for candidate in (model_name_prefix, f"{model_name_prefix}Text"):
                if (
                    getattr(model_module, f"{candidate}Attention", None) is not None
                    and getattr(model_module, f"{candidate}DecoderLayer", None)
                    is not None
                ):
                    model_name_prefix = candidate
                    break

            attention_cls = getattr(model_module, f"{model_name_prefix}Attention", None)
            mlp_cls = getattr(model_module, f"{model_name_prefix}MLP", None)
            decoder_layer_cls = getattr(
                model_module, f"{model_name_prefix}DecoderLayer", None
            )

            # Discover MoE-specific classes
            experts_cls = getattr(model_module, f"{model_name_prefix}Experts", None)
            router_cls = getattr(model_module, f"{model_name_prefix}TopKRouter", None)

            required_classes = {
                "Attention": attention_cls,
                "DecoderLayer": decoder_layer_cls,
            }

            if all(required_classes.values()):
                logger.info(f"Applying Llama-like patch for {model_name_prefix}")
                self._patch_hf_llama_like(
                    decoder_layer_cls=decoder_layer_cls,
                    attention_cls=attention_cls,
                    mlp_cls=mlp_cls,  # mlp_cls can be None
                    experts_cls=experts_cls,
                    router_cls=router_cls,
                )
            else:
                missing = [name for name, cls in required_classes.items() if not cls]
                logger.warning(
                    f"Could not find required classes ({', '.join(missing)}) for {model_name_prefix}. "
                    "Skipping Llama-like patch."
                )

        except Exception as e:
            logger.warning(
                f"Failed to apply agnostic patch for {model_class_name} due to: {e}. "
                "Weight initialization might not match TorchTitan."
            )

        # Select the HF experts forward kernel from the explicit request in
        # TitanMoeModelConfig.experts_implementation. Honor it or fail — never
        # silently substitute a different kernel than the user asked for.
        #
        # - "native": use the HF model's own built-in experts kernel unchanged
        #   (valid for every model; the only valid choice for models that can't
        #   take a settable implementation).
        # - "grouped_mm"/"batched_mm"/"eager": require a model that supports a
        #   settable experts implementation (the @use_experts_implementation
        #   decorator, probed up front as a classmethod). If the model can't, the
        #   request cannot be honored -> raise rather than ignore it.
        #
        # All paths work transparently with hook-based EP/TP because the hooks
        # preserve the (hidden_states, top_k_index, top_k_weights) interface.
        if config.is_moe:
            impl = config.experts_implementation
            if impl == "native":
                config._experts_implementation = None
            elif model_cls._can_set_experts_implementation():
                config._experts_implementation = impl
            else:
                raise ValueError(
                    f"{model_class_name} does not support a settable experts "
                    f"implementation, so experts_implementation='{impl}' cannot "
                    "be honored. Set experts_implementation='native' to use the "
                    "HF model's built-in experts kernel."
                )

        # Size the learned position table to hold max_seq_len before building.
        # The config bridges max_seq_len -> max_position_embeddings, but a
        # family that reserves slots at the front addresses fewer rows than it
        # has: XLM-R starts position ids at padding_idx + 1, so a 2048-row
        # table tops out at 2046 and the last two positions of every batch read
        # out of range -- silently on TPU, where the gather clamps.
        _wanted = getattr(config, "max_seq_len", None)
        if _wanted:
            try:
                with torch.device("meta"):
                    _probe = model_cls(config=config)
                _cap = _structure.position_capacity(_probe)
                if _cap is not None and _cap < _wanted:
                    config.max_position_embeddings = (
                        config.max_position_embeddings + (_wanted - _cap)
                    )
                    logger.info(
                        f"Grew the position table by {_wanted - _cap} to address "
                        f"{_wanted} positions ({type(_probe).__name__} reserves "
                        "slots at the front of its table)."
                    )
            except Exception as e:
                logger.warning(f"Could not size the position table: {e}")

        self.model = self._build_with_supported_attention(model_cls, config)
        self.max_seq_len = config.max_seq_len
        self.cp_mesh = None
        self._position_capacity = _structure.position_capacity(self.model)
        # T5 positions with relative bias and DistilBERT with its own
        # embedding, so neither takes position_ids; passing it is a TypeError
        # rather than something they ignore. Ask the signature once.
        self._forward_kwargs = _structure.accepted_kwargs(self.model.forward)

        # BERT, DistilBERT and XLM-R normalize inside every block and end
        # without a trailing norm. Callers still expect the part to exist, so
        # stand in an Identity, which is what "no final norm" computes.
        self._added_norm = None
        if _structure.find_norm(self.model)[0] is None:
            self._added_norm = nn.Identity()

        # A bare encoder has no output projection, and a classification head
        # maps to num_labels rather than to the vocabulary -- neither can
        # produce token logits. Attach a head tied to the token embedding, as
        # any fine-tuning harness does to put a token objective on such a
        # model. This adds parameters the checkpoint does not carry; it is for
        # training and measurement, not for restoring a published head.
        self._added_lm_head = None
        _head, _ = _structure.find_head(self.model)
        _head_out = _structure.head_out_features(_head)
        _vocab = getattr(config, "vocab_size", None)
        if _head is not None and _head_out is not None and _vocab and _head_out != _vocab:
            logger.info(
                f"{type(self.model).__name__}'s head maps to {_head_out} labels, "
                f"not the {_vocab}-token vocabulary; attaching an lm_head and "
                "training the backbone on a token objective instead."
            )
            _head = None
        if _head is None:
            _emb, _ = _structure.find_embedding(self.model)
            if isinstance(_emb, nn.Embedding):
                self._added_lm_head = nn.Linear(
                    _emb.embedding_dim, _emb.num_embeddings, bias=False
                )
                self._added_lm_head.weight = _emb.weight
                logger.info(
                    f"{type(self.model).__name__} has no usable output projection; "
                    f"attached an lm_head tied to the token embedding "
                    f"({_emb.embedding_dim} -> {_emb.num_embeddings})."
                )

        # Convert ModuleList to ModuleDict to preserve original indices
        # This ensures state dict keys match checkpoint keys
        _blocks, _blocks_path = _structure.find_layers(self.model)
        if _blocks is None:
            raise AttributeError(
                f"Could not locate the block stack on {type(self.model).__name__}"
            )
        if isinstance(_blocks, nn.ModuleList):
            _blocks = SliceableModuleDict(
                {str(i): layer for i, layer in enumerate(_blocks)}
            )
            _structure.set_at(self.model, _blocks_path, _blocks)

        for layer in _blocks.values():
            # Detect MoE layers by checking for gate/router and experts sub-modules.
            # Gemma4 has router/experts as siblings of the dense MLP at the
            # layer level (not inside ``mlp``).
            moe_attr = _structure.moe_attr_name(layer)
            moe_module = getattr(layer, moe_attr, None) if moe_attr else None
            if moe_module is not None:
                has_gate = hasattr(moe_module, "gate") or hasattr(moe_module, "router")
                layer.moe_enabled = has_gate and hasattr(moe_module, "experts")
            else:
                layer.moe_enabled = False

            # Layer-level MoE: router and experts are direct children of the
            # decoder layer, not nested inside the MLP block (e.g. Gemma4).
            if not layer.moe_enabled:
                has_layer_router = hasattr(layer, "router") or hasattr(layer, "gate")
                has_layer_experts = hasattr(layer, "experts")
                if has_layer_router and has_layer_experts:
                    layer.moe_enabled = True
                    layer._layer_level_moe = True

        if config.is_moe:
            from .moe_replacement import prepare_native_moe_configs

            prepare_native_moe_configs(self, config)

    def set_cp_mesh(self, mesh):
        self.cp_mesh = mesh

    def _patch_hf_llama_like(
        self,
        decoder_layer_cls,
        attention_cls,
        mlp_cls=None,
        experts_cls=None,
        router_cls=None,
    ):
        """
        This patch modifies a Hugging Face Llama-like model's weight initialization to match
        the initialization scheme used in TorchTitan. This is crucial for ensuring
        bit-for-bit reproducibility when converting checkpoints between the
        TorchTitan format and the Hugging Face format.

        The patch targets the following aspects of the model:
        - `PreTrainedModel._initialize_weights`: Handles meta device initialization correctly.
        - `PreTrainedModel._init_weights`: Implements TorchTitan's specific initialization
          for attention, MLP, embedding, and layer norm layers. This includes depth-dependent
          initialization for attention and MLP layers.
        - `DecoderLayer.__init__`: Adds `layer_idx` to attention and MLP modules within
          each decoder layer, which is required for the depth-dependent initialization.
        """

        _original_decoder_layer_init = decoder_layer_cls.__init__

        def _decoder_layer_init_patched(self, config: PretrainedConfig, layer_idx: int):
            _original_decoder_layer_init(self, config, layer_idx)
            self.layer_idx = layer_idx
            # Ensure both attention and mlp modules have layer_idx for depth-based init
            if hasattr(self, "self_attn"):
                self.self_attn.layer_idx = layer_idx
            # some models might not have mlp in each layer
            if hasattr(self, "mlp") and self.mlp is not None:
                self.mlp.layer_idx = layer_idx
                # Propagate to shared experts (e.g., Qwen2 MoE, DeepSeek V3)
                for shared_name in ("shared_expert", "shared_experts"):
                    shared = getattr(self.mlp, shared_name, None)
                    if shared is not None:
                        shared.layer_idx = layer_idx

        def _initialize_weights_patched(self, module, is_remote_code: bool = False):
            # NOTE(3outeille): monkey-patch PreTrainedModel to handle meta device initialization correctly
            # The default _initialize_weights sets _is_hf_initialized = True even on a meta device,
            # which prevents subsequent proper initialization.
            #
            # This mirrors HF's PreTrainedModel._initialize_weights and only adds
            # the meta-device early-return below. The `is_remote_code` arg and the
            # branch that follows are copied verbatim from HF: transformers 5.x
            # calls this via smart_apply as `fn(module, self.is_remote_code())`,
            # so the replacement must accept the arg and keep the remote-code
            # guard (remote _init_weights may write params in place, so already
            # initialized modules must be skipped). Source (transformers v5.9.0):
            # https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/modeling_utils.py#L2464
            if getattr(module, "_is_hf_initialized", False):
                return

            if (
                is_remote_code
                and all(
                    getattr(param, "_is_hf_initialized", False)
                    for param in module.parameters(recurse=False)
                )
                and all(
                    getattr(buffer, "_is_hf_initialized", False)
                    for buffer in module.buffers(recurse=False)
                    if buffer is not None
                )
            ):
                module._is_hf_initialized = True
                return

            for param in module.parameters(recurse=True):
                if param.device.type == "meta":
                    return

            # If not on a meta device, call the original weight initialization
            self._init_weights(module)
            module._is_hf_initialized = True

        def _init_weights_patched(self, module):
            """
            Patched version of _init_weights to match TorchTitan's initialization for Llama-like models.
            `self` is a PreTrainedModel instance.
            """
            config = self.config
            # Build tuple of classes to check for layer_idx-based init_std calculation
            layer_idx_classes = [attention_cls]
            if mlp_cls:
                layer_idx_classes.append(mlp_cls)
            layer_idx_classes = tuple(layer_idx_classes)

            if isinstance(module, layer_idx_classes):
                if not hasattr(module, "layer_idx"):
                    raise ValueError(
                        f"Module {module} does not have a layer_idx attribute"
                    )

                layer_idx = module.layer_idx

                if hasattr(config, "depth_init") and config.depth_init:
                    init_std = 0.02 / (2 * (layer_idx + 1)) ** 0.5
                else:
                    init_std = 0.02 / (2 * config.num_hidden_layers) ** 0.5

            if isinstance(module, attention_cls):
                # Initialize weights and biases for q, k, v projections with
                # std=0.02, matching Titan models. Besides full-rank q/k/v,
                # cover the MLA low-rank projections (DeepSeek V2/V3, GLM):
                # q_a_proj/q_b_proj (low-rank Q) and kv_a_proj_with_mqa/kv_b_proj
                # (low-rank KV). Titan's deepseek_v3 inits all of wq_a/wq_b/wkv_a/
                # wkv_b at std=0.02 too; only the output proj is depth-scaled.
                # The MLA RMSNorms (q_a_layernorm/kv_a_layernorm) are handled by
                # the norm branch below. Missing these left them at their
                # construction-time values (not reset by init).
                for proj_name in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "q_a_proj",
                    "q_b_proj",
                    "kv_a_proj_with_mqa",
                    "kv_b_proj",
                ]:
                    proj = getattr(module, proj_name, None)
                    if proj is None:
                        continue
                    nn.init.trunc_normal_(proj.weight, mean=0.0, std=0.02)
                    if proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(proj.bias, -bound, bound)

                # Handle different names for the output projection layer
                o_proj = getattr(module, "o_proj", getattr(module, "dense", None))
                if o_proj is not None:
                    nn.init.trunc_normal_(o_proj.weight, mean=0.0, std=init_std)
                    if o_proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(o_proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(o_proj.bias, -bound, bound)

            elif mlp_cls and isinstance(module, mlp_cls):
                # Handle different names for MLP layers
                gate_proj = getattr(module, "gate_proj", getattr(module, "fc1", None))
                up_proj = getattr(module, "up_proj", None)
                down_proj = getattr(module, "down_proj", getattr(module, "fc2", None))

                # gate_proj (or fc1) should always use std=0.02 for numerical stability.
                if gate_proj is not None:
                    nn.init.trunc_normal_(gate_proj.weight, mean=0.0, std=0.02)
                    if gate_proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(gate_proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(gate_proj.bias, -bound, bound)
                # up_proj and down_proj (or fc2) use the depth-dependent init_std.
                if up_proj is not None:
                    nn.init.trunc_normal_(up_proj.weight, mean=0.0, std=init_std)
                    if up_proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(up_proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(up_proj.bias, -bound, bound)
                if down_proj is not None:
                    nn.init.trunc_normal_(down_proj.weight, mean=0.0, std=init_std)
                    if down_proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(down_proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(down_proj.bias, -bound, bound)

            elif experts_cls and isinstance(module, experts_cls):
                # MoE expert weights are 3D parameter tensors (not nn.Linear)
                if hasattr(module, "gate_up_proj"):
                    nn.init.trunc_normal_(module.gate_up_proj, mean=0.0, std=0.02)
                if hasattr(module, "down_proj"):
                    nn.init.trunc_normal_(module.down_proj, mean=0.0, std=0.02)

            elif router_cls and isinstance(module, router_cls):
                if hasattr(module, "weight"):
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)

            elif module is getattr(
                self, "lm_head", None
            ):  # TODO(3outeille): find a better way to detect lm_head
                final_out_std = config.hidden_size**-0.5
                cutoff_factor = 3
                nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )
                if module.bias is not None:
                    module.bias.data.zero_()

            elif isinstance(module, nn.Embedding):
                # When tie_word_embeddings is True, use lm_head initialization
                if (
                    hasattr(config, "tie_word_embeddings")
                    and config.tie_word_embeddings
                ):
                    final_out_std = config.hidden_size**-0.5
                    cutoff_factor = 3
                    nn.init.trunc_normal_(
                        module.weight,
                        mean=0.0,
                        std=final_out_std,
                        a=-cutoff_factor * final_out_std,
                        b=cutoff_factor * final_out_std,
                    )
                else:
                    # TorchTitan initializes an untied token embedding at
                    # std=1.0, independent of the HF config's initializer_range.
                    module.weight.data.normal_(mean=0.0, std=_TITAN_EMBED_INIT_STD)

                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

            elif (
                isinstance(
                    module,
                    (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
                )
                or "LayerNorm" in module.__class__.__name__
                or "RMSNorm" in module.__class__.__name__
            ):
                # Norms can exist without weights (in which case they are None from torch primitives)
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight.data.fill_(1.0)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data.zero_()

        decoder_layer_cls.__init__ = _decoder_layer_init_patched
        PreTrainedModel._init_weights = _init_weights_patched
        PreTrainedModel._initialize_weights = _initialize_weights_patched

    @property
    def tok_embeddings(self):
        """The token embedding table, wherever this architecture keeps it."""
        found, _ = _structure.find_embedding(self.model)
        if found is None:
            raise AttributeError(
                f"Could not locate tok_embeddings on {type(self.model).__name__}"
            )
        return found

    @tok_embeddings.setter
    def tok_embeddings(self, value):
        _, path = _structure.find_embedding(self.model)
        if path is None:
            raise AttributeError("Could not locate tok_embeddings to assign")
        _structure.set_at(self.model, path, value)

    @property
    def layers(self):
        """The decoder/encoder block stack."""
        found, _ = _structure.find_layers(self.model)
        if found is None:
            raise AttributeError(
                f"Could not locate the block stack on {type(self.model).__name__}"
            )
        return found

    @layers.setter
    def layers(self, value):
        _, path = _structure.find_layers(self.model)
        if path is None:
            raise AttributeError("Could not locate layers to assign")
        _structure.set_at(self.model, path, value)

    @property
    def norm(self):
        """The final norm, or an Identity for an architecture without one."""
        if getattr(self, "_added_norm", None) is not None:
            return self._added_norm
        found, _ = _structure.find_norm(self.model)
        if found is None:
            raise AttributeError(
                f"Could not locate norm on {type(self.model).__name__}"
            )
        return found

    @norm.setter
    def norm(self, value):
        _, path = _structure.find_norm(self.model)
        if path is None:
            raise AttributeError("Could not locate norm to assign")
        _structure.set_at(self.model, path, value)

    @property
    def lm_head(self):
        """The output projection, native to the architecture or attached for it."""
        if getattr(self, "_added_lm_head", None) is not None:
            return self._added_lm_head
        found, _ = _structure.find_head(self.model)
        if found is None:
            raise AttributeError(
                f"Could not locate lm_head on {type(self.model).__name__}"
            )
        return found

    @lm_head.setter
    def lm_head(self, value):
        _, path = _structure.find_head(self.model)
        if path is None:
            raise AttributeError("Could not locate lm_head to assign")
        _structure.set_at(self.model, path, value)

    @property
    def rotary_emb(self):
        """Rotary embedding module, or None for a learned-position model."""
        found, _ = _structure.find_rotary(self.model)
        return found

    @rotary_emb.setter
    def rotary_emb(self, value):
        _, path = _structure.find_rotary(self.model)
        if path is None:
            raise AttributeError("This architecture has no rotary_emb to assign")
        _structure.set_at(self.model, path, value)

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build the attention mask (when positions are present), CP-shard, return."""
        del max_num_documents, max_context_length
        # Function-local import avoids a circular import.
        from torchtitan.distributed.context_parallel.api import (
            prepare_context_parallel_input,
        )

        batch: dict[str, Any] = dict(input_dict)
        batch.pop("padding_mask", None)
        if "attention_masks" not in batch:
            positions = batch.get("positions")
            if positions is not None:
                masks = self.get_attention_masks(positions=positions)
                if masks is not None:
                    batch["attention_masks"] = masks

        if parallel_dims.cp_enabled:
            batch = prepare_context_parallel_input(
                batch,
                None,
                parallel_dims.get_mesh("cp"),
                parallelism.context_parallel_load_balancer,
                parallelism.context_parallel_ptrr_mask_key,
            )
        if parallelism.spmd_backend == "spmd_types":
            from torchtitan.distributed.spmd_types import annotate_input_spmd_types
            from torchtitan.models.common.decoder_sharding import decoder_input_sharding

            input_sharding = decoder_input_sharding()
            # DSA attention masks are dense tensors but are not decoder inputs;
            # preserve the old trainer behavior by annotating only declared names.
            annotated = annotate_input_spmd_types(
                parallel_dims,
                {name: batch[name] for name in input_sharding if name in batch},
                input_sharding,
            )
            batch.update(annotated)
        inputs = batch.pop("input")
        labels = batch.pop("labels")
        return inputs, labels, batch

    def get_attention_masks(self, positions: torch.Tensor):
        """Build a flex BlockMask (causal or document-causal).

        ``forward`` (or the trainer under CP) calls this and passes the result
        through as the HF ``attention_mask``. ``attn_mask_type`` selects the
        mask: "block_causal" is causal AND same-document, so packed samples
        don't attend across boundaries (which ``is_causal`` alone cannot
        express); "causal" is plain causal (flex with no mask would be full
        attention, so a causal BlockMask is still required).
        """

        # A flex BlockMask is only consumable by the flex attention path, and
        # flex has no TPU lowering -- building one here is what puts an
        # untranslatable gather/concatenate into the graph. This lane runs
        # sdpa or the splash kernel, which build their own causal masking, so
        # only produce a BlockMask when flex is actually the implementation.
        if getattr(self.model.config, "_attn_implementation", "") != _ATTN_IMPLEMENTATION:
            return None
        if _uses_dsa(self.model.config):
            # DSA models cannot consume a flex BlockMask: their indexer and main
            # attention treat the mask as a dense additive tensor (see
            # ``_uses_dsa``). Return a dense 4D additive causal mask so the DSA
            # modeling code works (its ``create_causal_mask`` returns an
            # already-4D mask as-is) and flex applies the model-built dense mask
            # as a ``score_mask``.
            return self._build_dense_attention_mask(positions)
        if getattr(self.model.config, "attn_mask_type", "causal") == "block_causal":
            mask_mod = and_masks(
                get_causal_mask_mod(),
                get_document_mask_mod(positions),
            )
        else:
            mask_mod = get_causal_mask_mod()
        num_tokens = positions.shape[0]
        return create_attention_mask(
            mask_mod,
            1,
            None,
            num_tokens,
            num_tokens,
            device=positions.device,
            BLOCK_SIZE=128,
            separate_full_blocks=not is_in_batch_invariant_mode(),
        )

    def _build_dense_attention_mask(self, positions: torch.Tensor) -> torch.Tensor:
        """Dense 4D additive attention mask for DSA models under flex.

        DSA (DeepSeek sparse attention) models cannot use a flex BlockMask (see
        ``_uses_dsa``); they need a dense additive tensor mask. Build a
        ``[1, 1, num_tokens, num_tokens]`` mask with ``0.0`` on allowed positions and ``-inf``
        elsewhere. "causal" allows key ``j <= query i``; "block_causal"
        additionally requires same-document (positions reset to 0 at each packed
        sample boundary), mirroring ``get_causal_mask_mod`` /
        ``get_document_mask_mod``.
        """
        num_tokens = positions.shape[0]
        device = positions.device
        dtype = self.tok_embeddings.weight.dtype
        idx = torch.arange(num_tokens, device=device)
        # Query i may attend to key j when j <= i.
        allowed = idx[:, None] >= idx[None, :]
        if getattr(self.model.config, "attn_mask_type", "causal") == "block_causal":
            doc_ids = torch.cumsum((positions == 0).int(), dim=0) - 1
            same_doc = doc_ids[:, None] == doc_ids[None, :]
            allowed = allowed & same_doc
        mask = torch.zeros((num_tokens, num_tokens), device=device, dtype=dtype)
        mask.masked_fill_(~allowed, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)

    def _check_position_capacity(self, seq_len: int) -> None:
        """Refuse a sequence the learned position table cannot address.

        An out-of-range embedding gather raises on CPU but is silently clamped
        on TPU, so overrunning the table costs no error and shows up only as
        numerics wrong by an order of magnitude. RoPE models report no capacity
        and are not checked.
        """
        cap = getattr(self, "_position_capacity", None)
        if cap is not None and seq_len > cap:
            raise ValueError(
                f"{type(self.model).__name__} positions at most {cap} tokens "
                f"but was given seq_len={seq_len}. Lower the config's seq_len "
                f"to {cap} or less; running past the table reads out-of-range "
                "entries, which TPU returns silently as garbage."
            )

    def forward(self, *args, **kwargs):
        positions = kwargs.pop("positions", None)
        attention_masks = kwargs.pop("attention_masks", None)
        model_args = (args[0].unsqueeze(0), *args[1:])

        if positions is not None:
            # Per-document positions (reset at packed-sample boundaries) drive
            # RoPE; the BlockMask handles cross-sample masking. Under CP these are
            # the sequence-sharded global positions (load-balancer permuted),
            # which is exactly what RoPE needs for each local shard -- a plain
            # arange would use the wrong positions.
            #
            # The BlockMask is prebuilt in ``preprocess_inputs`` and passed
            # in via ``attention_masks``.
            kwargs["position_ids"] = positions.unsqueeze(0)
        else:
            local_seq_len = args[0].shape[0]
            kwargs["position_ids"] = torch.arange(
                local_seq_len, device=args[0].device
            ).unsqueeze(0)

        if attention_masks is not None:
            # HF returns an already-4D mask / BlockMask as-is (see
            # masking_utils._preprocess_mask_arguments), so the titan-built
            # BlockMask flows straight through to the flex attention function.
            kwargs["attention_mask"] = attention_masks

        self._check_position_capacity(model_args[0].shape[-1])

        # An encoder-decoder (T5, BART, Marian, mBART) will not run without
        # something to feed its decoder. Teacher-force the same sequence, which
        # is the token objective this lane trains.
        if getattr(self.model.config, "is_encoder_decoder", False):
            kwargs.setdefault("decoder_input_ids", model_args[0])

        # T5 positions with relative bias and DistilBERT with its own
        # embedding, so neither accepts position_ids -- and neither ignores it.
        if self._forward_kwargs is not None:
            for k in [k for k in kwargs if k not in self._forward_kwargs]:
                kwargs.pop(k)

        # Resolve the backbone rather than assuming the Llama nesting: OPT
        # keeps it at model.decoder, GPT-2 at transformer, and a bare encoder
        # is its own backbone.
        # Run the whole model and take hidden states from whatever it
        # returns. Calling a resolved backbone directly only works when the
        # backbone accepts input_ids -- BERT's is its encoder, which takes
        # embeddings -- and the wrapper's output type varies by head
        # (BaseModelOutput, MaskedLMOutput, ...). Asking for hidden states is
        # the one contract every HF model honours.
        # When the head we train through is one this backend attached, the
        # hidden states are what we need -- and asking for them through
        # output_hidden_states no longer works: transformers 5.x dropped that
        # argument from the task-head forwards (checked on Roberta/Bert/GPT-2/
        # Qwen3), so the kwarg filter correctly removes it and nothing comes
        # back. base_model is HF's own handle on the trunk and takes input_ids
        # on every family checked, so call that instead.
        use_trunk = self._added_lm_head is not None
        base = getattr(self.model, "base_model", None) if use_trunk else None
        if base is not None and base is not self.model:
            output = base(*model_args, **kwargs)
        else:
            output = self.model(*model_args, **kwargs)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            states = getattr(output, "hidden_states", None)
            hidden = states[-1] if states else None
        if hidden is None and isinstance(output, (tuple, list)) and output:
            first = output[0]
            if hasattr(first, "dim") and first.dim() == 3:
                hidden = first
        if hidden is None:
            logits = getattr(output, "logits", None)
            if logits is None:
                raise AttributeError(
                    f"{type(self.model).__name__} returned neither hidden "
                    "states nor logits to train on"
                )
            return logits.squeeze(0)
        hidden_states = hidden.squeeze(0)

        if self._skip_lm_head:
            return hidden_states
        output = self.lm_head(hidden_states)

        # Numerical-test hook: when HF_BACKEND_LOGIT_DUMP=<dir> is set, append
        # this rank's per-forward logits (+ CP coordinate) to a file. Used by
        # tests/run_cp_pp_numerical.sh to compare CP-only vs CP+PP logits from a
        # shared seed checkpoint. Off (no overhead) unless the env var is set.
        _dump_dir = os.environ.get("HF_BACKEND_LOGIT_DUMP")
        if _dump_dir is not None:
            self._maybe_dump_logits(_dump_dir, output)

        return output

    def _maybe_dump_logits(self, dump_dir: str, logits: torch.Tensor) -> None:
        """Append this rank's logits (one entry per forward) for numerical tests."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        cp_coord = self.cp_mesh.get_local_rank() if self.cp_mesh is not None else 0
        recs = getattr(self, "_logit_dump_recs", None)
        if recs is None:
            recs = self._logit_dump_recs = []
        recs.append((cp_coord, logits.detach().float().cpu()))
        torch.save(recs, os.path.join(dump_dir, f"logits_rank{rank}.pt"))

    def verify_module_protocol(self) -> None:
        """Skip recursive verification for HuggingFace model internals.

        HF PreTrainedModel submodules are plain nn.Module and cannot
        conform to the Module protocol. Initialization is handled
        entirely by HF's own _init_weights mechanism.
        """
        pass

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        # This method replicates the behavior of the original PreTrainedModel.init_weights,
        # but with a custom weight initialization function that skips nn.Identity modules (when PP is enabled)

        if getattr(self.model.config, "pruned_heads", None):
            logger.info("Pruning heads as per model configuration.")
            self.model.prune_heads(self.model.config.pruned_heads)

        original_init_weights_fn = self.model._init_weights

        def selective_init(module):
            # For pipeline parallel, we need to skip nn.Identity modules
            if not isinstance(module, nn.Identity):
                original_init_weights_fn(module)
            else:
                logger.info("Skipping nn.Identity module during weight initialization.")

        self.model.apply(selective_init)

        # HF rotary embeddings compute their `inv_freq` buffer in __init__, not in
        # `_init_weights`. With meta-device init + `to_empty()`, that buffer is
        # left uninitialized (zeros), which silently disables RoPE (no positional
        # information -> near-random outputs). Recompute it from each rotary
        # module's `rope_init_fn` so positions work after materialization.
        for module in self.model.modules():
            rope_init_fn = getattr(module, "rope_init_fn", None)
            if rope_init_fn is not None and hasattr(module, "inv_freq"):
                device = module.inv_freq.device
                inv_freq, attention_scaling = rope_init_fn(module.config, device)
                module.inv_freq.copy_(
                    inv_freq.to(device=device, dtype=module.inv_freq.dtype)
                )
                module.attention_scaling = attention_scaling

        # Titan MoE modules swapped in by moe_replacement initialize themselves
        # at parallelize time (to_empty(cpu) + init_states(cpu)), but the
        # trainer parallelizes FIRST and materializes SECOND, so the to_empty()
        # that precedes this call hands those expert weights fresh,
        # uninitialized storage and the CPU values are gone. The walk above
        # cannot restore them: HF's _init_weights dispatches on module type,
        # and a stacked [E, *, *] expert weight is not an nn.Linear. Nothing
        # then writes them, so training sees expert weights that read back as
        # nan -- measured on device at the first MoE forward as finite=0.0000
        # for w1/w2/w3, while the activations and routing counts entering the
        # same call were healthy. This is the same failure the inv_freq
        # recompute above exists for: a tensor whose value is established
        # outside _init_weights and destroyed by to_empty().
        #
        # Re-run each swapped module's own init_states(), which consults the
        # param_init table keyed by the real parameter names (w1_EFD/w2_EDF/...
        # discovered dynamically by _get_expert_param_info). Must run with
        # SimpleFSDP's parametrization disabled, otherwise reads of .w1_EFD
        # return the computed tensor and the initializer writes into a
        # temporary instead of the parameter -- the same trap documented for
        # tie_weights in the TPU lane's _defer_parametrization_during_init.
        self._init_swapped_native_modules()

        # TODO(3outeille): For pipeline parallel, only tie weights if both input and output embeddings are on the same device
        # Maybe better way of handling this?
        if not isinstance(self.tok_embeddings, nn.Identity) and not isinstance(
            self.lm_head, nn.Identity
        ):
            self.model.tie_weights()

        # to_empty() hands every parameter fresh storage, which breaks the tie
        # made in __init__. HF's tie_weights only knows about its own head.
        if getattr(self, "_added_lm_head", None) is not None:
            self._added_lm_head.weight = self.tok_embeddings.weight

        self._reassert_device()

    def _init_swapped_native_modules(self) -> None:
        """Re-initialize Titan modules swapped into the HF tree after to_empty().

        Only modules that own a ``param_init`` table are touched -- that is
        exactly the set built by ``moe_replacement``, whose parameters HF's
        type-dispatched ``_init_weights`` cannot reach. ``init_states`` is
        called on the outermost such module; it recurses to its own children,
        so nested experts are covered once.
        """
        roots = [
            m for m in self.model.modules()
            if getattr(m, "_param_init", None) is not None
        ]
        if not roots:
            return

        # Deduplicate to outermost modules: a root that is a descendant of
        # another root would otherwise be initialized twice.
        inner = set()
        for r in roots:
            for child in r.modules():
                if child is not r:
                    inner.add(id(child))
        roots = [r for r in roots if id(r) not in inner]

        # Only take the guard if parametrization is actually active.
        # disable_active_parametrization() is NOT reentrant -- its finally
        # clause sets the flag back to True unconditionally rather than
        # restoring the previous value -- so nesting it inside a caller that
        # already disabled it (which the TPU lane does around this whole
        # method, see parallelize.py's _defer_parametrization_during_init)
        # re-enables parametrization on exit and the caller's subsequent
        # tie_weights() fails with "`model.embed_tokens.weight` is neither a
        # parameter, buffer, nor extra state."
        ctx = contextlib.nullcontext()
        try:
            from torchtitan.experiments.graph_trainer import simple_fsdp as _sfsdp

            if getattr(_sfsdp, "_active_parametrization", False):
                ctx = _sfsdp.disable_active_parametrization()
        except Exception:  # SimpleFSDP not in use on this lane
            pass
        with ctx:
            for r in roots:
                r.init_states(buffer_device=self._reinit_buffer_device())
        logger.info(
            "Re-initialized %d swapped native module(s) after materialization.",
            len(roots),
        )

    def _reinit_buffer_device(self) -> torch.device | None:
        """Device for buffers rebuilt during the post-materialization re-init."""
        for param in self.model.parameters():
            if param.device.type != "meta":
                return param.device
        return None

    def to_empty(self, *, device, recurse: bool = True):
        """Materialize parameters, but keep buffer values.

        torch's to_empty() runs empty_like over buffers too, discarding the
        causal masks / rope tables / normalizers that __init__ computed.
        """
        return _meta_init.to_empty_keep_buffers(self, device)

    def _reassert_device(self) -> None:
        """Pull back anything init left on the wrong device.

        Some HF initializers build a tensor without a device and assign it over
        the materialized parameter -- Pegasus and Marian do this for sinusoidal
        positions:

            out = torch.empty(n_pos, dim, dtype=self.weight.dtype)
            self.weight = nn.Parameter(out, requires_grad=False)

        torch.empty with no device= lands on CPU, so the weight to_empty() had
        placed on the accelerator is silently replaced and the first forward
        dies mixing devices. Upstream HF bug; guard until it is fixed.
        """
        device = None
        for prm in self.model.parameters():
            if prm.device.type != "cpu":
                device = prm.device
                break
        if device is None:
            return
        moved = []
        for name, mod in self.model.named_modules():
            for attr, val in list(mod._parameters.items()):
                if val is not None and val.device != device:
                    mod._parameters[attr] = nn.Parameter(
                        val.data.to(device), requires_grad=val.requires_grad
                    )
                    moved.append(f"{name}.{attr}")
            for attr, val in list(mod._buffers.items()):
                if val is not None and val.device != device:
                    mod._buffers[attr] = val.to(device)
                    moved.append(f"{name}.{attr}")
        if moved:
            logger.warning(
                f"init left {len(moved)} tensor(s) off {device}; moved back "
                f"(first: {moved[0]}). This is an upstream initializer that "
                "builds a tensor without a device."
            )

    def named_children(self):
        """A flattened view of the parts torchtitan expects.

        Parts an architecture does not have are skipped rather than yielded as
        None (GPT-2 and OPT position with learned embeddings and have no
        rotary_emb), and anything outside the five buckets is yielded too:
        nn.Module._apply walks children(), so a module omitted here is one
        .to_empty() never materializes -- it stays on meta and dies at the
        first forward with "Cannot copy out of meta tensor". GPT-2 has wpe,
        OPT has embed_positions.
        """
        seen: set[int] = set()
        for name in ("tok_embeddings", "layers", "norm", "lm_head", "rotary_emb"):
            try:
                child = getattr(self, name)
            except AttributeError:
                continue
            if child is None or id(child) in seen:
                continue
            seen.add(id(child))
            yield name, child

        # Mark the backbone covered before walking: its children are yielded
        # below, so yielding the backbone itself as well would hand the same
        # subtree to the caller twice -- and Module.parallelize() refuses to
        # visit a module more than once.
        backbone, _ = _structure.find_backbone(self.model)
        if backbone is not None:
            seen.add(id(backbone))
        for holder in (backbone, self.model):
            if holder is None:
                continue
            for child_name, child in holder.named_children():
                if child is None or id(child) in seen:
                    continue
                # Skip a module that *contains* something already yielded.
                # BERT keeps its head at cls.predictions.decoder and its token
                # table at embeddings.word_embeddings, so yielding `cls` or
                # `embeddings` here would hand the caller a second route to a
                # part already given -- and Module.parallelize() refuses to
                # visit a module twice.
                if any(id(d) in seen for d in child.modules()):
                    continue
                seen.add(id(child))
                yield child_name, child

    def __setattr__(self, name, value):
        # If a property with a setter exists for this name, use it.
        # This is to bypass the nn.Module.__setattr__ logic that
        # directly registers modules and skips property setters.
        cls = self.__class__
        if hasattr(cls, name):
            prop = getattr(cls, name)
            if isinstance(prop, property) and prop.fset is not None:
                prop.fset(self, value)
                return

        # Otherwise, fall back to the default nn.Module behavior.
        super().__setattr__(name, value)
