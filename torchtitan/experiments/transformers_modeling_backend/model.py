# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib
import math
from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.nn import init

from torch.nn.attention.flex_attention import BlockMask

from transformers import AutoConfig
from . import _meta_init, _structure
from transformers.configuration_utils import PretrainedConfig
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import AttentionInterface, PreTrainedModel

from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import ModuleDict
from torchtitan.tools.logging import logger


# Shape suffix legend (matches torchtitan/models/common/attention.py):
#   B = batch, L = sequence length, N = num heads, H = head dimension.
# HF attention layout is (B, N, L, H); native TorchTitan layout is (B, L, N, H).
class HFFlexAttention(FlexAttention):
    """FlexAttention kernel for HF models.

    Accepts Q/K/V in native TorchTitan layout (B, L, N, H) so that
    apply_cp_to_forward's K/V all-gather (dim=1, the seq dim) works correctly.
    The caller (_flex_torchtitan_attention_forward) transposes from HF layout
    before calling this module, and transposes back after.

    This subclass passes the isinstance(FlexAttention) check in
    apply_cp_to_forward, so CP wrapping works naturally.
    """

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        attention_masks: BlockMask | None = None,
        scale: float | None = None,
        enable_gqa: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        from torch.nn.attention.flex_attention import flex_attention

        # Transpose to (B, N, L, H) for the flex_attention kernel
        q_BNLH = q_BLNH.transpose(1, 2)
        k_BNLH = k_BLNH.transpose(1, 2)
        v_BNLH = v_BLNH.transpose(1, 2)
        out_BNLH = flex_attention(
            q_BNLH,
            k_BNLH,
            v_BNLH,
            block_mask=attention_masks,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        # Transpose back to (B, L, N, H)
        return out_BNLH.transpose(1, 2)


def _flex_torchtitan_attention_forward(
    module, query_BNLH, key_BNLH, value_BNLH, attention_mask, **kwargs
):
    """FlexAttention forward registered via AttentionInterface.

    Routes the kernel call through the HFFlexAttention module attached to
    each HF attention layer, so apply_cp_to_forward's K/V all-gather wrapping
    applies automatically.

    Transposes Q/K/V from HF layout (B, N, L, H) to native TorchTitan layout
    (B, L, N, H) before calling the module, so CP's dim=1 all-gather targets
    the sequence dimension correctly.
    """
    scaling = kwargs.get("scaling")
    block_mask = attention_mask if isinstance(attention_mask, BlockMask) else None

    # This forward is only registered as the "flex_torchtitan" attention impl,
    # and HFTransformerModel attaches a _flex_kernel to every self_attn when that
    # impl is selected, so the kernel module is always present here.
    flex_module = module._flex_kernel

    # HF layout (B, N, L, H) -> native layout (B, L, N, H)
    q_BLNH = query_BNLH.transpose(1, 2)
    k_BLNH = key_BNLH.transpose(1, 2)
    v_BLNH = value_BNLH.transpose(1, 2)
    out_BLNH = flex_module(
        q_BLNH, k_BLNH, v_BLNH, attention_masks=block_mask, scale=scaling
    )
    # HF attention interface expects output as (B, L, N, H) (batch, seq, heads,
    # head_dim) -- same layout sdpa_attention_forward returns. out_BLNH is
    # already in that layout, so return it directly (no transpose).
    return out_BLNH, None


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
# TorchTitan initializes an untied token embedding at unit std.
_TITAN_EMBED_INIT_STD = 1.0

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
    }
}

# Declarative list of TorchTitan-only attributes (no HF equivalent)
_TT_SPECIFIC_ATTRIBUTES = [
    "multiple_of",
    "ffn_dim_multiplier",
    "depth_init",
    "use_flex_attn",
    "attn_mask_type",
]


class HFTransformerModel(BaseModel):
    # TODO(#ISSUE): Remove after fixing PP backward to skip non-tensor inputs.
    _skip_lm_head: bool = False
    # When set, forward() returns (hidden_states, lm_head_weight.T) instead of
    # logits so a fused projection+softmax+CE kernel can consume the pair. See
    # the fused-loss branch in forward().
    _fused_loss: bool = False

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config, PretrainedConfig):
        """Configuration that bridges TorchTitan and HuggingFace Transformers.

        Uses properties to provide TorchTitan-style access while maintaining
        HuggingFace compatibility.
        """

        def __init__(
            self,
            titan_dense_config,
            # HuggingFace specific args
            attn_implementation: str = "sdpa_torchtitan",
            **kwargs,
        ):
            # Explicitly call PretrainedConfig.__init__ (not via MRO, since
            # Configurable.Config's generated __init__ doesn't chain to it)
            PretrainedConfig.__init__(
                self, attn_implementation=attn_implementation, **kwargs
            )
            # Set param_init and sharding_config before Module.Config.build()
            # accesses them. PretrainedConfig.__getattribute__ doesn't
            # recognize slots inherited from Module.Config.
            self.param_init = (
                None  # noqa: this sets Config.param_init, not Module._param_init
            )
            self.sharding_config = None

            assert titan_dense_config is not None, "titan_dense_config is required"

            # Create getter/setter dynamically for TT <-> HF attribute mappings
            self._create_getter_setter_dynamically(has_moe=False)

            self._titan_injected_model_args = {}
            self._configure_hf_attention(attn_implementation)

            self._initialize_dense_attributes(titan_dense_config)

        def build(self, **kwargs):
            """Override build() to use _replace() instead of dataclasses.replace().

            dataclasses.replace() re-invokes __init__, which is incompatible
            with the custom __init__ here (expects titan_dense_config).

            The build also runs under `params_meta_buffers_real`: HF modules
            compute real buffers in `__init__` (causal masks, rope tables,
            embedding normalizers) and a plain meta build throws all of them
            away. See _meta_init.py.
            """
            clone = self._replace()
            with _meta_init.params_meta_buffers_real():
                instance = self._owner(config=clone, **kwargs)
            if self.param_init is not None:
                instance._param_init = self.param_init
            return instance

        def _replace(self, **overrides):
            """Override to use ``copy.copy()`` instead of ``dataclasses.replace()``.

            ``dataclasses.replace()`` re-invokes ``__init__``, which is
            incompatible with the custom ``__init__`` here (it expects
            ``titan_dense_config`` and calls ``PretrainedConfig.__init__``).
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

        def _initialize_dense_attributes(self, titan_dense_config):
            """Initialize all dense model attributes."""
            # Set mapped attributes (TorchTitan <-> HuggingFace)
            for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
                if hasattr(titan_dense_config, titan_name):
                    value = getattr(titan_dense_config, titan_name)
                    setattr(self, hf_name, value)

            # Set TorchTitan-only attributes
            for attr_name in _TT_SPECIFIC_ATTRIBUTES:
                if hasattr(titan_dense_config, attr_name):
                    setattr(self, attr_name, getattr(titan_dense_config, attr_name))

            # Update passed_args
            self._titan_injected_model_args.update(titan_dense_config.__dict__)

        def _configure_hf_attention(self, attn_implementation: str):
            """Configure HuggingFace attention settings."""
            self._titan_injected_model_args["attn_implementation"] = attn_implementation
            self.attn_implementation = attn_implementation
            # HF dispatches on the private field, not this one. Without this
            # line, selecting an implementation after the Config is built is a
            # silent no-op: the attribute reads back correctly while every
            # attention layer keeps calling whatever was passed to
            # PretrainedConfig.__init__.
            self._attn_implementation = attn_implementation
            if attn_implementation == "flex_torchtitan":
                AttentionInterface._global_mapping[
                    attn_implementation
                ] = _flex_torchtitan_attention_forward
            elif attn_implementation not in AttentionInterface._global_mapping:
                logger.info(
                    f"'{attn_implementation}' not in HF AttentionInterface registry, "
                    "defaulting to sdpa_attention_forward"
                )
                AttentionInterface._global_mapping[
                    attn_implementation
                ] = sdpa_attention_forward

        def _create_getter_setter_dynamically(self, has_moe: bool):
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
            if has_moe:
                self._tt_to_hf_attribute_map.update(_TT_TO_HF_MAPPINGS["moe"])

            for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
                # A self-mapped name (e.g. vocab_size -> vocab_size) is already
                # stored under the correct name; creating an alias property would
                # make its setter call setattr() on itself and recurse forever.
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
            # Load HF config (overwrites our HF attributes)
            hf_model_config = AutoConfig.from_pretrained(
                hf_model_id,
                attn_implementation=self.attn_implementation,
                trust_remote_code=True,
            )

            # Explicitly update attributes based on mappings
            for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
                if hasattr(hf_model_config, hf_name):
                    setattr(self, titan_name, getattr(hf_model_config, hf_name))

            # Copy any other attributes that might not be in the mapping.
            # to_dict() serializes nested configs (vision_config, ...) into
            # plain dicts; re-attach the original objects instead, or the
            # submodel constructors reject them:
            #   TypeError: Parameter config in `GitVisionModel(config)` should
            #   be an instance of class `PretrainedConfig`
            for key, value in hf_model_config.to_dict().items():
                original = getattr(hf_model_config, key, None)
                if isinstance(original, PretrainedConfig):
                    value = original
                setattr(self, key, value)

            # Update our attributes with the passed args from flavors
            for key, value in self._titan_injected_model_args.items():
                if hasattr(self, key) and value is not None:
                    setattr(self, key, value)

            self.max_seq_len = training.seq_len

            self.deterministic = debug.deterministic

            # Configure HF-specific settings to match TorchTitan settings
            # TODO: false ?
            self.attention_bias = False
            self.mlp_bias = False
            self.use_cache = False
            # Keep the model's own initializer_range. This used to be forced to
            # 1.0 because TorchTitan's embedding init wants std=1.0, but the
            # field is also what HF's stock _init_weights uses as the std for
            # every Linear/Conv1D/Embedding. Architectures the Llama-like patch
            # does not match (GPT-2, BERT, XLM-R -- their classes are not named
            # <Prefix>DecoderLayer) fall back to that stock init and were being
            # built with std=1.0 instead of 0.02, which lands the initial loss
            # near 120 instead of ln(vocab). TorchTitan's own init now states
            # its 1.0 directly, at the one place it means it.
            if not hasattr(self, "initializer_range"):
                self.initializer_range = 0.02
            # FlexAttention does not support dropout (not part of PyTorch's
            # flex_attention API). Override to 0 if the model sets it.
            if hasattr(self, "attention_dropout") and self.attention_dropout > 0:
                logger.warning(
                    f"attention_dropout={self.attention_dropout} is not supported "
                    "with FlexAttention, setting to 0.0"
                )
                self.attention_dropout = 0.0

            # When dim is explicitly overridden (e.g. debugmodel), derive the
            # dependent sizes from it. Otherwise keep what AutoConfig loaded from
            # the HF config -- models like Qwen3 decouple head_dim and
            # intermediate_size from hidden_size/num_heads, so deriving them here
            # would silently build the wrong architecture.
            dim_overridden = self._titan_injected_model_args.get("dim") is not None
            if dim_overridden:
                if not hasattr(self, "inter_dim"):  # Only for llama model
                    ffn_hidden_size = 4 * self.dim
                    ffn_hidden_size = int(2 * ffn_hidden_size / 3)
                    if self.ffn_dim_multiplier is not None:
                        ffn_hidden_size = int(self.ffn_dim_multiplier * ffn_hidden_size)
                    self.intermediate_size = self.multiple_of * (
                        (ffn_hidden_size + self.multiple_of - 1) // self.multiple_of
                    )
                self.head_dim = self.dim // self.num_attention_heads
            elif not getattr(self, "head_dim", None):
                # HF config did not provide head_dim; use the standard
                # derivation when there is something to derive from. A
                # composite config (CLIP, BLIP, CLAP, SegFormer) keeps
                # hidden_size and num_attention_heads inside its text_config /
                # vision_config, so both read None at the top level and the
                # derivation is not just wrong but a TypeError. Those towers
                # carry their own head_dim; leave it unset for them.
                if self.dim is not None and getattr(self, "num_attention_heads", None):
                    self.head_dim = self.dim // self.num_attention_heads

            return self

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_dense_model_nparams_and_flops(
                model,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                head_dims=self.head_dim,
                seq_len=seq_len,
            )

    def __init__(self, config: Config):
        super().__init__()

        # Try to import the model class dynamically from the transformers library if not found in globals
        model_class_name = config.architectures[0]
        model_cls = globals().get(model_class_name, None)
        if model_cls is None:
            try:
                transformers_mod = importlib.import_module("transformers")
                model_cls = getattr(transformers_mod, model_class_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Could not find model class '{model_class_name}' in globals or transformers. "
                    f"Make sure the class is available. Original error: {e}"
                ) from e

        # Attempt to patch model weight initialization based on architecture type
        try:
            model_name_prefix = model_class_name.replace("ForCausalLM", "")
            model_module = importlib.import_module(model_cls.__module__)

            attention_cls = getattr(model_module, f"{model_name_prefix}Attention", None)
            mlp_cls = getattr(model_module, f"{model_name_prefix}MLP", None)
            decoder_layer_cls = getattr(
                model_module, f"{model_name_prefix}DecoderLayer", None
            )

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

        # Size the learned position table to actually hold max_seq_len
        # positions before building for real. The config bridges
        # max_seq_len -> max_position_embeddings, but a family that reserves
        # slots at the front of the table addresses fewer rows than it has:
        # XLM-R starts position ids at padding_idx + 1, so a 2048-row table
        # tops out at 2046 and the last two positions of every batch read out
        # of range. Measuring on a meta build costs no memory and keeps this
        # free of per-architecture knowledge.
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
        # Resolved once: forward() compares an int, so it stays traceable.
        self._position_capacity = _structure.position_capacity(self.model)
        # T5 positions with relative bias and DistilBERT with its own
        # embedding, so neither takes position_ids; passing it is a TypeError
        # rather than something they ignore. Ask the signature once.
        self._forward_kwargs = _structure.accepted_kwargs(self.model.forward)

        # A bare encoder -- sentence-transformers ships MiniLM as a plain
        # BertModel -- has no output projection at all, so there is nothing to
        # compute token logits with. Give it one, tied to the token embedding,
        # which is what any fine-tuning harness does to train such a model on a
        # token objective. This adds parameters the checkpoint does not have;
        # it is for training and measurement, not for restoring a released
        # model's own head.
        self._added_lm_head = None
        _head, _ = _structure.find_head(self.model)
        # A classification head is not an output projection: it maps the
        # backbone to num_labels, not to the vocabulary, so the token
        # objective cannot be computed from it. Treat it as absent and
        # measure the backbone with a head that produces token logits.
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
                    f"{type(self.model).__name__} has no output projection; "
                    f"attached an lm_head tied to the token embedding "
                    f"({_emb.embedding_dim} -> {_emb.num_embeddings})."
                )
            else:
                logger.warning(
                    f"{type(self.model).__name__} has neither an output "
                    "projection nor a token embedding to tie one to."
                )
        # Carried from the Config: forward() reads it off the instance.
        self._fused_loss = bool(getattr(config, "_fused_loss", False))
        self.max_seq_len = config.max_seq_len
        self.cp_mesh = None

        # Convert ModuleList to ModuleDict to preserve original indices
        # This ensures state dict keys match checkpoint keys. Resolve the
        # stack structurally -- OPT keeps it at model.decoder.layers, GPT-2
        # at transformer.h, so a hardcoded self.model.model.layers raises
        # before the first step for anything not shaped like Llama.
        _blocks, _blocks_path = _structure.find_layers(self.model)
        if _blocks is None:
            raise AttributeError(
                f"Could not locate the block stack on {type(self.model).__name__}; "
                "add its nesting to transformers_modeling_backend/_structure.py"
            )
        if isinstance(_blocks, nn.ModuleList):
            _blocks = SliceableModuleDict(
                {str(i): layer for i, layer in enumerate(_blocks)}
            )
            _structure.set_at(self.model, _blocks_path, _blocks)

        for layer in _blocks.values():
            layer.moe_enabled = False
            # Attach FlexAttention kernel module to each attention layer
            # so apply_cp_to_forward can find and wrap it for CP.
            if (
                hasattr(layer, "self_attn")
                and config.attn_implementation == "flex_torchtitan"
            ):
                layer.self_attn.register_module(
                    "_flex_kernel", HFFlexAttention(config=HFFlexAttention.Config())
                )

    @staticmethod
    def _build_with_supported_attention(model_cls, config):
        """Build the model, backing off to an attention path it accepts.

        Registering an implementation with AttentionInterface only reaches
        architectures that dispatch through it. The BERT family indexes a
        fixed class dict instead -- BERT_SELF_ATTENTION_CLASSES[
        config._attn_implementation] -- so a name it does not know is a
        KeyError in __init__, and Electra and the T5 family raise ValueError
        for the same reason. Neither is recoverable by registration, so try
        the implementations every architecture does know. Attention that is
        slower is not attention that is wrong.
        """
        wanted = getattr(config, "_attn_implementation", None)
        attempts = [wanted] + [a for a in ("sdpa", "eager") if a != wanted]
        last = None
        for attempt in attempts:
            if attempt != wanted:
                config._attn_implementation = attempt
                config.attn_implementation = attempt
            try:
                model = model_cls(config=config)
            except (ValueError, KeyError) as e:
                # Only back off when the complaint is about attention.
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

    def set_cp_mesh(self, mesh):
        self.cp_mesh = mesh

    def _patch_hf_llama_like(self, decoder_layer_cls, attention_cls, mlp_cls=None):
        """
        This patch modifies a Hugging Face Llama-like model's weight initialization to match
        the initialization scheme used in TorchTitan. This is crucial for ensuring
        bit-for-bit reproducibility when converting checkpoints between the native
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

        def _initialize_weights_patched(self, module):
            # NOTE(3outeille): monkey-patch PreTrainedModel to handle meta device initialization correctly
            # The default _initialize_weights sets _is_hf_initialized = True even on a meta device,
            # which prevents subsequent proper initialization.
            if getattr(module, "_is_hf_initialized", False):
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
                # Initialize weights and biases for q, k, v projections
                for proj_name in ["q_proj", "k_proj", "v_proj"]:
                    proj = getattr(module, proj_name)
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
        """The token embedding table."""
        found, _ = _structure.find_embedding(self.model)
        if found is None:
            raise AttributeError(
                "Could not locate tok_embeddings on "
                f"{type(self.model).__name__}; add its nesting to "
                "transformers_modeling_backend/_structure.py"
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
        """The block stack, wherever this architecture keeps it."""
        found, _ = _structure.find_layers(self.model)
        if found is None:
            raise AttributeError(
                "Could not locate layers on "
                f"{type(self.model).__name__}; add its nesting to "
                "transformers_modeling_backend/_structure.py"
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
        """The final norm."""
        found, _ = _structure.find_norm(self.model)
        if found is None:
            raise AttributeError(
                "Could not locate norm on "
                f"{type(self.model).__name__}; add its nesting to "
                "transformers_modeling_backend/_structure.py"
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
        """The output head, native to the architecture or attached for it."""
        if getattr(self, "_added_lm_head", None) is not None:
            return self._added_lm_head
        found, _ = _structure.find_head(self.model)
        if found is None:
            raise AttributeError(
                "Could not locate lm_head on "
                f"{type(self.model).__name__}; add its nesting to "
                "transformers_modeling_backend/_structure.py"
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
        """Rotary embedding module, or None for learned-position models.

        OPT and GPT-2 have no rotary embedding at all; returning None lets
        named_children skip it instead of aborting the run.
        """
        found, _ = _structure.find_rotary(self.model)
        return found

    @rotary_emb.setter
    def rotary_emb(self, value):
        _, path = _structure.find_rotary(self.model)
        if path is None:
            raise AttributeError("This architecture has no rotary_emb to assign")
        _structure.set_at(self.model, path, value)


    def _check_position_capacity(self, seq_len: int) -> None:
        """Refuse a sequence the learned position table cannot address.

        An out-of-range embedding gather raises on CPU but is silently
        clamped on TPU, so overrunning the table costs no error and shows up
        only as numerics that are wrong by an order of magnitude. Models that
        position with RoPE report no capacity and are not checked.
        """
        cap = self._position_capacity
        if cap is not None and seq_len > cap:
            raise ValueError(
                f"{type(self.model).__name__} positions at most {cap} tokens "
                f"but was given seq_len={seq_len}. Lower the config's seq_len "
                f"to {cap} or less; running past the table reads out-of-range "
                "entries, which TPU returns silently as garbage."
            )

    def forward(self, *args, positions=None, attention_masks=None, **kwargs):
        # Position the batch we were actually handed. Deriving this from
        # max_seq_len instead only agrees with the input when the two happen
        # to match, which is every config we ship and nothing that guarantees it.
        local_seq_len = (
            args[0].shape[1] if args and hasattr(args[0], "shape") else self.max_seq_len
        )
        if positions is None and self.cp_mesh is not None and self.cp_mesh.size() > 1:
            local_seq_len //= self.cp_mesh.size()

        self._check_position_capacity(local_seq_len)

        if positions is not None:
            kwargs["position_ids"] = positions.to(args[0].device)
        else:
            kwargs["position_ids"] = torch.arange(
                local_seq_len, device=args[0].device
            ).unsqueeze(0)

        if attention_masks is not None:
            kwargs["attention_mask"] = attention_masks

        if self._forward_kwargs is not None:
            dropped = [k for k in kwargs if k not in self._forward_kwargs]
            for k in dropped:
                kwargs.pop(k)

        # An encoder-decoder (T5, BART, Marian, mBART) will not run without
        # something to feed its decoder. Teacher-force the same sequence, which
        # is the token objective the rest of this lane trains on.
        if getattr(self.model.config, "is_encoder_decoder", False):
            kwargs.setdefault("decoder_input_ids", args[0])

        # Calling the backbone and head separately only works for models that
        # nest them as model.model / model.lm_head. Running the whole model and
        # taking its logits works for every HF causal LM; the split is kept only
        # for the pipeline-parallel case that needs the head detached.
        if self._skip_lm_head or self._fused_loss:
            backbone, _ = _structure.find_backbone(self.model)
            if backbone is None:
                raise AttributeError(
                    f"Cannot split the head off {type(self.model).__name__}"
                )
            hidden = backbone(*args, **kwargs).last_hidden_state
            if not self._fused_loss:
                return hidden
            # Hand the loss the operands instead of the logits. Materializing
            # them costs B*S*vocab: for Qwen3-0.6B that is 2048 x 151936 in
            # bf16, ~622 MB a step, against 1.2 GB of weights -- the vocab
            # projection dominates a model this small. A fused kernel does the
            # projection, softmax and CE without ever writing that tensor.
            head = self.lm_head
            weight = getattr(head, "weight", None)
            if weight is None:
                raise AttributeError(
                    f"fused loss needs a weight on {type(head).__name__}"
                )
            return hidden, weight.t()

        if self._added_lm_head is not None:
            # Either the architecture has no output projection, or the one it
            # has maps to labels rather than to the vocabulary. Ask for hidden
            # states and produce token logits from them; whatever task head
            # the model carries is left to run and ignored.
            kwargs["output_hidden_states"] = True
            output = self.model(*args, **kwargs)
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                states = getattr(output, "hidden_states", None)
                hidden = states[-1] if states else None
            if hidden is None:
                raise AttributeError(
                    f"{type(self.model).__name__} returned neither hidden "
                    "states nor logits to train on"
                )
            return self._added_lm_head(hidden)

        output = self.model(*args, **kwargs)
        return getattr(output, "logits", output)

    def to_empty(self, *, device, recurse: bool = True):
        """Materialize parameters, but keep buffer values.

        torch's to_empty() runs empty_like over buffers too, discarding the
        causal masks / rope tables / normalizers that __init__ computed.
        """
        return _meta_init.to_empty_keep_buffers(self, device)

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

        # TODO(3outeille): For pipeline parallel, only tie weights if both input and output embeddings are on the same device
        # Maybe better way of handling this?
        if not isinstance(self.tok_embeddings, nn.Identity) and not isinstance(
            self.lm_head, nn.Identity
        ):
            self.model.tie_weights()

        # to_empty() hands every parameter fresh storage, which breaks the tie
        # made in __init__. HF's tie_weights only knows its own head.
        if self._added_lm_head is not None:
            self._added_lm_head.weight = self.tok_embeddings.weight


    def named_children(self):
        """
        Provides a flattened view of the model's main components,
        making it compatible with TorchTitan's expectations.

        Parts that a given architecture does not have are skipped rather than
        raising: OPT and GPT-2 use learned positional embeddings and so have
        no rotary_emb at all, and some heads are tied away entirely.
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

        # Anything the architecture carries outside those five buckets must
        # still be yielded. nn.Module._apply walks children(), which is this
        # method, so a module omitted here is a module .to_empty() never
        # materializes -- it stays on meta and dies at the first forward with
        # "Cannot copy out of meta tensor". Llama has nothing outside the five;
        # GPT-2 has wpe and OPT has embed_positions.
        backbone, _ = _structure.find_backbone(self.model)
        for holder in (backbone, self.model):
            if holder is None:
                continue
            for child_name, child in holder.named_children():
                if child is None or id(child) in seen:
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
