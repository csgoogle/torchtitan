import re
from typing import Any

import torch

from torchtitan.experiments.tpu.examples.gemma4.model import Gemma4Model
from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

# Non-text towers of the multimodal checkpoint, with any leading "model." removed.
_IGNORED_HF_PREFIXES = (
    "vision_tower.",
    "embed_vision.",
    "audio_tower.",
    "embed_audio.",
    "multi_modal_projector.",
)


def _normalize_hf_key(key: str) -> str | None:
    """Normalize text-only HF keys (``model.layers.*``) to ``model.language_model.layers.*``.

    Returns None for keys of the non-text towers, which are ignored.
    """
    if key.startswith("model.language_model."):
        return key
    bare_key = key.removeprefix("model.")
    if bare_key.startswith(_IGNORED_HF_PREFIXES):
        return None
    if key.startswith("model."):
        return "model.language_model." + bare_key
    return key


class Gemma4StateDictAdapter(StateDictAdapter):
    """Converts state dicts between TorchTitan Gemma4Model and HuggingFace Gemma4 format."""

    def __init__(
        self, model_config: Gemma4Model.Config, hf_assets_path: str | None
    ) -> None:
        super().__init__(model_config, hf_assets_path)
        for idx, layer in enumerate(model_config.layers):
            if isinstance(layer.attention.qkv_linear, FusedQKVLinear.Config):
                raise NotImplementedError(
                    f"Gemma4StateDictAdapter does not support FusedQKVLinear "
                    f"(layer {idx} has use_fused_qkv=True)."
                )

        self.from_hf_map = {
            "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention module
            "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.language_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.language_model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.language_model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            # Transformer layer norms & scalar
            "model.language_model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.language_model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_layernorm.weight",
            "model.language_model.layers.{}.pre_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.language_model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.post_feedforward_layernorm.weight",
            "model.language_model.layers.{}.layer_scalar": "layers.{}.layer_scalar",
            # Dense MLP
            "model.language_model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.language_model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.language_model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # MoE norms, router, and down_proj
            "model.language_model.layers.{}.pre_feedforward_layernorm_2.weight": "layers.{}.moe.pre_feedforward_layernorm_2.weight",
            "model.language_model.layers.{}.post_feedforward_layernorm_1.weight": "layers.{}.post_feedforward_layernorm_1.weight",
            "model.language_model.layers.{}.post_feedforward_layernorm_2.weight": "layers.{}.post_feedforward_layernorm_2.weight",
            "model.language_model.layers.{}.router.proj.weight": "layers.{}.moe.router.gate.weight",
            "model.language_model.layers.{}.router.scale": "layers.{}.moe.router.scale",
            "model.language_model.layers.{}.router.per_expert_scale": "layers.{}.moe.router.per_expert_scale",
            "model.language_model.layers.{}.experts.down_proj": "layers.{}.moe.routed_experts.inner_experts.w2_EDF",
            # Final norm & output head
            "model.language_model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from a TorchTitan Gemma4 state dict to a HuggingFace state dict."""
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict: dict[str, Any] = {}
        pending_w1: dict[str, torch.Tensor] = {}
        pending_w3: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            if "layers" in key:
                match = re.search(r"\d+", key)
                if match is None:
                    continue
                layer_num = match.group(0)
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)

                if (
                    abstract_key
                    == "layers.{}.moe.routed_experts.inner_experts.w1_EFD"
                ):
                    pending_w1[layer_num] = value
                    if layer_num in pending_w3:
                        hf_key = f"model.language_model.layers.{layer_num}.experts.gate_up_proj"
                        hf_state_dict[hf_key] = torch.cat(
                            [pending_w1.pop(layer_num), pending_w3.pop(layer_num)],
                            dim=1,
                        )
                    continue

                if (
                    abstract_key
                    == "layers.{}.moe.routed_experts.inner_experts.w3_EFD"
                ):
                    pending_w3[layer_num] = value
                    if layer_num in pending_w1:
                        hf_key = f"model.language_model.layers.{layer_num}.experts.gate_up_proj"
                        hf_state_dict[hf_key] = torch.cat(
                            [pending_w1.pop(layer_num), pending_w3.pop(layer_num)],
                            dim=1,
                        )
                    continue

                if abstract_key in to_hf_map:
                    new_key = to_hf_map[abstract_key].format(layer_num)
                    hf_state_dict[new_key] = value
            else:
                if key == "lm_head.weight" and self.model_config.enable_weight_tying:
                    continue
                if key in to_hf_map:
                    hf_state_dict[to_hf_map[key]] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from a HuggingFace Gemma4 state dict to a TorchTitan state dict."""
        state_dict: dict[str, Any] = {}

        embed_weight = (
            hf_state_dict.get("model.language_model.embed_tokens.weight")
            if "model.language_model.embed_tokens.weight" in hf_state_dict
            else hf_state_dict.get("model.embed_tokens.weight")
        )
        if (
            self.model_config.enable_weight_tying
            and "lm_head.weight" not in hf_state_dict
            and embed_weight is not None
        ):
            state_dict["lm_head.weight"] = embed_weight

        for raw_key, value in hf_state_dict.items():
            key = _normalize_hf_key(raw_key)
            if key is None:
                continue

            if "layers" in key:
                match = re.search(r"\d+", key)
                if match is None:
                    continue
                layer_num = match.group(0)
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)

                if (
                    abstract_key
                    == "model.language_model.layers.{}.experts.gate_up_proj"
                ):
                    w1_EFD, w3_EFD = torch.chunk(value, 2, dim=1)
                    state_dict[
                        f"layers.{layer_num}.moe.routed_experts.inner_experts.w1_EFD"
                    ] = w1_EFD.contiguous()
                    state_dict[
                        f"layers.{layer_num}.moe.routed_experts.inner_experts.w3_EFD"
                    ] = w3_EFD.contiguous()
                    continue

                if abstract_key in self.from_hf_map:
                    new_key = self.from_hf_map[abstract_key].format(layer_num)
                    state_dict[new_key] = value
            else:
                if key in self.from_hf_map:
                    state_dict[self.from_hf_map[key]] = value

        return state_dict
