"""HF ``Olmo2ForCausalLM`` <-> TorchTitan ``Olmo2Model`` weight mapping.

The mapping is key renaming only; every tensor keeps its HF layout:

  HF                                                 TorchTitan
  model.embed_tokens.weight                          tok_embeddings.weight
  model.layers.{i}.self_attn.q_proj.weight           layers.{i}.attention.qkv_linear.wq.weight
  model.layers.{i}.self_attn.k_proj.weight           layers.{i}.attention.qkv_linear.wk.weight
  model.layers.{i}.self_attn.v_proj.weight           layers.{i}.attention.qkv_linear.wv.weight
  model.layers.{i}.self_attn.o_proj.weight           layers.{i}.attention.wo.weight
  model.layers.{i}.self_attn.q_norm.weight           layers.{i}.attention.q_norm.weight   [H*K]
  model.layers.{i}.self_attn.k_norm.weight           layers.{i}.attention.k_norm.weight   [Hkv*K]
  model.layers.{i}.mlp.gate_proj.weight              layers.{i}.feed_forward.w1.weight
  model.layers.{i}.mlp.up_proj.weight                layers.{i}.feed_forward.w3.weight
  model.layers.{i}.mlp.down_proj.weight              layers.{i}.feed_forward.w2.weight
  model.layers.{i}.post_attention_layernorm.weight   layers.{i}.attention_norm.weight
  model.layers.{i}.post_feedforward_layernorm.weight layers.{i}.ffn_norm.weight
  model.norm.weight                                  norm.weight
  lm_head.weight                                     lm_head.weight

Decisions:

* No q/k permutation: both sides use the half-rotation RoPE layout
  (``CosSinRoPE``), and the full-width q/k norm weights are indexed the same
  way as the projection rows they scale.
* ``attention_norm`` / ``ffn_norm`` hold OLMo 2's post-sublayer norms (see
  ``model.py``); there is no ``input_layernorm`` in OLMo 2.
* ``lm_head`` is untied in every OLMo 2 checkpoint
  (``tie_word_embeddings: false``); ``enable_weight_tying`` is rejected.
* ``rotary_emb.inv_freq`` buffers are dropped.
"""

import re
from typing import Any

from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import Olmo2Model

_LAYER_KEY = re.compile(r"(\d+)")


class Olmo2StateDictAdapter(StateDictAdapter):
  """Renames keys between HF OLMo 2 and ``Olmo2Model``; no layout changes."""

  def __init__(
      self, model_config: Olmo2Model.Config, hf_assets_path: str | None
  ):
    super().__init__(model_config, hf_assets_path)
    if model_config.enable_weight_tying:
      raise ValueError("OLMo 2 checkpoints do not tie lm_head to embeddings.")
    self.from_hf_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
        "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

  @staticmethod
  def _rename(key: str, mapping: dict[str, str | None]) -> str | None:
    """Maps ``key`` through ``mapping``; raises on a key it does not know."""
    abstract_key = _LAYER_KEY.sub("{}", key, count=1)
    if abstract_key not in mapping:
      raise KeyError(f"No OLMo 2 mapping for {key!r}")
    new_key = mapping[abstract_key]
    if new_key is None or "{}" not in abstract_key:
      return new_key
    return new_key.format(_LAYER_KEY.search(key).group(0))

  def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
    to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
    return {self._rename(k, to_hf_map): v for k, v in state_dict.items()}

  def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
    self._validate_hf_rope_config(CosSinRoPE.Config)
    state_dict = {}
    for key, value in hf_state_dict.items():
      new_key = self._rename(key, self.from_hf_map)
      if new_key is not None:
        state_dict[new_key] = value
    return state_dict
