"""OLMo 2 decoder, built on ``torchtitan.models.common``.

OLMo 2 (HF ``Olmo2ForCausalLM``) is a Llama-style dense decoder with three
deviations, each carried by one subclass here:

* Reordered norms: each sublayer's output is normalized before the residual
  add, ``x + norm(attn(x))`` and ``x + norm(mlp(x))``, with no pre-norm.
  ``Olmo2TransformerBlock`` reuses ``TransformerBlock.Config``'s
  ``attention_norm`` / ``ffn_norm`` fields for the two post-sublayer norms
  (HF ``post_attention_layernorm`` / ``post_feedforward_layernorm``).
* QK-norm over the whole projection, not per head: HF builds
  ``RMSNorm(num_heads * head_dim)`` and applies it to ``q_proj(x)`` before the
  head split. ``FlatHeadsRMSNorm`` applies an RMSNorm of that width to the
  ``[T, H, K]`` tensors ``GQAttention`` hands it, so the stock
  ``GQAttention.forward`` is kept.
* HF half-rotation RoPE, i.e. ``CosSinRoPE``; no q/k permutation is needed.
"""

from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.utils import get_nparams_and_active_nparams
from torchtitan.models.utils import quadratic_attention_flops_per_token


class FlatHeadsRMSNorm(RMSNorm):
  """RMSNorm over the flattened head dims of a ``[..., H, K]`` tensor.

  ``normalized_shape`` is ``H * K``. Under tensor parallelism the local tensor
  holds only a slice of the heads, so the statistic would be wrong;
  ``Olmo2Model.Config.update_from_config`` rejects TP.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(RMSNorm.Config):
    pass

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return super().forward(x.flatten(-2)).view_as(x)


class Olmo2Attention(GQAttention):
  """``GQAttention`` with OLMo 2's full-width q/k norms.

  q and k get separate widths (``n_heads * head_dim`` and
  ``n_kv_heads * head_dim``), so they cannot share ``GQAttention``'s single
  ``qk_norm`` config; ``qk_norm`` must stay None.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(GQAttention.Config):
    q_norm: FlatHeadsRMSNorm.Config
    k_norm: FlatHeadsRMSNorm.Config

    def __post_init__(self) -> None:
      GQAttention.Config.__post_init__(self)
      if self.qk_norm is not None:
        raise ValueError("Olmo2Attention uses q_norm / k_norm, not qk_norm.")

  def __init__(self, config: Config):
    super().__init__(config)
    self.q_norm = config.q_norm.build()
    self.k_norm = config.k_norm.build()


class Olmo2TransformerBlock(TransformerBlock):
  """OLMo 2 block: ``h = x + attention_norm(attn(x))``, ``h + ffn_norm(ffn(h))``."""

  @dataclass(kw_only=True, slots=True)
  class Config(TransformerBlock.Config):
    pass

  def __init__(self, config: Config):
    super().__init__()
    self.attention = config.attention.build()
    assert config.feed_forward is not None
    self.feed_forward = config.feed_forward.build()
    self.attention_norm = config.attention_norm.build()
    self.ffn_norm = config.ffn_norm.build()

  def forward(
      self,
      x: torch.Tensor,
      attention_masks: AttentionMasksType | None,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
  ):
    h = x + self.attention_norm(
        self.attention(x, attention_masks, positions, segment_ids)
    )
    return h + self.ffn_norm(self.feed_forward(h))


class Olmo2Model(Decoder):
  """OLMo 2 decoder; the stock ``Decoder.forward`` (homogeneous layers)."""

  @dataclass(kw_only=True, slots=True)
  class Config(Decoder.Config):
    dim: int = 2048
    vocab_size: int = 100352

    def update_from_config(self, *, config, **kwargs) -> None:
      """Rejects what OLMo 2 does not support on TPU, then the base update.

      Runs before either parallelize path (eager or GraphTrainer), so both
      reuse the llama3 parallelization unchanged. Context parallelism needs no
      check here: config parsing already rejects it, since OLMo 2's inner
      attention is not a ``CPInnerAttention``.
      """
      parallelism = config.parallelism
      # The q/k norms span all heads (FlatHeadsRMSNorm); a TP shard holds only
      # some of them, so its local statistic would be wrong.
      if parallelism.tensor_parallel_degree > 1:
        raise NotImplementedError(
            "Tensor parallelism is not supported for OLMo 2 on TPU: its q/k"
            " norms span all heads."
        )
      if parallelism.pipeline_parallel_degree > 1:
        raise NotImplementedError(
            "Pipeline parallelism is not supported for OLMo 2 on TPU yet."
        )
      if parallelism.spmd_backend != "partial_dtensor":
        raise NotImplementedError(
            "OLMo 2 on TPU supports only parallelism.spmd_backend=partial_dtensor."
        )
      Decoder.Config.update_from_config(self, config=config, **kwargs)

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
      nparams, active_nparams = get_nparams_and_active_nparams(model)
      attention_op_flops = 0
      for layer in self.layers:
        attention = layer.attention
        head_dim = attention.head_dim or attention.dim // attention.n_heads
        attention_op_flops += quadratic_attention_flops_per_token(
            num_heads=attention.n_heads,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            seq_len=seq_len,
        )
      return nparams, 6 * active_nparams + attention_op_flops
