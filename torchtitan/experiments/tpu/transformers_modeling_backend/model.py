"""TPU overrides of the shared HF backend model.

`TPUHFTransformerModel` subclasses the backend's `HFTransformerModel` and
changes only what this lane needs; the backend itself is used unmodified.
Its nested Config subclasses the backend's Config, so building it builds the
TPU model directly.
"""

import dataclasses

import torch

from torchtitan.experiments.graph_trainer.simple_fsdp import (
    disable_active_parametrization,
)
from torchtitan.experiments.tpu.loss import (
    _legacy_pallas_loss_config,
    PallasCrossEntropyLoss,
)
from torchtitan.experiments.tpu.workarounds import _extract_splash_kwargs
from torchtitan.experiments.transformers_modeling_backend.model import (
    HFTransformerModel,
)


class TPUHFTransformerModel(HFTransformerModel):
  """HFTransformerModel with the TPU lane's overrides.

  - Attention implementation. The backend configures flex unconditionally,
    which has no TPU lowering. This lane runs "sdpa", or "splash_tpu" (this
    package's attention module) when the job's
    ``splash_attention_kernel.use_splash_attention_kernel`` is set.
  - No flex BlockMask. The backend builds one whenever the batch carries
    positions, but its construction lowers to a 5-D i64 concatenate XLA
    rejects. sdpa and the splash kernel do their own causal masking.
  - Fused loss. When ``loss_kernel.use_loss_kernel`` is set (or the job's
    loss is `PallasCrossEntropyLoss`), forward returns
    ``(hidden_states, lm_head.weight.T)`` instead of logits, so the tokens x
    vocab logits tensor is never built. The backend's own
    ``_skip_lm_head`` stops forward at the hidden states; this adds the
    weight.
  - Weight tying under SimpleFSDP. The trainer calls ``init_states`` after
    parallelizing, when SimpleFSDP's parametrization is installed; HF's
    ``tie_weights`` would then read ``.weight`` as a computed plain tensor
    and fail to assign it as a Parameter. Run init with the parametrization
    disabled, as torchtitan's own SimpleFSDP models do.
  - Rotary buffers. Recomputed after init; see `_recompute_rope_buffers`.
  """

  # init=False: keep the backend's hand-written __init__ rather than a
  # generated one. Methods call the base explicitly because slots=True
  # rebuilds the class, which breaks zero-argument super().
  @dataclasses.dataclass(kw_only=True, slots=True, init=False)
  class Config(HFTransformerModel.Config):

    # Needed even as a pass-through: without an __init__ in the class body,
    # transformers' PretrainedConfig machinery installs a generated one that
    # demands every dataclass field ("Missing required field").
    def __init__(self, model_config, **kwargs):
      HFTransformerModel.Config.__init__(self, model_config, **kwargs)

    def update_from_config(self, *, config=None, **kwargs):
      HFTransformerModel.Config.update_from_config(
          self, config=config, **kwargs
      )
      # Read the lane's kernel switches off the final job config (after CLI
      # overrides), then clear them: the TPU entry point's
      # workarounds.apply_patches also acts on them, with patches written
      # for torchtitan's own modules -- the splash module swap finds no
      # attention module in an HF model and raises, and the lm_head wrapper
      # breaks HF's weight tying. Consequence: the logged job config shows
      # both switches as False even when they were used.
      sa = config.splash_attention_kernel
      if sa.use_tokamax_splash_attention_kernel:
        raise ValueError(
            "use_tokamax_splash_attention_kernel is not supported on the HF"
            " backend lane; use use_splash_attention_kernel."
        )
      # HF picks the attention function from `_attn_implementation`; the
      # backend's update_from_config never rewrites underscore fields.
      self._attn_implementation = (
          "splash_tpu" if sa.use_splash_attention_kernel else "sdpa"
      )
      sa.use_splash_attention_kernel = False
      # The HF model keeps this Config as its `config`, so attention modules
      # reach the splash options as `module.config._tpu_splash_kwargs`.
      self._tpu_splash_kwargs = _extract_splash_kwargs(sa)

      if config.loss_kernel.use_loss_kernel:
        config.loss = _legacy_pallas_loss_config(config)
        config.loss_kernel.use_loss_kernel = False
      self._tpu_fused_loss = isinstance(
          config.loss, PallasCrossEntropyLoss.Config
      )

  def __init__(self, config: Config):
    super().__init__(config)
    self._tpu_fused_loss = getattr(config, "_tpu_fused_loss", False)
    self._skip_lm_head = self._tpu_fused_loss

  def get_attention_masks(self, positions: torch.Tensor):
    del positions
    return None

  def forward(self, *args, **kwargs):
    output = super().forward(*args, **kwargs)
    if self._tpu_fused_loss:
      return output, self.lm_head.weight.t()
    return output

  def init_states(self, *args, **kwargs):
    with disable_active_parametrization():
      super().init_states(*args, **kwargs)
    self._recompute_rope_buffers()

  @torch.no_grad()
  def _recompute_rope_buffers(self) -> None:
    """Recompute rotary ``inv_freq`` after the meta build + ``to_empty()``.

    HF computes ``inv_freq`` in the rotary module's ``__init__``, which a meta
    build skips, and ``to_empty()`` leaves it as uninitialized memory -- a NaN
    loss from step 1. The backend's ``init_states`` recomputes it only for
    modules exposing ``rope_init_fn``, but transformers 5.x keeps that as a
    local in ``__init__`` for most models (Llama, Qwen3, ...), so the backend's
    loop matches nothing. Resolve the init function the way ``__init__`` does.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # pylint: disable=g-import-not-at-top

    for module in self.model.modules():
      rope_type = getattr(module, "rope_type", None)
      if rope_type is None or not hasattr(module, "inv_freq"):
        continue
      init_fn = getattr(module, "rope_init_fn", None)
      if init_fn is None:
        init_fn = (
            module.compute_default_rope_parameters
            if rope_type == "default"
            else ROPE_INIT_FUNCTIONS[rope_type]
        )
      # On CPU, as __init__ computes it, so the values match a non-meta build.
      inv_freq, module.attention_scaling = init_fn(module.config, "cpu")
      module.inv_freq.copy_(inv_freq)
      if hasattr(module, "original_inv_freq"):
        module.original_inv_freq.copy_(inv_freq)
