"""TPU overrides of the shared HF backend model.

`TPUHFTransformerModel` subclasses the backend's `HFTransformerModel` and
changes only what this lane needs; the backend itself is used unmodified.
Its nested Config subclasses the backend's Config, so building it builds the
TPU model directly.
"""

import dataclasses
import sys

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

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
    SliceableModuleDict,
)
from torchtitan.tools.logging import logger

from . import _meta_init, _structure


def _checkpoint_initializer_range(hf_model: str) -> float:
  """The initializer_range the checkpoint's own config states."""
  d, _ = PretrainedConfig.get_config_dict(hf_model)
  for holder in (d, d.get("text_config") or {}):
    v = holder.get("initializer_range")
    if isinstance(v, (int, float)):
      return float(v)
  try:
    from transformers import CONFIG_MAPPING  # pylint: disable=g-import-not-at-top

    return float(CONFIG_MAPPING[d["model_type"]]().initializer_range)
  except Exception:  # pylint: disable=broad-except
    return 0.02


def _gets_llama_like_init(model: nn.Module) -> bool:
  """Whether the backend's Llama-like init patch applies to this model.

  Mirrors the backend's own test: the model's module defines
  <Prefix>Attention and <Prefix>DecoderLayer (Prefix optionally + "Text").
  """
  prefix = type(model).__name__.replace("ForCausalLM", "")
  module = sys.modules.get(type(model).__module__)
  return any(
      getattr(module, f"{p}Attention", None) is not None
      and getattr(module, f"{p}DecoderLayer", None) is not None
      for p in (prefix, f"{prefix}Text")
  )


class TPUHFTransformerModel(HFTransformerModel):
  """HFTransformerModel with the TPU lane's overrides.

  - Attention implementation. The backend configures flex unconditionally,
    which has no TPU lowering. This lane runs sdpa (eager for classes without
    sdpa support), or "splash_tpu" (this package's attention module) when
    the job's ``splash_attention_kernel.use_splash_attention_kernel`` is set.
  - No flex BlockMask. The backend builds one whenever the batch carries
    positions, but its construction lowers to a 5-D i64 concatenate XLA
    rejects. sdpa and the splash kernel do their own causal masking.
  - Fused loss. When ``loss_kernel.use_loss_kernel`` is set (or the job's
    loss is `PallasCrossEntropyLoss`), forward returns
    ``(hidden_states, lm_head.weight.T)`` instead of logits, so the tokens x
    vocab logits tensor is never built.
  - Any architecture, not only the Llama nesting. The backend reaches every
    part as ``model.model.<part>``; this class resolves the parts
    structurally (_structure.py), so encoders, classifier heads, GPT-2 and
    encoder-decoders build and train too. A model with no usable output
    projection gets an lm_head tied to its token embedding, one with no final
    norm an Identity.
  - Buffers survive the meta build. HF computes buffers in ``__init__``
    (rotary inv_freq, position ids, causal masks) that a meta build and
    ``to_empty()`` would leave as uninitialized memory; see _meta_init.py.
  - Weight tying under SimpleFSDP. The trainer calls ``init_states`` after
    parallelizing, when SimpleFSDP's parametrization is installed; HF's
    ``tie_weights`` would then read ``.weight`` as a computed plain tensor
    and fail to assign it as a Parameter. Run init with the parametrization
    disabled, as torchtitan's own SimpleFSDP models do.
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
      # backend's update_from_config never rewrites underscore fields. None
      # rather than "sdpa": HF then resolves sdpa where the class supports it
      # and eager where it does not (T5, DeBERTa-v2, LayoutLM, GIT, ...),
      # while an explicit "sdpa" makes those classes raise.
      self._attn_implementation = (
          "splash_tpu" if sa.use_splash_attention_kernel else None
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

      # The backend sets initializer_range=1.0, the std its Llama-like init
      # patch wants for the embedding. HF's stock _init_weights, which every
      # other architecture runs, reads the same field as the std for every
      # Linear/Embedding, so GPT-2/BERT/XLM-R would train from std=1.0
      # (cross entropy near 120 where ln(vocab) is ~10.8). Keep the
      # checkpoint's own value to restore for those; see __init__.
      self._tpu_hf_initializer_range = _checkpoint_initializer_range(
          config.hf_model
      )

  def __init__(self, config: Config):
    _grow_position_table(config)
    # HF modules compute real buffers in __init__ (rope tables, position ids,
    # causal masks) and a plain meta build throws all of them away.
    with _meta_init.params_meta_buffers_real():
      try:
        super().__init__(config)
      except AttributeError:
        # The backend's __init__ ends by walking model.model.layers, the
        # Llama nesting. For any other layout that is the one step that
        # fails, after the HF model is built; finish it structurally.
        if "model" not in self._modules or (
            _structure.find_layers(self.model)[1] == ("model", "layers")
        ):
          raise
        self._finish_init_structurally(config)

    attn = self.model.config._attn_implementation
    if attn not in ("sdpa", "splash_tpu"):
      logger.warning(
          "%s has no sdpa support; attention runs HF's %r implementation.",
          type(self.model).__name__, attn,
      )
    if not _gets_llama_like_init(self.model):
      self.model.config.initializer_range = getattr(
          config, "_tpu_hf_initializer_range", 0.02
      )
    self._tpu_fused_loss = getattr(config, "_tpu_fused_loss", False)
    self._skip_lm_head = self._tpu_fused_loss
    self._position_capacity = _structure.position_capacity(self.model)
    # T5 positions with relative bias and DistilBERT with its own embedding,
    # so neither accepts position_ids -- and neither ignores it.
    self._forward_kwargs = _structure.accepted_kwargs(self.model.forward)

    # BERT and XLM-R normalize inside every block and end without a final
    # norm; an Identity is what "no final norm" computes.
    self._added_norm = (
        nn.Identity() if _structure.find_norm(self.model)[0] is None else None
    )
    # A bare encoder has no output projection, and a classification head maps
    # to num_labels rather than the vocabulary -- neither yields token logits.
    # Attach a head tied to the token embedding, as a fine-tuning harness
    # does to put a token objective on such a model. It adds parameters the
    # checkpoint does not carry: it is for training and measurement.
    self._added_lm_head = None
    head, _ = _structure.find_head(self.model)
    head_out = _structure.head_out_features(head)
    vocab = getattr(config, "vocab_size", None)
    if head is None or (head_out is not None and vocab and head_out != vocab):
      emb, _ = _structure.find_embedding(self.model)
      if isinstance(emb, nn.Embedding):
        self._added_lm_head = nn.Linear(
            emb.embedding_dim, emb.num_embeddings, bias=False
        )
        self._added_lm_head.weight = emb.weight
        logger.info(
            "%s has no token output projection; attached an lm_head tied to"
            " the token embedding (%d -> %d).",
            type(self.model).__name__, emb.embedding_dim, emb.num_embeddings,
        )

  def _finish_init_structurally(self, config) -> None:
    """The tail of the backend's __init__, for a non-Llama layout."""
    if config.is_moe:
      raise NotImplementedError(
          "MoE models outside the Llama nesting are not supported on the TPU"
          " lane."
      )
    blocks, path = _structure.find_layers(self.model)
    if blocks is None:
      raise AttributeError(
          f"Could not locate the block stack on {type(self.model).__name__}"
      )
    # ModuleList -> ModuleDict keeps the original indices in state-dict keys.
    if isinstance(blocks, nn.ModuleList):
      blocks = SliceableModuleDict(
          {str(i): layer for i, layer in enumerate(blocks)}
      )
      _structure.set_at(self.model, path, blocks)
    for layer in blocks.values():
      layer.moe_enabled = False

  # --- Parts, wherever this architecture keeps them. -----------------------

  @property
  def tok_embeddings(self):
    found, _ = _structure.find_embedding(self.model)
    if found is None:
      raise AttributeError(
          f"Could not locate tok_embeddings on {type(self.model).__name__}"
      )
    return found

  @tok_embeddings.setter
  def tok_embeddings(self, value):
    _structure.set_at(self.model, _structure.find_embedding(self.model)[1], value)

  @property
  def layers(self):
    found, _ = _structure.find_layers(self.model)
    if found is None:
      raise AttributeError(
          f"Could not locate the block stack on {type(self.model).__name__}"
      )
    return found

  @layers.setter
  def layers(self, value):
    _structure.set_at(self.model, _structure.find_layers(self.model)[1], value)

  @property
  def norm(self):
    if getattr(self, "_added_norm", None) is not None:
      return self._added_norm
    return _structure.find_norm(self.model)[0]

  @norm.setter
  def norm(self, value):
    _structure.set_at(self.model, _structure.find_norm(self.model)[1], value)

  @property
  def lm_head(self):
    if getattr(self, "_added_lm_head", None) is not None:
      return self._added_lm_head
    return _structure.find_head(self.model)[0]

  @lm_head.setter
  def lm_head(self, value):
    if getattr(self, "_added_lm_head", None) is not None:
      self._added_lm_head = value
      return
    _structure.set_at(self.model, _structure.find_head(self.model)[1], value)

  @property
  def rotary_emb(self):
    """The rotary embedding, or None for a learned-position model."""
    return _structure.find_rotary(self.model)[0]

  @rotary_emb.setter
  def rotary_emb(self, value):
    _structure.set_at(self.model, _structure.find_rotary(self.model)[1], value)

  def named_children(self):
    """The parts torchtitan expects, then everything else once.

    Parts an architecture lacks are skipped rather than yielded as None, and
    any other child is yielded too: nn.Module._apply walks children(), so a
    module omitted here is one to_empty() never materializes (GPT-2's wpe).
    A child that contains an already-yielded part is skipped -- BERT keeps
    its head at cls.predictions.decoder -- because Module.parallelize()
    refuses to visit a module twice.
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
    backbone, _ = _structure.find_backbone(self.model)
    if backbone is not None:
      seen.add(id(backbone))
    for holder in (backbone, self.model):
      if holder is None:
        continue
      for child_name, child in holder.named_children():
        if child is None or id(child) in seen:
          continue
        if any(id(d) in seen for d in child.modules()):
          continue
        seen.add(id(child))
        yield child_name, child

  # --- Forward. -------------------------------------------------------------

  def get_attention_masks(self, positions: torch.Tensor):
    del positions
    return None

  def forward(self, *args, **kwargs):
    positions = kwargs.pop("positions", None)
    kwargs.pop("attention_masks", None)  # never built on this lane
    tokens = args[0].unsqueeze(0)
    model_args = (tokens, *args[1:])
    if positions is not None:
      kwargs["position_ids"] = positions.unsqueeze(0)
    else:
      kwargs["position_ids"] = torch.arange(
          tokens.shape[-1], device=tokens.device
      ).unsqueeze(0)

    cap = self._position_capacity
    if cap is not None and tokens.shape[-1] > cap:
      # An out-of-range embedding gather is silently clamped on TPU.
      raise ValueError(
          f"{type(self.model).__name__} positions at most {cap} tokens but"
          f" was given {tokens.shape[-1]}."
      )
    # An encoder-decoder will not run without decoder inputs: teacher-force
    # the same sequence.
    if getattr(self.model.config, "is_encoder_decoder", False):
      kwargs.setdefault("decoder_input_ids", tokens)
    if self._forward_kwargs is not None:
      kwargs = {k: v for k, v in kwargs.items() if k in self._forward_kwargs}

    # Through a head this class attached, or for the fused loss, take the
    # trunk's hidden states from base_model (HF's handle on the trunk; the
    # task-head forwards no longer accept output_hidden_states in 5.x).
    # Otherwise run the whole model and train on its own logits.
    use_trunk = self._added_lm_head is not None or self._tpu_fused_loss
    base = getattr(self.model, "base_model", None) if use_trunk else None
    if base is not None and base is not self.model:
      output = base(*model_args, **kwargs)
    else:
      output = self.model(*model_args, **kwargs)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)) and output:
      if hasattr(output[0], "dim") and output[0].dim() == 3:
        hidden = output[0]
    if hidden is None:
      logits = getattr(output, "logits", None)
      if logits is None:
        raise AttributeError(
            f"{type(self.model).__name__} returned neither hidden states nor"
            " logits to train on"
        )
      return logits.squeeze(0)
    hidden = hidden.squeeze(0)

    if self._tpu_fused_loss:
      return hidden, self.lm_head.weight.t()
    if self._skip_lm_head:
      return hidden
    return self.lm_head(hidden)

  # --- Materialization. -----------------------------------------------------

  def to_empty(self, *, device, recurse: bool = True):
    """Materialize parameters, but keep buffer values (see _meta_init.py)."""
    del recurse
    return _meta_init.to_empty_keep_buffers(self, device)

  def init_states(self, *args, **kwargs):
    with disable_active_parametrization():
      super().init_states(*args, **kwargs)
      # to_empty() gave every parameter fresh storage, breaking the tie made
      # in __init__; HF's tie_weights only knows its own head.
      if self._added_lm_head is not None:
        self._added_lm_head.weight = self.tok_embeddings.weight
    self._reassert_device()

  def _reassert_device(self) -> None:
    """Pull back anything init left on the host.

    Some HF initializers build a tensor with no device and assign it over the
    materialized parameter (Pegasus and Marian sinusoidal positions), so the
    first forward mixes devices.
    """
    device = next(
        (p.device for p in self.model.parameters() if p.device.type != "cpu"),
        None,
    )
    if device is None:
      return
    moved = 0
    for mod in self.model.modules():
      for attr, val in list(mod._parameters.items()):
        if val is not None and val.device != device:
          mod._parameters[attr] = nn.Parameter(
              val.data.to(device), requires_grad=val.requires_grad
          )
          moved += 1
      for attr, val in list(mod._buffers.items()):
        if val is not None and val.device != device:
          mod._buffers[attr] = val.to(device)
          moved += 1
    if moved:
      logger.warning("init left %d tensor(s) off %s; moved back.", moved, device)


def _grow_position_table(config) -> None:
  """Size a learned position table to address max_seq_len.

  The backend bridges max_seq_len -> max_position_embeddings, but a family
  that reserves slots at the front addresses fewer rows than it has: XLM-R
  starts position ids at padding_idx + 1, so a 2048-row table tops out at
  2046 and the last positions read out of range -- silently on TPU.
  """
  wanted = getattr(config, "max_seq_len", None)
  if not wanted:
    return
  import transformers  # pylint: disable=g-import-not-at-top

  model_cls = getattr(transformers, config.architectures[0], None)
  if model_cls is None:
    return
  try:
    with torch.device("meta"):
      probe = model_cls(config=config)
  except Exception as e:  # pylint: disable=broad-except
    logger.warning("Could not size the position table: %s", e)
    return
  cap = _structure.position_capacity(probe)
  if cap is not None and cap < wanted:
    config.max_position_embeddings += wanted - cap
    logger.info(
        "Grew the position table by %d to address %d positions.",
        wanted - cap, wanted,
    )
