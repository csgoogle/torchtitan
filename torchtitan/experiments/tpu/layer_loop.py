"""Lets a TPU recipe replace the decoder layer loop, without touching ``Decoder``.

``Decoder.forward`` loops over the layers inline, so running them any other way
-- as one ``scan``, or with a gather or an offload around each layer -- means
owning that loop. :class:`TpuLayerLoop` is ``model.layers`` as a container that
runs its own layers, and :func:`with_tpu_layer_loop` puts the model on a
subclass whose ``forward`` calls it. Recipes that skip the rewrite are
untouched. Design: google-pytorch/torchtitan#380.
"""

import functools
from dataclasses import dataclass

import torch

from torchtitan.config import derive
from torchtitan.models.common.attention import (
    AttentionMasksType,
    segment_ids_from_positions,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.protocols.module import ModuleDict


__all__ = ["TpuLayerLoop", "runs_layer_loop", "with_tpu_layer_loop"]


class TpuLayerLoop(ModuleDict):
    """``model.layers``, as a container that runs its own layers.

    Subclass to run them differently. Still a ``ModuleDict``, and the layer
    modules inside are the same objects, so FQNs and the optimizer are
    unaffected.
    """

    def forward(self, h: torch.Tensor, *args) -> torch.Tensor:
        for layer in self.values():
            h = layer(h, *args)
        return h


def _forward(
    self,
    tokens: torch.Tensor,
    positions: torch.Tensor | None = None,
    attention_masks: AttentionMasksType | None = None,
):
    """Runs the decoder: embeddings, the layers, the final norm, the LM head.

    A copy of ``Decoder.forward`` (``models/common/decoder.py``) that hands the
    layers to ``model.layers`` when that container runs them itself, and
    otherwise loops as upstream does. Only that hunk differs, so upstream's
    inline comments stay there, and ``tests/test_layer_loop.py`` diffs the two
    ASTs to catch an upstream edit.
    """
    h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

    segment_ids = (
        segment_ids_from_positions(positions) if positions is not None else None
    )

    # The only change from `Decoder.forward`: let the container run the layers.
    # The `else` is upstream's loop, kept for the plain, empty `ModuleDict` a
    # layer-less pipeline stage gets -- that one has no `forward`.
    #
    # A TpuLayerLoop itself cannot be split across pipeline stages.
    # TODO: support pipeline parallelism.
    if isinstance(self.layers, TpuLayerLoop):
        h = self.layers(h, attention_masks, positions, segment_ids)
    else:
        for layer in self.layers.values():
            h = layer(h, attention_masks, positions, segment_ids)

    h = self.norm(h) if self.norm is not None else h

    if self._skip_lm_head:
        return h
    output = self.lm_head(h) if self.lm_head is not None else h
    return output


def runs_layer_loop(model: torch.nn.Module) -> bool:
    """Whether ``model`` hands its layers to ``model.layers`` to run."""
    return type(model).forward is _forward


@functools.cache
def _tpu_model_class(config_cls: type) -> type:
    """Builds the ``Tpu<Model>`` subclass whose ``forward`` is ``_forward``.

    ``config_cls`` is a model's nested ``Config``, and ``Config._owner`` the
    model class that declares it -- the class to subclass, and the one
    ``build()`` instantiates, so the subclass needs a ``Config`` of its own.

    Written with ``type`` because neither base is known here: the model class
    differs per family and per trainer, and the ``Config`` base has to be the
    one the recipe ended up with (qwen3 substitutes its own, for FLOP
    accounting). One class per config class, cached.
    """
    model_cls = config_cls._owner
    if model_cls.forward is _forward:
        return model_cls  # already rewritten
    if not issubclass(model_cls, Decoder) or model_cls.forward is not Decoder.forward:
        raise TypeError(
            f"{model_cls.__name__} is not a Decoder running the stock forward"
        )

    name = f"Tpu{model_cls.__name__}"
    tpu_config_cls = dataclass(kw_only=True, slots=True)(
        type("Config", (config_cls,), {"__qualname__": f"{name}.Config"})
    )
    return type(name, (model_cls,), {"forward": _forward, "Config": tpu_config_cls})


def with_tpu_layer_loop(model_config: Decoder.Config) -> Decoder.Config:
    """Puts ``model_config`` on a model that runs its layers container.

    Every other field is carried over unchanged, and the model still loops
    exactly as ``Decoder`` did until something puts a :class:`TpuLayerLoop` in
    ``model.layers``.

    Applied by ``to_tpu_graph_trainer_config`` (``graph_trainer/config.py``)
    when stacked or scanned layers are on, and applied last: any rewrite that
    rebuilds the root config from a fixed class undoes this one, which is also
    why it cannot go on a recipe -- ``experiments/tpu/train.py`` rebuilds
    ``model_spec`` from the graph-trainer registry.
    """
    return derive(model_config, _tpu_model_class(type(model_config)).Config)
