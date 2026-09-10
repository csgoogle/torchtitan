# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Replaceable decoder layer loop for TPU recipes, without touching ``Decoder``.

``Decoder.forward`` runs the layers with an inlined ``for`` loop, so anything
that wants to run them differently (one ``scan`` over stacked weights, an
all-gather inside the body, host offload inside the loop) has to own the loop.
Rather than editing ``torchtitan/models/common/decoder.py``, this module
supplies:

* :class:`TpuLayerLoop` -- a ``ModuleDict`` that knows how to run its layers.
  The base class runs them in order; a TPU-specific subclass replaces
  ``forward`` (e.g. with a ``scan``) and is swapped in for ``model.layers``
  after build.  The layer modules inside are the same objects, so FQNs,
  ``state_dict`` and the optimizer are unchanged.
* :class:`TpuLayerLoopMixin` -- ``Decoder.forward`` with the loop delegated to
  a :class:`TpuLayerLoop` container when one is installed, and the upstream
  loop otherwise (so a plain ``ModuleDict`` -- including the empty one the
  pipeline splitter installs on a layer-less stage -- behaves exactly as on
  ``main``).
* :class:`TpuLayerLoopConverter` -- a ``ModelConfigConverter`` that swaps the
  model class for ``Tpu<Model> = (TpuLayerLoopMixin, <Model>)`` through the
  config system, for whatever ``Decoder`` subclass the recipe uses.  Add it to
  a recipe's ``converters=[...]`` list; nothing outside ``experiments/tpu``
  changes.

``TpuLayerLoopMixin.forward`` is a copy of ``Decoder.forward`` with one hunk
replaced.  ``tests/unit_tests/test_tpu_layer_loop.py`` pins it to upstream two
ways: behaviorally (upstream's own ``forward`` is the oracle, on the same
instance) and structurally (AST equality modulo that one hunk), so drift in
``decoder.py`` fails a test instead of going unnoticed.
"""

from dataclasses import dataclass

import torch

from torchtitan.config import derive
from torchtitan.models.common.attention import (
    AttentionMasksType,
    segment_ids_from_positions,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.module import ModuleDict


class TpuLayerLoop(ModuleDict):
    """``model.layers`` that runs its own layers.

    The base implementation is the upstream loop.  Subclass and override
    ``forward`` to run the same layers another way; the container is a
    ``ModuleDict``, so ``keys()``/``items()``/``values()``, ``state_dict`` and
    the pipeline splitter's ``del layers[name]`` keep working.
    """

    def forward(self, h: torch.Tensor, *args) -> torch.Tensor:
        for layer in self.values():
            h = layer(h, *args)
        return h


class TpuLayerLoopMixin:
    """``Decoder.forward`` with the layer loop delegated to :class:`TpuLayerLoop`.

    Everything except the ``if isinstance(self.layers, TpuLayerLoop)`` block
    is byte-for-byte upstream (see the structural test).
    """

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ):
        # positions is listed before attention_masks so AutoParallel's input_fn,
        # which returns (tokens, positions) and binds them positionally, maps
        # positions to the right parameter (it would otherwise land in the
        # attention_masks slot and break the maskless SDPA backend).
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        # Computed here, above the layers, because every layer shares
        # `positions` and each layer is its own activation-checkpoint region
        # -- computing it inside an attention module would run it once per
        # layer instead of once per step.
        segment_ids = (
            segment_ids_from_positions(positions) if positions is not None else None
        )

        # TPU: the ONLY divergence from ``Decoder.forward``.  A ``TpuLayerLoop``
        # container runs the layers itself (sequentially, or as a scan);
        # anything else gets the upstream loop, so a plain/empty ``ModuleDict``
        # (pipeline stages) behaves exactly as on main.
        if isinstance(self.layers, TpuLayerLoop):
            h = self.layers(h, attention_masks, positions, segment_ids)
        else:
            for layer in self.layers.values():
                h = layer(h, attention_masks, positions, segment_ids)

        h = self.norm(h) if self.norm is not None else h

        # _skip_lm_head is an attribute rather than a forward kwarg because PP backward
        # calls .requires_grad on all stage inputs, which fails on bool kwargs.
        # TODO: fix PP backward upstream to skip non-tensor inputs
        if self._skip_lm_head:
            return h
        output = self.lm_head(h) if self.lm_head is not None else h
        return output


_TPU_CLASSES: dict[type, type] = {}


def tpu_layer_loop_class(model_cls: type) -> type:
    """``Tpu<Model>``: ``model_cls`` with :class:`TpuLayerLoopMixin` in front.

    Built once per model class and cached, so the converter is idempotent and
    every converted config of one model shares one class.  The nested
    ``Config`` is a field-less subclass of ``model_cls.Config``, so ``derive``
    carries every field over and ``Configurable.__init_subclass__`` wires
    ``Config._owner`` to the new class (it sees ``Config`` in the class dict).
    """
    if model_cls in _TPU_CLASSES:
        return _TPU_CLASSES[model_cls]
    if not issubclass(model_cls, Decoder):
        raise TypeError(
            f"TpuLayerLoopConverter expects a Decoder subclass, got {model_cls.__name__}"
        )
    if issubclass(model_cls, TpuLayerLoopMixin):
        return model_cls  # already converted (idempotent)
    if model_cls.forward is not Decoder.forward:
        # The mixin replaces Decoder.forward; a model with its own forward
        # (e.g. one that loops over the layers itself) would lose it silently.
        raise TypeError(
            f"{model_cls.__name__} overrides Decoder.forward; TpuLayerLoopMixin "
            "cannot replace a model-specific forward"
        )
    name = f"Tpu{model_cls.__name__}"
    config_cls = dataclass(kw_only=True, slots=True)(
        type(
            "Config",
            (model_cls.Config,),
            {"__module__": __name__, "__qualname__": f"{name}.Config"},
        )
    )
    tpu_cls = type(
        name,
        (TpuLayerLoopMixin, model_cls),
        {
            "Config": config_cls,
            "__module__": __name__,
            "__qualname__": name,
            "__doc__": f"{model_cls.__name__} with the TPU-replaceable layer loop.",
        },
    )
    _TPU_CLASSES[model_cls] = tpu_cls
    return tpu_cls


class TpuLayerLoopConverter(ModelConfigConverter):
    """Swap the model class for its ``Tpu<Model>`` variant (see module docstring).

    Operates on the root config only: ``model_registry`` hands converters the
    ``Decoder.Config`` itself, and this converter returns the replacement root
    (the documented ``ModelConfigConverter`` contract).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config):
        pass

    def convert(self, model_config):
        if not isinstance(model_config, Decoder.Config):
            raise TypeError(
                "TpuLayerLoopConverter must be applied to a Decoder config root, "
                f"got {type(model_config).__name__}"
            )
        owner = type(model_config)._owner
        if owner is None:
            raise TypeError(f"{type(model_config).__name__} has no owner class")
        tpu_cls = tpu_layer_loop_class(owner)
        if type(model_config) is tpu_cls.Config:
            return model_config
        return derive(model_config, tpu_cls.Config)


__all__ = [
    "TpuLayerLoop",
    "TpuLayerLoopConverter",
    "TpuLayerLoopMixin",
    "tpu_layer_loop_class",
]
