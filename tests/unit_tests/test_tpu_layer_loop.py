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
"""``TpuLayerLoopMixin`` must stay a faithful copy of ``Decoder.forward``.

Two guards, neither of which pins a digest:

* parity: upstream ``Decoder.forward`` is called UNBOUND on the very instance
  the mixin runs on, so both implementations see the same weights and inputs,
  over every branch ``Decoder.forward`` has (embeddings / norm / lm_head
  present or ``None`` as on pipeline stages, ``_skip_lm_head``, ``positions``
  given or not, and a plain, empty, or ``TpuLayerLoop`` container);
* structure: the two functions' ASTs are equal after the one intended hunk
  (upstream's ``for`` loop, the mixin's ``if isinstance(...)``) is replaced by
  the same placeholder, and their signatures are identical.  Comments are not
  in the AST, so only code changes upstream fail this.
"""

import ast
import dataclasses
import inspect
import textwrap

import pytest
import torch

from torchtitan.experiments.tpu.layer_loop import (
    TpuLayerLoop,
    TpuLayerLoopConverter,
    TpuLayerLoopMixin,
    tpu_layer_loop_class,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.protocols.module import ModuleDict

SEQ = 16
BATCH = 2


def _small_config():
    config = llama3_configs["debugmodel"](attn_backend="sdpa")
    # Keep the RoPE cache small; the flavor defaults to max_seq_len=131072.
    for layer_cfg in config.layers:
        layer_cfg.attention.rope = dataclasses.replace(
            layer_cfg.attention.rope, max_seq_len=SEQ
        )
    return config


def _build_tpu_model():
    config = TpuLayerLoopConverter.Config().build().convert(_small_config())
    model = config.build()
    torch.manual_seed(0)
    model.init_states()
    model.eval()
    return model


def _inputs(model, *, with_positions: bool, tokens_are_embeddings: bool):
    g = torch.Generator().manual_seed(1)
    if tokens_are_embeddings:
        tokens = torch.randn(BATCH, SEQ, model.config.dim, generator=g)
    else:
        tokens = torch.randint(0, model.config.vocab_size, (BATCH, SEQ), generator=g)
    positions = (
        torch.arange(SEQ).unsqueeze(0).expand(BATCH, SEQ).clone()
        if with_positions
        else None
    )
    return tokens, positions


class _ScanLikeLoop(TpuLayerLoop):
    """A container that runs the layers some other way (here: same order, but
    through a different code path), standing in for the scan container."""

    def __init__(self, layers):
        super().__init__(layers)
        self.calls = 0

    def forward(self, h, *args):
        self.calls += 1
        for name in list(self.keys()):
            h = self[name](h, *args)
        return h


CONTAINERS = ("module_dict", "tpu_loop", "scan_like", "empty_module_dict")
STAGES = ("full", "no_embeddings", "no_norm_no_lm_head", "skip_lm_head")


def _apply_stage(model, stage):
    if stage == "no_embeddings":
        model.tok_embeddings = None
    elif stage == "no_norm_no_lm_head":
        model.norm = None
        model.lm_head = None
    elif stage == "skip_lm_head":
        model._skip_lm_head = True


def _apply_container(model, container):
    if container == "tpu_loop":
        model.layers = TpuLayerLoop(model.layers)
    elif container == "scan_like":
        model.layers = _ScanLikeLoop(model.layers)
    elif container == "empty_module_dict":
        # What ``_pipeline_module_split`` installs on a stage with no layers.
        model.layers = ModuleDict()


@pytest.mark.parametrize("with_positions", (False, True), ids=("nopos", "pos"))
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("container", CONTAINERS)
def test_mixin_forward_matches_upstream_forward(container, stage, with_positions):
    model = _build_tpu_model()
    _apply_stage(model, stage)
    _apply_container(model, container)
    tokens, positions = _inputs(
        model,
        with_positions=with_positions,
        tokens_are_embeddings=(stage == "no_embeddings"),
    )
    with torch.no_grad():
        expected = Decoder.forward(model, tokens, positions)  # upstream body
        actual = model(tokens, positions)  # TpuLayerLoopMixin body
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if container == "scan_like":
        # Both forwards ran; only the mixin went through the container.
        assert model.layers.calls == 1


def test_mixin_forward_gradients_match_upstream():
    model = _build_tpu_model()
    model.layers = TpuLayerLoop(model.layers)
    tokens, positions = _inputs(model, with_positions=True, tokens_are_embeddings=False)

    def grads(fn):
        model.zero_grad(set_to_none=True)
        fn(model, tokens, positions).float().sum().backward()
        return [p.grad.clone() for p in model.parameters()]

    expected = grads(Decoder.forward)
    actual = grads(type(model).forward)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


# ----------------------------------------------------------------------------
# Structural guard: AST equality modulo the one intended hunk.
# ----------------------------------------------------------------------------


def _function_def(fn) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    (node,) = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    return node


def _is_upstream_layer_loop(stmt: ast.stmt) -> bool:
    # for layer in self.layers.values(): ...
    return (
        isinstance(stmt, ast.For)
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id == "layer"
        and ast.unparse(stmt.iter) == "self.layers.values()"
    )


def _is_mixin_layer_dispatch(stmt: ast.stmt) -> bool:
    # if isinstance(self.layers, TpuLayerLoop): ... else: <upstream loop>
    return (
        isinstance(stmt, ast.If)
        and ast.unparse(stmt.test) == "isinstance(self.layers, TpuLayerLoop)"
        and len(stmt.orelse) == 1
        and _is_upstream_layer_loop(stmt.orelse[0])
    )


def _normalized(fn, is_hole) -> str:
    node = _function_def(fn)
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # docstring
    holes = [i for i, stmt in enumerate(body) if is_hole(stmt)]
    assert len(holes) == 1, f"{fn.__qualname__}: expected exactly one layer-loop hunk, found {len(holes)}"
    body = list(body)
    body[holes[0]] = ast.Pass()
    node.body = body
    node.decorator_list = []
    return ast.dump(ast.fix_missing_locations(node), include_attributes=False)


def test_mixin_forward_is_upstream_forward_modulo_the_loop():
    upstream = _normalized(Decoder.forward, _is_upstream_layer_loop)
    mixin = _normalized(TpuLayerLoopMixin.forward, _is_mixin_layer_dispatch)
    assert mixin == upstream, (
        "Decoder.forward changed upstream (or the mixin drifted). Re-sync "
        "TpuLayerLoopMixin.forward with torchtitan/models/common/decoder.py; "
        "only the layer-loop hunk may differ.\n"
        + "\n".join(
            _unified_diff(
                textwrap.dedent(inspect.getsource(Decoder.forward)),
                textwrap.dedent(inspect.getsource(TpuLayerLoopMixin.forward)),
            )
        )
    )


def _unified_diff(a: str, b: str):
    import difflib

    return difflib.unified_diff(
        a.splitlines(), b.splitlines(), "Decoder.forward", "TpuLayerLoopMixin.forward", lineterm=""
    )


def test_mixin_forward_signature_matches_upstream():
    assert inspect.signature(TpuLayerLoopMixin.forward) == inspect.signature(
        Decoder.forward
    )


def test_the_mixin_hunk_falls_back_to_the_upstream_loop():
    # The ``else`` branch of the dispatch IS the upstream loop, verbatim.
    node = _function_def(TpuLayerLoopMixin.forward)
    (dispatch,) = [s for s in node.body if _is_mixin_layer_dispatch(s)]
    upstream = _function_def(Decoder.forward)
    (loop,) = [s for s in upstream.body if _is_upstream_layer_loop(s)]
    assert ast.dump(dispatch.orelse[0]) == ast.dump(loop)


# ----------------------------------------------------------------------------
# Converter mechanics.
# ----------------------------------------------------------------------------


def test_converter_swaps_class_through_the_config_system():
    config = TpuLayerLoopConverter.Config().build().convert(_small_config())
    tpu_cls = tpu_layer_loop_class(Llama3Model)
    assert type(config) is tpu_cls.Config
    assert tpu_cls.__name__ == "TpuLlama3Model"
    assert tpu_cls.__mro__[:3] == (tpu_cls, TpuLayerLoopMixin, Llama3Model)
    model = config.build()
    assert isinstance(model, Llama3Model) and isinstance(model, TpuLayerLoopMixin)
    assert type(model).forward is TpuLayerLoopMixin.forward
    # Idempotent and cached.
    assert TpuLayerLoopConverter.Config().build().convert(config) is config
    assert tpu_layer_loop_class(Llama3Model) is tpu_cls
    # The stock class is untouched.
    assert Llama3Model.forward is Decoder.forward


def test_converter_preserves_every_config_field_and_state_dict_keys():
    stock_config = _small_config()
    # ``derive`` builds a new root and leaves ``stock_config`` untouched, so the
    # sub-configs are shared objects and compare by identity/value either way.
    tpu_config = TpuLayerLoopConverter.Config().build().convert(stock_config)
    assert type(stock_config) is Llama3Model.Config
    for f in dataclasses.fields(stock_config):
        assert getattr(tpu_config, f.name) == getattr(stock_config, f.name), f.name
    assert list(tpu_config.build().state_dict()) == list(stock_config.build().state_dict())


def test_converter_rejects_models_with_their_own_forward():
    class Custom(Llama3Model):
        @dataclasses.dataclass(kw_only=True, slots=True)
        class Config(Llama3Model.Config):
            pass

        def forward(self, tokens, positions=None, attention_masks=None):  # noqa: D401
            return tokens

    with pytest.raises(TypeError, match="overrides Decoder.forward"):
        tpu_layer_loop_class(Custom)


def test_converter_rejects_non_decoder_roots():
    from torchtitan.models.common.linear import Linear

    with pytest.raises(TypeError, match="Decoder config root"):
        TpuLayerLoopConverter.Config().build().convert(
            Linear.Config(in_features=4, out_features=4)
        )
