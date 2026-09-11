"""The TPU ``forward`` must stay a faithful copy of ``Decoder.forward``.

Two guards, so an upstream edit to ``decoder.py`` fails here rather than
diverging silently: *parity* runs both forwards on one instance, and *structure*
compares their ASTs with each one's layer-loop hunk blanked out.
"""

import ast
import dataclasses
import difflib
import inspect
import textwrap

from absl.testing import absltest
from absl.testing import parameterized
import torch

from torchtitan.experiments.tpu.layer_loop import (
    _forward,
    runs_layer_loop,
    TpuLayerLoop,
    with_tpu_layer_loop,
)
from torchtitan.models.common import Linear
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.protocols.module import ModuleDict

BATCH, SEQ = 2, 16


def _stock_config():
  config = llama3_configs["debugmodel"](attn_backend="sdpa")
  # The flavor defaults to max_seq_len=131072; keep the RoPE cache small.
  for layer_cfg in config.layers:
    layer_cfg.attention.rope = dataclasses.replace(
        layer_cfg.attention.rope, max_seq_len=SEQ
    )
  return config


def _recast(config, cls):
  """Re-class a config, the way `MaxTextQwen3Config.from_qwen3` does."""
  # Shallow: `dataclasses.asdict` would flatten the nested configs.
  fields = (f.name for f in dataclasses.fields(config) if f.init)
  return cls(**{name: getattr(config, name) for name in fields})


def _tpu_model():
  model = with_tpu_layer_loop(_stock_config()).build()
  torch.manual_seed(0)
  model.init_states()
  model.eval()
  return model


def _tokens(model, *, with_positions):
  g = torch.Generator().manual_seed(1)
  tokens = torch.randint(0, model.config.vocab_size, (BATCH, SEQ), generator=g)
  positions = (
      torch.arange(SEQ).unsqueeze(0).expand(BATCH, SEQ).clone()
      if with_positions
      else None
  )
  return tokens, positions


class _ScanLikeLoop(TpuLayerLoop):
  """Stands in for the scan container: runs the same layers, its own way."""

  def __init__(self, layers):
    super().__init__(layers)
    self.calls = 0

  def forward(self, h, *args):
    self.calls += 1
    for name in list(self.keys()):
      h = self[name](h, *args)
    return h


CONTAINERS = {
    "module_dict": lambda layers: layers,  # untouched: takes upstream's loop
    "tpu_loop": TpuLayerLoop,
    "scan_like": _ScanLikeLoop,
    "empty": lambda layers: ModuleDict(),  # what the pipeline splitter installs
}


class ForwardParityTest(parameterized.TestCase):

  @parameterized.product(
      container=tuple(CONTAINERS), with_positions=(False, True)
  )
  def test_forward_matches_upstream(
      self, container: str, with_positions: bool
  ) -> None:
    model = _tpu_model()
    model.layers = CONTAINERS[container](model.layers)
    tokens, positions = _tokens(model, with_positions=with_positions)

    with torch.no_grad():
      expected = Decoder.forward(model, tokens, positions)  # upstream body
      actual = model(tokens, positions)  # the TPU body

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if container == "scan_like":
      # Both forwards ran, but only ours went through the container.
      self.assertEqual(model.layers.calls, 1)

  def test_layerless_pipeline_stage_is_a_passthrough(self) -> None:
    """An unconditional `self.layers(...)` would raise: ModuleDict has no forward."""
    model = _tpu_model()
    model.tok_embeddings = None
    model.norm = None
    model.lm_head = None
    model.layers = ModuleDict()

    h = torch.randn(BATCH, SEQ, model.config.dim)
    with torch.no_grad():
      torch.testing.assert_close(model(h), h, rtol=0, atol=0)

  def test_gradients_match_upstream(self) -> None:
    model = _tpu_model()
    model.layers = TpuLayerLoop(model.layers)
    tokens, positions = _tokens(model, with_positions=True)

    def grads(fn):
      model.zero_grad(set_to_none=True)
      fn(model, tokens, positions).float().sum().backward()
      return [p.grad.clone() for p in model.parameters()]

    expected = grads(Decoder.forward)
    actual = grads(type(model).forward)
    for a, e in zip(actual, expected, strict=True):
      torch.testing.assert_close(a, e, rtol=0, atol=0)


def _is_upstream_loop(stmt) -> bool:
  """`for layer in self.layers.values(): ...`"""
  return (
      isinstance(stmt, ast.For)
      and isinstance(stmt.target, ast.Name)
      and stmt.target.id == "layer"
      and ast.unparse(stmt.iter) == "self.layers.values()"
  )


def _is_tpu_dispatch(stmt) -> bool:
  """`if isinstance(self.layers, TpuLayerLoop): ... else: <upstream loop>`"""
  return (
      isinstance(stmt, ast.If)
      and ast.unparse(stmt.test) == "isinstance(self.layers, TpuLayerLoop)"
      and len(stmt.orelse) == 1
      and _is_upstream_loop(stmt.orelse[0])
  )


def _blank_the_loop(fn, is_hole) -> str:
  """Dump `fn`'s AST with its layer-loop statement replaced by `pass`."""
  tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
  (node,) = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
  body = list(node.body)
  if (
      body
      and isinstance(body[0], ast.Expr)
      and isinstance(getattr(body[0], "value", None), ast.Constant)
      and isinstance(body[0].value.value, str)
  ):
    body = body[1:]  # drop the docstring
  holes = [i for i, stmt in enumerate(body) if is_hole(stmt)]
  if len(holes) != 1:
    raise AssertionError(
        f"{fn.__qualname__}: expected one layer-loop hunk, found {len(holes)}"
    )
  body[holes[0]] = ast.Pass()
  node.body = body
  node.name = "forward"  # `Decoder.forward` vs the module-level `_forward`
  node.decorator_list = []
  return ast.dump(ast.fix_missing_locations(node), include_attributes=False)


class ForwardStructureTest(absltest.TestCase):

  def test_forward_is_upstream_forward_modulo_the_loop(self) -> None:
    diff = "\n".join(
        difflib.unified_diff(
            textwrap.dedent(inspect.getsource(Decoder.forward)).splitlines(),
            textwrap.dedent(inspect.getsource(_forward)).splitlines(),
            "Decoder.forward",
            "layer_loop._forward",
            lineterm="",
        )
    )
    self.assertEqual(
        _blank_the_loop(_forward, _is_tpu_dispatch),
        _blank_the_loop(Decoder.forward, _is_upstream_loop),
        "Decoder.forward changed upstream, or the copy drifted; re-sync "
        f"layer_loop._forward with models/common/decoder.py.\n{diff}",
    )


class ConfigRewriteTest(absltest.TestCase):

  def test_rewrite_swaps_the_model_class_and_keeps_every_field(self) -> None:
    stock = _stock_config()
    converted = with_tpu_layer_loop(stock)

    self.assertEqual(type(converted)._owner.__name__, "TpuLlama3Model")
    for f in dataclasses.fields(stock):
      self.assertEqual(getattr(converted, f.name), getattr(stock, f.name), f.name)
    # One class per config class, so rewriting twice is a no-op.
    self.assertIs(type(with_tpu_layer_loop(converted)), type(converted))

    model = converted.build()
    self.assertIsInstance(model, Llama3Model)
    self.assertTrue(runs_layer_loop(model))

  def test_rewrite_keeps_an_earlier_root_config_replacement(self) -> None:
    """qwen3's TPU registry swaps the root config to override FLOP accounting."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class CustomFlopsConfig(Llama3Model.Config):

      def get_nparams_and_flops(self, *args, **kwargs):
        return (0, 0)

    converted = with_tpu_layer_loop(_recast(_stock_config(), CustomFlopsConfig))

    self.assertIsInstance(converted, CustomFlopsConfig)
    self.assertEqual(converted.get_nparams_and_flops(), (0, 0))
    self.assertTrue(runs_layer_loop(converted.build()))

  def test_state_dict_keys_are_unchanged(self) -> None:
    torch.manual_seed(0)
    stock = _stock_config().build()
    stock.init_states()
    self.assertEqual(
        list(_tpu_model().state_dict()), list(stock.state_dict())
    )

  def test_rewrite_rejects_what_it_cannot_convert(self) -> None:
    """A non-decoder root, and a model whose own `forward` it would shadow."""
    with self.assertRaisesRegex(TypeError, "stock forward"):
      with_tpu_layer_loop(Linear.Config(in_features=4, out_features=4))

    class ForwardOwningModel(Llama3Model):

      @dataclasses.dataclass(kw_only=True, slots=True)
      class Config(Llama3Model.Config):
        pass

      def forward(self, *args, **kwargs):
        raise AssertionError("never called")

    with self.assertRaisesRegex(TypeError, "stock forward"):
      with_tpu_layer_loop(_recast(_stock_config(), ForwardOwningModel.Config))


if __name__ == "__main__":
  absltest.main()
