"""Tests SimpleFSDP shard_dim configuration and SPMD sharding config for Gemma 4."""

from types import SimpleNamespace
from absl.testing import absltest
import torch
from torch import nn

from torchtitan.experiments.tpu.gemma4 import _build_flavor_config
from torchtitan.experiments.tpu.gemma4.infra.parallelize import (
    configure_gemma4_simple_fsdp_shard_dims,
)
from torchtitan.experiments.tpu.gemma4.sharding import (
    set_gemma4_sharding_config,
)
from torchtitan.experiments.tpu.graph_trainer.layer_stack import (
    LayerStackHolder,
)
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher


def _fake_parallel_dims(dp_shard: int = 8, cp: int = 1, ep_enabled: bool = False):
  return SimpleNamespace(dp_shard=dp_shard, cp=cp, ep_enabled=ep_enabled)


def _toy_model(*, hidden_dim: int = 8, vocab_size: int = 8) -> nn.Module:
  model = nn.Module()
  model.tok_embeddings = nn.Embedding(vocab_size, 8)
  model.lm_head = nn.Linear(8, vocab_size, bias=False)
  layer = nn.Module()
  layer.moe = nn.Module()
  layer.moe.experts = TpuGroupedExperts.Config(
      dim=8,
      hidden_dim=hidden_dim,
      num_experts=2,
      token_dispatcher=LocalTokenDispatcher.Config(num_experts=2, top_k=1),
  ).build()
  model.layers = nn.ModuleDict({"0": layer})
  return model


class TestSimpleFSDPSharding(absltest.TestCase):
  """Tests SimpleFSDP _fsdp_shard_dim assignment and SPMD sharding configs."""

  def test_uneven_vocab_uses_hidden_dim_1(self):
    config = _build_flavor_config("testmodel")
    config.vocab_size = 2050  # Not divisible by 8, while dim=128 is divisible by 8
    config.tok_embeddings.num_embeddings = 2050
    config.lm_head.out_features = 2050
    with torch.device("meta"):
      model = config.build()

    configure_gemma4_simple_fsdp_shard_dims(
        model, _fake_parallel_dims(dp_shard=8)
    )

    self.assertEqual(getattr(model.tok_embeddings, "_fsdp_shard_dim", 0), 1)
    self.assertEqual(getattr(model.lm_head, "_fsdp_shard_dim", 0), 1)

  def test_uneven_num_experts_uses_hidden_dim_1(self):
    config = _build_flavor_config("testmodel_moe")
    converter = TpuGroupedExpertsConverter(TpuGroupedExpertsConverter.Config())
    config = converter.convert(config)
    for layer in config.layers:
      if layer.moe is not None:
        layer.moe.num_experts = 6  # Not divisible by 8, while hidden_dim=128 is
        layer.moe.experts.num_experts = 6
        layer.moe.router.num_experts = 6
        layer.moe.router.gate.out_features = 6

    with torch.device("meta"):
      model = config.build()

    configure_gemma4_simple_fsdp_shard_dims(
        model, _fake_parallel_dims(dp_shard=8, ep_enabled=False)
    )

    for layer in model.layers.values():
      self.assertIsNotNone(layer.moe)
      self.assertEqual(getattr(layer.moe.experts, "_fsdp_shard_dim", 0), 1)

  def test_keeps_default_axis_when_leading_axis_is_even(self):
    model = _toy_model()
    configure_gemma4_simple_fsdp_shard_dims(
        model, _fake_parallel_dims(dp_shard=2)
    )
    self.assertFalse(hasattr(model.layers["0"].moe.experts, "_fsdp_shard_dim"))

  def test_finds_experts_beneath_layer_wrappers(self):
    model = _toy_model()
    wrapper = nn.Module()
    wrapper.inner = model.layers["0"]
    model.layers["0"] = wrapper

    configure_gemma4_simple_fsdp_shard_dims(
        model, _fake_parallel_dims(dp_shard=4)
    )
    self.assertEqual(model.layers["0"].inner.moe.experts._fsdp_shard_dim, 1)

  def test_plans_an_axis_per_stack_when_layers_are_stacked(self):
    model = _toy_model()
    model.layer_stacks = LayerStackHolder()
    model.layer_stacks.register_parameter(
        "moe__experts__w1_EDF", nn.Parameter(torch.empty(2, 2, 8, 8))
    )
    model.layers["0"].moe.experts._parameters.clear()

    configure_gemma4_simple_fsdp_shard_dims(
        model, _fake_parallel_dims(dp_shard=4)
    )
    self.assertEqual(
        model.layer_stacks._fsdp_shard_dim, {"moe__experts__w1_EDF": 2}
    )

  def test_rejects_an_empty_expert_module_when_nothing_is_stacked(self):
    model = _toy_model()
    model.layers["0"].moe.experts._parameters.clear()

    with self.assertRaisesRegex(ValueError, "no direct parameters"):
      configure_gemma4_simple_fsdp_shard_dims(
          model, _fake_parallel_dims(dp_shard=4)
      )

  def test_rejects_when_hidden_axis_is_not_divisible(self):
    model = _toy_model(hidden_dim=6)

    with self.assertRaisesRegex(ValueError, "moe.experts.*hidden dimension 1"):
      configure_gemma4_simple_fsdp_shard_dims(
          model, _fake_parallel_dims(dp_shard=4)
      )

  def test_set_gemma4_sharding_config(self):
    for flavor in ("testmodel", "testmodel_moe"):
      config = _build_flavor_config(flavor)
      set_gemma4_sharding_config(config, enable_sp=True, enable_ep=False)
      self.assertIsNotNone(config.tok_embeddings.sharding_config)
      for layer in config.layers:
        self.assertIsNotNone(layer.sharding_config)
        self.assertIn("layer_scalar", layer.sharding_config.state_shardings)
        self.assertIsNotNone(layer.attention.v_norm.sharding_config)
        if layer.moe is not None:
          self.assertIsNotNone(layer.moe.router.sharding_config)
          self.assertIn(
              "scale", layer.moe.router.sharding_config.state_shardings
          )
          self.assertIn(
              "per_expert_scale",
              layer.moe.router.sharding_config.state_shardings,
          )


if __name__ == "__main__":
  absltest.main()
