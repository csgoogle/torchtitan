"""Tests SimpleFSDP shard_dim configuration and SPMD sharding config for Gemma 4."""

from types import SimpleNamespace
from absl.testing import absltest
import torch

from torchtitan.experiments.tpu.gemma4 import _build_flavor_config
from torchtitan.experiments.tpu.gemma4.infra.parallelize import (
    configure_gemma4_simple_fsdp_shard_dims,
)
from torchtitan.experiments.tpu.gemma4.sharding import (
    set_gemma4_sharding_config,
)
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExpertsConverter,
)


def _fake_parallel_dims(dp_shard: int = 8, cp: int = 1, ep_enabled: bool = False):
  return SimpleNamespace(dp_shard=dp_shard, cp=cp, ep_enabled=ep_enabled)


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
