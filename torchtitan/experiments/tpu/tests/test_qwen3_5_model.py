# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end CPU checks for the text-only TPU Qwen3.5 model.

`test_gated_delta` covers the linear-attention kernel in isolation. This file
covers everything around it: that a text-only config (no vision tower) builds,
that a forward and backward pass produce finite values, and that swapping the
`tpu_chunked` kernel for the reference `torch_native` one does not change the
answer. A NaN loss on TPU is expensive to diagnose, so the same failure should
be reproducible here in seconds.
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torchtitan.models.common import Embedding, Linear
import torchtitan.models.qwen3_5 as qwen3_5_models
from torchtitan.models.qwen3_5.model import Qwen35Model
from torchtitan.models.qwen3_5.rope import MRoPE

_VOCAB_SIZE = 64
_DIM = 64
_HEAD_DIM = 32
_ROTARY_DIM = 8
# A multiple of the full-attention interval, so the stack contains both a
# GatedDeltaNet layer and a full-attention layer.
_N_LAYERS = 4


def _small_text_config(fla_backend: str) -> Qwen35Model.Config:
  """A miniature of `experiments.tpu.qwen3_5._35b_a3b_tpu`.

  Shapes are shrunk to keep the test fast, but every structural feature that
  distinguishes Qwen3.5 is kept: the hybrid GatedDeltaNet / full-attention
  decoder, partial RoPE, grouped value heads, a shared expert alongside the
  routed ones, and no vision tower.

  Args:
    fla_backend: Linear-attention backend to build the delta-net layers with.

  Returns:
    A text-only `Qwen35Model.Config`.
  """
  return Qwen35Model.Config(
      vocab_size=_VOCAB_SIZE,
      dim=_DIM,
      # pyrefly: ignore [bad-argument-type]
      norm=qwen3_5_models._offset_norm(_DIM),
      tok_embeddings=Embedding.Config(
          num_embeddings=_VOCAB_SIZE,
          embedding_dim=_DIM,
          param_init=qwen3_5_models._EMBEDDING_INIT,
      ),
      lm_head=Linear.Config(
          in_features=_DIM,
          out_features=_VOCAB_SIZE,
          param_init=qwen3_5_models._output_linear_init(_DIM),
      ),
      # pytype: disable=wrong-keyword-args
      layers=qwen3_5_models._build_qwen35_moe_layers(
          rope=MRoPE.Config(
              dim=_ROTARY_DIM,
              max_seq_len=512,
              theta=10_000_000.0,
              mrope_section=[3, 3, 2],
          ),
          attn_backend="sdpa",
          n_layers=_N_LAYERS,
          dim=_DIM,
          n_heads=4,
          n_kv_heads=2,
          head_dim=_HEAD_DIM,
          rotary_dim=_ROTARY_DIM,
          moe_hidden_dim=32,
          num_experts=8,
          top_k=2,
          shared_expert_hidden_dim=32,
          n_key_heads=2,
          n_value_heads=4,
          key_head_dim=32,
          value_head_dim=32,
          full_attention_interval=4,
          fla_backend=fla_backend,
          moe_comm_backend="standard",
      ),
      # pytype: enable=wrong-keyword-args
      vision_encoder=None,
  )


def _build(fla_backend: str, seed: int = 0) -> Qwen35Model:
  torch.manual_seed(seed)
  model = _small_text_config(fla_backend).build()
  model.init_weights()
  return model


def _batch(batch: int = 2, seq_len: int = 24, seed: int = 1):
  generator = torch.Generator().manual_seed(seed)
  tokens = torch.randint(0, _VOCAB_SIZE, (batch, seq_len), generator=generator)
  positions = torch.arange(seq_len).unsqueeze(0).expand(batch, seq_len)
  return tokens, positions


class Qwen35TpuModelTest(parameterized.TestCase):

  def test_text_only_config_builds_without_vision_tower(self):
    model = _build("tpu_chunked")

    self.assertIsNone(model.vision_encoder)
    self.assertIsNone(model.spatial_merge_size)

  @parameterized.named_parameters(
      ("tpu_chunked", "tpu_chunked"),
      ("torch_native", "torch_native"),
  )
  def test_forward_is_finite(self, fla_backend: str):
    model = _build(fla_backend)
    tokens, positions = _batch()

    logits = model(tokens, positions=positions)

    self.assertEqual(logits.shape, (2, 24, _VOCAB_SIZE))
    self.assertTrue(torch.isfinite(logits).all())

  def test_backward_is_finite(self):
    """A NaN gradient with a finite forward is the harder failure to spot."""
    model = _build("tpu_chunked")
    tokens, positions = _batch()

    logits = model(tokens, positions=positions)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(), tokens.flatten()
    )
    loss.backward()

    self.assertTrue(torch.isfinite(loss))
    for name, param in model.named_parameters():
      if param.grad is not None:
        with self.subTest(param=name):
          self.assertTrue(torch.isfinite(param.grad).all())

  def test_tpu_backend_matches_reference_backend(self):
    """The two backends must be the same function, not merely both finite."""
    tokens, positions = _batch()

    chunked = _build("tpu_chunked")(tokens, positions=positions)
    native = _build("torch_native")(tokens, positions=positions)

    torch.testing.assert_close(chunked, native, rtol=1e-4, atol=1e-4)


class Qwen35TpuRealisticScaleTest(parameterized.TestCase):
  """Regression tests at the shapes the 35B-A3B config actually uses.

  The miniature config above passed while the real 4-layer model produced a
  NaN loss on TPU, so the per-head dimensions, the partial RoPE split and the
  training sequence length are all reproduced exactly here. Only the vocabulary
  and the expert count are shrunk, since neither feeds the linear-attention
  recurrence.
  """

  def _build_realistic(self, *, dtype: torch.dtype):
    torch.manual_seed(0)
    config = Qwen35Model.Config(
        vocab_size=_VOCAB_SIZE,
        dim=2048,
        # pyrefly: ignore [bad-argument-type]
        norm=qwen3_5_models._offset_norm(2048),
        tok_embeddings=Embedding.Config(
            num_embeddings=_VOCAB_SIZE,
            embedding_dim=2048,
            param_init=qwen3_5_models._EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=2048,
            out_features=_VOCAB_SIZE,
            param_init=qwen3_5_models._output_linear_init(2048),
        ),
        # pytype: disable=wrong-keyword-args
        layers=qwen3_5_models._build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=64,
                max_seq_len=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend="sdpa",
            n_layers=4,
            dim=2048,
            n_heads=16,
            n_kv_heads=2,
            # head_dim 256 with rotary_dim 64 is the partial-RoPE split.
            head_dim=256,
            rotary_dim=64,
            # Shrunk: the MoE block is not on the path under test.
            moe_hidden_dim=32,
            num_experts=8,
            top_k=2,
            shared_expert_hidden_dim=32,
            # Real linear-attention geometry: value heads outnumber key heads
            # 2:1, so `GatedDeltaKernel` repeat-interleaves q and k.
            n_key_heads=16,
            n_value_heads=32,
            key_head_dim=128,
            value_head_dim=128,
            full_attention_interval=4,
            fla_backend="tpu_chunked",
            moe_comm_backend="standard",
        ),
        # pytype: enable=wrong-keyword-args
        vision_encoder=None,
    )
    model = config.build()
    model.init_weights()
    return model.to(dtype)

  @parameterized.named_parameters(
      # 2048 is the training sequence length: 32 chunks of 64, so the carried
      # state is threaded through the full recurrence rather than a single
      # chunk.
      ("training_seq_len_fp32", 2048, torch.float32),
      ("training_seq_len_bf16", 2048, torch.bfloat16),
      ("short_seq_len_bf16", 128, torch.bfloat16),
  )
  def test_forward_is_finite(self, seq_len: int, dtype: torch.dtype):
    model = self._build_realistic(dtype=dtype)
    tokens, positions = _batch(batch=1, seq_len=seq_len)

    logits = model(tokens, positions=positions)

    self.assertTrue(torch.isfinite(logits).all())

  def test_full_35b_and_proxy_config_registry(self):
    from torchtitan.experiments.tpu.qwen3_5 import (  # pylint: disable=g-import-not-at-top
        qwen35_moe_35b,
        qwen35_moe_35b_fsdp_only_full_ac_benchmark,
        qwen35_moe_35b_proxy,
    )

    full_cfg = qwen35_moe_35b()
    self.assertIsNotNone(full_cfg.model_spec)
    self.assertLen(full_cfg.model_spec.model.layers, 40)
    self.assertEqual(full_cfg.training.local_batch_size, 6)
    self.assertEqual(full_cfg.training.global_batch_size, 768)

    bench_cfg = qwen35_moe_35b_fsdp_only_full_ac_benchmark()
    self.assertIsNotNone(bench_cfg.model_spec)
    self.assertLen(bench_cfg.model_spec.model.layers, 40)
    self.assertEqual(bench_cfg.training.steps, 30)
    self.assertEqual(bench_cfg.training.mixed_precision_reduce, "float32")
    self.assertTrue(bench_cfg.tpu_config.enable_mfu_benchmark)
    self.assertEqual(bench_cfg.tpu_config.mfu_benchmark_start_step, 10)
    self.assertTrue(
        bench_cfg.splash_attention_kernel.use_splash_attention_kernel
    )
    self.assertEqual(bench_cfg.splash_attention_kernel.sa_block_q_dkv, 1024)
    self.assertIn(
        "torchtitan.experiments.tpu.overrides.fused_loss",
        bench_cfg.override.imports,
    )

    proxy_cfg = qwen35_moe_35b_proxy()
    self.assertIsNotNone(proxy_cfg.model_spec)
    self.assertLen(proxy_cfg.model_spec.model.layers, 4)


if __name__ == "__main__":
  absltest.main()
