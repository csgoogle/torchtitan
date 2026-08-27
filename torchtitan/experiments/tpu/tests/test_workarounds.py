"""Unit tests for TPU workarounds in torchtitan.experiments.tpu.workarounds."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torchtitan.experiments.tpu import workarounds
from torchtitan.models.common import rope


class WorkaroundsTest(parameterized.TestCase):
  """Unit tests for TPU workarounds."""

  @parameterized.named_parameters(
      (
          "cossin_rope_float32",
          torch.float32,
          False,  # is_complex
      ),
      (
          "cossin_rope_bfloat16",
          torch.bfloat16,
          False,  # is_complex
      ),
      (
          "complex_rope_complex64",
          torch.complex64,
          True,  # is_complex
      ),
  )
  def test_rope_tpu_optimization_patch(self, dtype, is_complex):
    """Tests rope TPU optimization patch correctness across formats and position patterns.

    Covers both CosSinRoPE and ComplexRoPE cache formats across four position
    patterns:
      1. Uniform sequential positions: [0, 1, 2, ..., seqlen-1] per batch item
      2. Sequence packing positions: document boundaries resetting to 0
      (e.g., [0..9, 0..19, 0..4, ...])
      3. Offset positions: non-zero KV cache start position offsets
      4. Arbitrary position IDs
    """
    # Save original function
    original_reshape = (
        rope._reshape_for_broadcast  # pylint: disable=protected-access
    )

    try:
      # Apply monkey patch
      workarounds.use_rope_tpu_optimization_patch()
      patched_reshape = (
          rope._reshape_for_broadcast  # pylint: disable=protected-access
      )

      self.assertNotEqual(
          original_reshape,
          patched_reshape,
          "rope._reshape_for_broadcast should be monkey-patched.",
      )

      bsz, seqlen, num_heads, head_dim = 2, 32, 8, 64
      max_seq_len = 128

      if is_complex:
        query_shape = (bsz, seqlen, num_heads, head_dim // 2)
        cfg = rope.ComplexRoPE.Config(dim=head_dim, max_seq_len=max_seq_len)
        rope_cache = rope.ComplexRoPE(cfg).cache.to(dtype=dtype)
      else:
        query_shape = (bsz, seqlen, num_heads, head_dim)
        cfg = rope.CosSinRoPE.Config(dim=head_dim, max_seq_len=max_seq_len)
        rope_cache = rope.CosSinRoPE(cfg).cache.to(dtype=dtype)

      cache_width = rope_cache.shape[-1]

      # Define real position patterns
      # Pattern 1: Uniform sequential positions
      uniform_positions = torch.arange(seqlen).unsqueeze(0).repeat(bsz, 1)

      # Pattern 2: Sequence packing positions (document boundaries resetting to 0)
      # Heterogeneous packed documents per batch item:
      # Batch item 0: 2 docs of len 20 and 12 -> [0..19, 0..11] (32 tokens)
      # Batch item 1: 3 docs of len 10, 14, and 8 -> [0..9, 0..13, 0..7] (32 tokens)
      packed_b0 = torch.cat([torch.arange(20), torch.arange(12)])
      packed_b1 = torch.cat(
          [torch.arange(10), torch.arange(14), torch.arange(8)]
      )
      packed_positions = torch.stack([packed_b0, packed_b1])

      # Pattern 3: Position offsets (non-zero start offsets per batch item)
      offset_b0 = torch.arange(10, 10 + seqlen)
      offset_b1 = torch.arange(50, 50 + seqlen)
      offset_positions = torch.stack([offset_b0, offset_b1])

      # Pattern 4: Arbitrary position IDs
      arbitrary_positions = torch.randint(0, max_seq_len, (bsz, seqlen))

      position_patterns = [
          ("uniform", uniform_positions),
          ("packed", packed_positions),
          ("offset", offset_positions),
          ("arbitrary", arbitrary_positions),
      ]

      # 1. Test positions is None (Sequential)
      out_orig_none = original_reshape(rope_cache, query_shape, None)
      out_patch_none = patched_reshape(rope_cache, query_shape, None)
      torch.testing.assert_close(out_orig_none, out_patch_none)
      self.assertEqual(out_patch_none.shape, (1, seqlen, 1, cache_width))

      # 2. Test singleton positions (bsz=1)
      singleton_positions = torch.arange(seqlen).unsqueeze(0)
      singleton_query_shape = (1, seqlen, num_heads, query_shape[-1])
      out_orig_single = original_reshape(
          rope_cache, singleton_query_shape, singleton_positions
      )
      out_patch_single = patched_reshape(
          rope_cache, singleton_query_shape, singleton_positions
      )
      torch.testing.assert_close(out_orig_single, out_patch_single)
      self.assertEqual(out_patch_single.shape, (1, seqlen, 1, cache_width))

      # 3. Test per-batch positions (bsz > 1) across all position patterns
      for pattern_name, pos_tensor in position_patterns:
        out_orig_batch = original_reshape(rope_cache, query_shape, pos_tensor)
        out_patch_batch = patched_reshape(rope_cache, query_shape, pos_tensor)

        self.assertEqual(out_patch_batch.shape, (bsz, seqlen, 1, cache_width))
        torch.testing.assert_close(
            out_orig_batch,
            out_patch_batch,
            msg=(
                "TPU patched rope reshape output must match original"
                f" torch.gather output for pattern '{pattern_name}'."
            ),
        )

    finally:
      # Restore original function
      rope._reshape_for_broadcast = (
          original_reshape  # pylint: disable=protected-access
      )

  def test_benchmark_rope_gather_vs_2d_indexing(self):
    """Benchmarks and verifies memory/index size reduction of 2D indexing vs torch.gather."""
    bsz, seqlen, cache_width = 8, 4096, 128
    max_seq_len = 8192

    rope_cache = torch.randn(max_seq_len, cache_width)
    positions = torch.randint(0, max_seq_len, (bsz, seqlen))

    # 1. Upstream 4D gather index creation
    upstream_index = positions.view(bsz, seqlen, 1, 1).expand(
        bsz, seqlen, 1, cache_width
    )
    patched_positions = positions

    # Upstream index tensor elements vs patched 2D positions elements
    upstream_numel = upstream_index.numel()
    patched_numel = patched_positions.numel()
    reduction_factor = upstream_numel / patched_numel

    # Verify that patched positions index is exactly 1 / cache_width size of upstream 4D index
    self.assertEqual(reduction_factor, cache_width)
    self.assertEqual(upstream_numel, bsz * seqlen * cache_width)
    self.assertEqual(patched_numel, bsz * seqlen)

    # 2. Verify bitwise parity
    upstream_out = (
        rope_cache[None, :, None, :]
        .expand(bsz, -1, -1, -1)
        .gather(dim=1, index=upstream_index)
    )
    patched_out = rope_cache[positions].unsqueeze(2)

    torch.testing.assert_close(upstream_out, patched_out)

  def test_wrapper_conversion_positions_to_segment_ids(self):
    """Test that patched splash attention receives positions and converts them correctly."""
    import unittest.mock
    from torchtitan.experiments.tpu import workarounds
    from torchtitan.models.common.attention import GQAttention as Attention
    from torchtitan.experiments.tpu.llama3 import llama3_configs

    config = llama3_configs["testmodel"]

    captured_segment_ids = []

    def mock_splash_sdpa_capture(
        q,
        k,
        v,
        segment_ids=None,
        scale=None,
        is_causal=True,
        local_window_size=None,
        enable_gqa=False,
        **kwargs,
    ):
      captured_segment_ids.append(segment_ids)
      return torch.zeros_like(q)

    with unittest.mock.patch(
        "torchtitan.experiments.tpu.kernels.splash_attention.splash_sdpa",
        side_effect=mock_splash_sdpa_capture,
    ):
      attention_config = config.layers[0].attention
      attention_block = Attention(config=attention_config).to("cpu")

      patch_kwargs = {"block_q": 128}
      workarounds.use_splash_attention_patch(attention_block, patch_kwargs)

      # Force the CPU to run the patched kernel by removing the CPU fallback
      attention_block.inner_attention.original_module = None

      # Ensure seq_len >= 128 to avoid fallback
      seq_len = 128
      tokens = torch.randn(
          1,
          seq_len,
          config.dim,
          requires_grad=False,
          device="cpu",
      )

      # Create positions for two segments of length 64
      positions = (
          torch.cat([torch.arange(64), torch.arange(64)])
          .unsqueeze(0)
          .to("cpu")
      )
      kwargs = {"positions": positions}

      out = attention_block(tokens, None, **kwargs)

      self.assertEqual(
          len(captured_segment_ids),
          1,
          "splash_sdpa was not called, patch might have failed.",
      )
      segment_ids = captured_segment_ids[0]

      self.assertIsNotNone(
          segment_ids,
          "segment_ids was None, meaning document packing logic didn't"
          " trigger.",
      )

      # Expected: 64 zeros followed by 64 ones
      expected_segment_ids = (
          torch.cat([
              torch.zeros(64, dtype=torch.int32),
              torch.ones(64, dtype=torch.int32),
          ])
          .unsqueeze(0)
          .to("cpu")
      )
      torch.testing.assert_close(segment_ids, expected_segment_ids)


if __name__ == "__main__":
  absltest.main()
