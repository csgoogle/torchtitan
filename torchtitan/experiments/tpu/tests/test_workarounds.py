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
    self.skipTest(
        "use_rope_tpu_optimization_patch is a no-op on the token-major layout:"
        " upstream's _reshape_for_broadcast already indexes with 1-D positions"
    )
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
    """Patched splash attention converts positions to segment ids."""
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


      # Ensure seq_len >= 128 to avoid fallback. Activations are token-major
      # ``[T, D]``; the CPU device would take the reference path, so pretend
      # the tensors live on a TPU for the dispatch decision.
      seq_len = 128
      tokens = torch.randn(
          seq_len,
          config.dim,
          requires_grad=False,
          device="cpu",
      )

      # Create positions for two segments of length 64
      positions = torch.cat([torch.arange(64), torch.arange(64)]).to("cpu")
      kwargs = {"positions": positions}
      attention_block.inner_attention._use_kernel = lambda q, row_len: True

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

      # Expected: 64 zeros followed by 64 ones (one packed row of batch 1)
      expected_segment_ids = (
          torch.cat([
              torch.zeros(64, dtype=torch.int32),
              torch.ones(64, dtype=torch.int32),
          ])
          .unsqueeze(0)
          .to("cpu")
      )
      torch.testing.assert_close(segment_ids, expected_segment_ids)


  def test_pallas_loss_linear_preserves_state_dict_keys(self):
    """The loss-kernel wrapper must not rename the output-projection parameter.

    use_output_projection_patch replaces lm_head with PallasLossLinear. If the
    wrapper holds the Linear as a child module the parameter becomes
    `lm_head.original_linear.weight`, which the model state dict adapters cannot
    map -- loading pretrained HF safetensors then dies with
    KeyError: 'lm_head.original_linear.weight'.
    """
    import torch.nn as nn
    from torchtitan.experiments.tpu import workarounds

    class _TinyModel(nn.Module):

      def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(8, 16, bias=False)

    model = _TinyModel()
    before = sorted(model.state_dict().keys())
    original_weight = model.lm_head.weight

    workarounds.use_output_projection_patch(model)

    self.assertIsInstance(model.lm_head, workarounds.PallasLossLinear)
    self.assertEqual(
        sorted(model.state_dict().keys()),
        before,
        "loss-kernel patch renamed the output projection parameter",
    )
    # The Parameter object must be the same one, so FSDP shards it in place and
    # weight tying (lm_head.weight is tok_embeddings.weight) still holds.
    self.assertIs(model.lm_head.weight, original_weight)
    self.assertIsInstance(model.lm_head.weight, nn.Parameter)

    # And the wrapper still does its job: return (x, W^T) for the Pallas loss.
    x = torch.randn(2, 3, 8)
    out, weight_t = model.lm_head(x)
    torch.testing.assert_close(out, x)
    self.assertEqual(tuple(weight_t.shape), (8, 16))

  def test_fused_loss_matches_unfused_cross_entropy(self):
    """The deferred LM-head path preserves loss and gradient numerics."""
    import sys
    import types
    from unittest import mock

    import torch.nn.functional as F
    from torchtitan.components import loss as components_loss
    from torchtitan.experiments.tpu import kernels
    from torchtitan.experiments.tpu.loss import pallas_cross_entropy_loss
    from torchtitan.experiments.tpu.overrides.fused_loss import (
        fused_loss_lm_head,
    )
    from torchtitan.models.common import Linear

    torch.manual_seed(0)
    linear_config = Linear.Config(
        in_features=8,
        out_features=16,
        bias=False,
    )
    unfused_head = linear_config.build()
    fused_head = fused_loss_lm_head(linear_config).build()
    fused_head.load_state_dict(unfused_head.state_dict())

    # Token-major ``[T, D]`` hidden states and ``[T]`` labels.
    unfused_hidden = torch.randn(8, 8, requires_grad=True)
    fused_hidden = unfused_hidden.detach().clone().requires_grad_(True)
    labels = torch.tensor([1, 4, components_loss.IGNORE_INDEX, 2, 5, 3, 0, 7])

    expected_loss = components_loss.cross_entropy_loss(
        unfused_head(unfused_hidden), labels
    )

    # Kernel parity is covered by test_linear_softmax_cross_entropy_loss. Stub
    # only its TPU-specific dispatch here so this CPU test covers the adapter,
    # weight orientation, sum reduction, and ignored-label behavior together.
    def reference_linear_loss(x, kernel_labels, weights, **unused_kwargs):
      return F.cross_entropy(
          x.float() @ weights.float(),
          kernel_labels,
          reduction="sum",
      )

    kernel_module_name = (
        "torchtitan.experiments.tpu.kernels."
        "linear_softmax_cross_entropy_loss"
    )
    fake_kernel_module = types.ModuleType(kernel_module_name)
    fake_kernel_module.linear_softmax_cross_entropy_loss = reference_linear_loss
    with (
        mock.patch.dict(sys.modules, {kernel_module_name: fake_kernel_module}),
        mock.patch.object(
            kernels,
            "linear_softmax_cross_entropy_loss",
            fake_kernel_module,
            create=True,
        ),
    ):
      actual_loss = pallas_cross_entropy_loss(
          fused_head(fused_hidden),
          labels,
          implementation="xla",
      )

    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-5)

    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(
        fused_hidden.grad,
        unfused_hidden.grad,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        fused_head.weight.grad,
        unfused_head.weight.grad,
        rtol=1e-5,
        atol=1e-5,
    )

  def test_pallas_loss_linear_patches_output_and_output_transform(self):
    """Key preservation must also hold for `output` / `output_transform`."""
    import torch.nn as nn
    from torchtitan.experiments.tpu import workarounds

    for attr in ("output", "output_transform"):
      model = nn.Module()
      setattr(model, attr, nn.Linear(4, 6, bias=False))
      before = sorted(model.state_dict().keys())

      workarounds.use_output_projection_patch(model)

      self.assertIsInstance(getattr(model, attr), workarounds.PallasLossLinear)
      self.assertEqual(
          sorted(model.state_dict().keys()), before, f"{attr} was renamed"
      )


  def test_loss_kernel_patch_keeps_hf_state_dict_adapter_working(self):
    """A patched model must still convert through the HF state dict adapter.

    This is the end-to-end regression: apply_patches runs before
    checkpointer.load(), so if the loss-kernel patch renames the output
    projection, loading pretrained HF safetensors dies in to_hf with
    KeyError: 'lm_head.original_linear.weight'.
    """
    from torchtitan.models.llama3 import llama3_configs
    from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
    from torchtitan.experiments.tpu import workarounds

    config = llama3_configs["debugmodel"][0](attn_backend="flex", seq_len=128)
    adapter = Llama3StateDictAdapter(config, None)

    with torch.device("meta"):
      model = config.build()
    workarounds.use_output_projection_patch(model)
    self.assertIsInstance(model.lm_head, workarounds.PallasLossLinear)

    # Must not raise, and must produce the HF-side name.
    hf_state_dict = adapter.to_hf(dict(model.state_dict()))
    self.assertIn("lm_head.weight", hf_state_dict)


  def test_loss_kernel_patch_preserves_weight_initialization(self):
    """A patched model must still have its output projection initialized.

    apply_patches runs before the trainer materializes weights, and
    Module.init_states initializes by recursing over children(). A wrapper that
    adopts the Parameter instead of keeping the Linear as a child drops out of
    that walk, leaving lm_head.weight all zeros for any run that does not load a
    checkpoint -- i.e. every from-scratch pretraining run, which then NaNs on
    the first step. SFT-from-pretrained cannot catch this, because the
    checkpoint load
    overwrites the zeros.
    """
    from torchtitan.models.llama3 import llama3_configs
    from torchtitan.experiments.tpu import workarounds

    config = llama3_configs["debugmodel"][0](attn_backend="flex", seq_len=128)

    def build(patch):
      torch.manual_seed(0)
      with torch.device("meta"):
        model = config.build()
      if patch:
        workarounds.use_output_projection_patch(model)
      model.to_empty(device="cpu")
      model.init_states()
      return model.lm_head.weight

    baseline = build(patch=False)
    patched = build(patch=True)

    self.assertTrue(torch.isfinite(patched).all(), "lm_head weight is not finite")
    self.assertGreater(
        patched.float().std().item(), 0.0, "lm_head weight was never initialized"
    )
    torch.testing.assert_close(patched, baseline)


if __name__ == "__main__":
  absltest.main()
