"""Unit tests for TPU workarounds in torchtitan.experiments.tpu.workarounds."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torchtitan.experiments.tpu import workarounds


class WorkaroundsTest(parameterized.TestCase):
  """Unit tests for TPU workarounds."""

  # NOTE: `test_rope_tpu_optimization_patch` and
  # `test_benchmark_rope_gather_vs_2d_indexing` were removed alongside
  # `workarounds.use_rope_tpu_optimization_patch` in the d771abb0 upstream
  # import. Upstream's `_reshape_for_broadcast` now performs the same 2D
  # advanced indexing over the folded `[T]` token stream, so there is no longer
  # a patched-vs-original pair to compare.


  def test_wrapper_conversion_positions_to_segment_ids(self):
    """Patched splash attention converts positions to segment ids."""
    import unittest.mock
    from torchtitan.experiments.tpu import workarounds
    from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
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

      # TT-UNFLATTEN-SHIM: the patch wraps the kernel in `UnflattenShim`, so
      # `inner_attention` is the shim and the kernel is its `.inner`. Remove the
      # kernel's CPU fallback so the patched path actually runs here.
      attention_block.inner_attention.inner.original_module = None

      # Upstream d771abb0 folds the batch axis away: the decoder now passes a
      # packed `[T, D]` token stream and `[T]` positions. The splash kernel is
      # still `[B, S, H, D]`, so the shim has to be active for this seam to
      # work at all -- that is what it is for.
      #
      # Both documents share one sequence (B=1, S=T). Splitting them across two
      # batch rows would make the unflattening itself separate them, and the
      # positions-to-segment-ids conversion under test would have nothing to do.
      num_tokens = 128
      tokens = torch.randn(
          num_tokens,
          config.dim,
          requires_grad=False,
          device="cpu",
      )

      # Create positions for two segments of length 64
      positions = torch.cat([torch.arange(64), torch.arange(64)]).to("cpu")
      kwargs = {"positions": positions}

      with set_unflatten_seq_len(num_tokens):
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

      # Expected: 64 zeros followed by 64 ones, on the one unflattened row.
      expected_segment_ids = torch.cat([
          torch.zeros(64, dtype=torch.int32),
          torch.ones(64, dtype=torch.int32),
      ]).unsqueeze(0).to("cpu")
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

    # Upstream d771abb0 folds the batch axis away: `cross_entropy_loss` takes
    # `pred[T, V]` and `labels[T]`.
    unfused_hidden = torch.randn(8, 8, requires_grad=True)
    fused_hidden = unfused_hidden.detach().clone().requires_grad_(True)
    labels = torch.tensor(
        [1, 4, components_loss.IGNORE_INDEX, 2, 5, 3, 0, 7]
    )

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

    # Upstream d771abb0 made `llama3_configs[flavor]` a
    # `(builder, max_context_length)` pair with a required keyword-only
    # `seq_len`.
    build_config, max_context_length = llama3_configs["debugmodel"]
    config = build_config(attn_backend="sdpa", seq_len=max_context_length)
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

    # Upstream d771abb0 made `llama3_configs[flavor]` a
    # `(builder, max_context_length)` pair with a required keyword-only
    # `seq_len`.
    build_config, max_context_length = llama3_configs["debugmodel"]
    config = build_config(attn_backend="sdpa", seq_len=max_context_length)

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


  def _tokamax_patch_fixture(self):
    """A module with one ScaledDotProductInnerAttention, and a populated splash config."""
    from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig  # pylint: disable=g-import-not-at-top
    from torchtitan.models.common.attention import ScaledDotProductInnerAttention  # pylint: disable=g-import-not-at-top

    class _Tiny(torch.nn.Module):

      def __init__(self):
        super().__init__()
        attn = ScaledDotProductInnerAttention.__new__(ScaledDotProductInnerAttention)
        torch.nn.Module.__init__(attn)
        self.attn = attn

    sa = SplashAttentionKernelConfig(
        use_tokamax_splash_attention_kernel=True,
        sa_block_q=1024,
        sa_block_kv=1024,
        sa_block_kv_compute=512,
        sa_block_q_dkv=2048,
        sa_block_kv_dkv=2048,
        sa_block_kv_dkv_compute=1024,
        sa_q_layout="SEQ_MINOR",
        sa_k_layout="SEQ_MINOR",
        sa_v_layout="HEAD_DIM_MINOR",
    )
    sa.use_vmap_bwd = True
    return _Tiny(), sa

  def test_tokamax_patch_forwards_splash_config(self):
    """The splash config must actually reach TokamaxConfig.

    Regression test. SplashAttentionKernelConfig declares its fields sa_-prefixed;
    reading them unprefixed returns None for every one, and TokamaxConfig.update
    skips None, so the patch silently kept tokamax's library defaults and the kernel
    ran at the wrong tiles. Nothing failed - it was just slower, which is exactly why
    this needs a test rather than a runtime error.
    """
    from torchtitan.experiments.tpu.kernels.tokamax_splash_attention import TokamaxConfig  # pylint: disable=g-import-not-at-top

    model, sa = self._tokamax_patch_fixture()
    self.assertEqual(TokamaxConfig.block_q, 512, "precondition: class default")

    workarounds.use_tokamax_splash_attention_patch(model, sa)

    self.assertEqual(TokamaxConfig.block_q, 1024)
    self.assertEqual(TokamaxConfig.block_kv, 1024)
    self.assertEqual(TokamaxConfig.block_kv_compute, 512)
    self.assertEqual(TokamaxConfig.block_q_dkv, 2048)
    self.assertEqual(TokamaxConfig.block_kv_dkv, 2048)
    self.assertEqual(TokamaxConfig.block_kv_dkv_compute, 1024)
    self.assertEqual(TokamaxConfig.q_layout, "SEQ_MINOR")
    self.assertEqual(TokamaxConfig.k_layout, "SEQ_MINOR")
    self.assertEqual(TokamaxConfig.v_layout, "HEAD_DIM_MINOR")
    self.assertTrue(TokamaxConfig.use_vmap_bwd)

  def test_tokamax_patch_raises_when_a_splash_config_resolves_nothing(self):
    """A real splash config whose fields resolve to nothing must fail loudly."""
    model, _ = self._tokamax_patch_fixture()

    class _BrokenSplash:  # exposes sa_ fields, so it IS a splash config
      use_tokamax_splash_attention_kernel = True
      sa_block_q = None
      sa_block_kv = None

    with self.assertRaisesRegex(ValueError, "no splash_attention_kernel values"):
      workarounds.use_tokamax_splash_attention_patch(model, _BrokenSplash())

  def test_tokamax_patch_accepts_a_config_with_no_splash_fields(self):
    """An empty config means "use the kernel defaults" and must not raise.

    redacted's parity test passes a bare `class MockConfig: pass`. The guard above has to
    distinguish that from a real splash config read incorrectly - the discriminator is
    whether sa_-prefixed attributes exist at all. Caught by CI, not by me.
    """
    model, _ = self._tokamax_patch_fixture()

    class _NotASplashConfig:
      pass

    workarounds.use_tokamax_splash_attention_patch(model, _NotASplashConfig())

  def test_tokamax_patch_drops_kwargs_the_kernel_does_not_accept(self):
    """_SPLASH_KEYS is the union over splash implementations.

    tokamax's splash_sdpa accepts a strict subset - it has no
    use_document_aware_block_skipping - so forwarding the whole set made every run
    die at first compile with "Unexpected keyword arguments".
    """
    import inspect  # pylint: disable=g-import-not-at-top
    from torchtitan.experiments.tpu.kernels.tokamax_splash_attention import splash_sdpa  # pylint: disable=g-import-not-at-top

    accepted = set(inspect.signature(splash_sdpa).parameters)
    self.assertNotIn("use_document_aware_block_skipping", accepted)
    self.assertIn("segment_ids", accepted)
    # every key the patch may forward must be one the callee accepts
    self.assertContainsSubset(
        workarounds._SPLASH_KEYS & accepted, accepted,
        "the forwarded set must be a subset of the callee signature",
    )

  def test_tokamax_patch_forwards_segment_ids(self):
    """Packed-document segment IDs must reach the tokamax kernel.

    Regression test. The patch filters forward kwargs against tokamax
    splash_sdpa's signature, which used to lack segment_ids, so the IDs that
    GQAttention passes were silently dropped and packed documents attended
    across their boundaries. Also covers the positions-only fallback.
    """
    import functools  # pylint: disable=g-import-not-at-top
    import unittest.mock  # pylint: disable=g-import-not-at-top
    from torchtitan.experiments.tpu.kernels import tokamax_splash_attention  # pylint: disable=g-import-not-at-top

    captured = []
    real_splash_sdpa = tokamax_splash_attention.splash_sdpa

    # functools.wraps keeps the real signature visible to the patch's filter.
    @functools.wraps(real_splash_sdpa)
    def capture(q, k, v, *, segment_ids=None, **kwargs):
      del k, v, kwargs
      captured.append(segment_ids)
      return torch.zeros_like(q)

    # The patch writes the process-global TokamaxConfig; restore it so tests
    # that assert its class defaults are order-independent.
    config_cls = tokamax_splash_attention.TokamaxConfig
    saved = {k: v for k, v in vars(config_cls).items() if not k.startswith("_")}
    self.addCleanup(lambda: [setattr(config_cls, k, v) for k, v in saved.items()])

    model, sa = self._tokamax_patch_fixture()
    with unittest.mock.patch.object(
        tokamax_splash_attention, "splash_sdpa", capture
    ):
      workarounds.use_tokamax_splash_attention_patch(model, sa)

    q = k = v = torch.zeros(2, 128, 2, 8)
    positions = torch.cat([torch.arange(64), torch.arange(64)]).expand(2, -1)
    expected = torch.cat(
        [torch.zeros(64, dtype=torch.int32), torch.ones(64, dtype=torch.int32)]
    ).expand(2, -1)
    # A transposed copy: same values, non-contiguous strides.
    non_contiguous = expected.t().contiguous().t()
    self.assertFalse(non_contiguous.is_contiguous())

    model.attn(q, k, v, segment_ids=non_contiguous, positions=positions)
    model.attn(q, k, v, positions=positions)
    model.attn(q, k, v)

    self.assertLen(captured, 3)
    torch.testing.assert_close(captured[0], expected)
    self.assertTrue(captured[0].is_contiguous())
    torch.testing.assert_close(captured[1], expected)
    self.assertIsNone(captured[2])

  def test_tokamax_patch_warns_on_unsupported_block_skipping(self):
    """Requesting document-aware block skipping must not be silently dropped."""
    from torchtitan.experiments.tpu.kernels import tokamax_splash_attention  # pylint: disable=g-import-not-at-top

    config_cls = tokamax_splash_attention.TokamaxConfig
    saved = {k: v for k, v in vars(config_cls).items() if not k.startswith("_")}
    self.addCleanup(lambda: [setattr(config_cls, k, v) for k, v in saved.items()])

    model, sa = self._tokamax_patch_fixture()
    self.assertTrue(sa.sa_use_document_aware_block_skipping, "precondition")
    with self.assertLogs(workarounds.logger, level="WARNING") as logs:
      workarounds.use_tokamax_splash_attention_patch(model, sa)
    self.assertTrue(
        any("use_document_aware_block_skipping" in m for m in logs.output)
    )

    model, sa = self._tokamax_patch_fixture()
    sa.sa_use_document_aware_block_skipping = False
    with self.assertNoLogs(workarounds.logger, level="WARNING"):
      workarounds.use_tokamax_splash_attention_patch(model, sa)


if __name__ == "__main__":
  absltest.main()
