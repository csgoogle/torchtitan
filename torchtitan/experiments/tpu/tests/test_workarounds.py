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


if __name__ == "__main__":
  absltest.main()
