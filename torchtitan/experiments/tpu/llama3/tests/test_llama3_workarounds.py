from absl.testing import absltest
import torch
from torch import nn
import torchtitan.experiments.tpu.base_device_test as base_device_test
from torchtitan.experiments.tpu import test_utils
from torchtitan.models.llama3.model import model as llama3_model
from torchtitan.experiments.tpu.llama3.model import workarounds
workarounds.apply_patch()  # Manually apply the patch to the model module to test


class Llama3WorkaroundsTest(base_device_test.BaseAcceleratorDeviceTest):

  def _get_model_args(self):
    return llama3_model.TransformerModelArgs(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=128,
        max_seq_len=16,
        multiple_of=16,
    )

  def test_patch_numerics_match_reference(self):
    """Verifies that the sin/cos patch is numerically identical to complex number math."""
    batch, seq_len, n_heads, head_dim = 2, 8, 4, 16
    xq = torch.randn(
        batch, seq_len, n_heads, head_dim, device=self.accelerator_device
    )
    xk = torch.randn(
        batch, seq_len, n_heads, head_dim, device=self.accelerator_device
    )

    # create frequencies using the original helper
    args = self._get_model_args()
    freqs_cis = llama3_model.precompute_freqs_cis(
        head_dim, seq_len, args.rope_theta
    ).to(self.accelerator_device)

    out_q_patched, out_k_patched = llama3_model.apply_rotary_emb(xq, xk, freqs_cis)

    # run original logic from apply_rotary_emb
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_broad = llama3_model.reshape_for_broadcast(freqs_cis, xq_)
    out_q_ref = torch.view_as_real(xq_ * freqs_cis_broad).flatten(3).type_as(xq)
    out_k_ref = torch.view_as_real(xk_ * freqs_cis_broad).flatten(3).type_as(xk)

    test_utils.check_equivalence(
        out_q_patched,
        out_q_ref,
        atol=1e-3,
        rtol=1e-3,
        check_name="RoPE Patched Q vs Reference",
        test_label="Patched",
        ref_label="ComplexRef",
    )

    test_utils.check_equivalence(
        out_k_patched,
        out_k_ref,
        atol=1e-3,
        rtol=1e-3,
        check_name="RoPE Patched K vs Reference",
        test_label="Patched",
        ref_label="ComplexRef",
    )

  def test_forward_backward_no_crash(self):
    """Runs a forward and backward pass on the device using patched model."""

    args = self._get_model_args()
    model = llama3_model.Transformer(args).to(self.accelerator_device)

    batch, seq_len = 2, 8
    tokens = torch.randint(
        0, args.vocab_size, (batch, seq_len), device=self.accelerator_device
    )

    output = model(tokens)
    loss = output.sum()
    loss.backward()

    # additional sanity check: ensure gradients are actually populated
    first_layer_wq = model.layers["0"].attention.wq.weight
    self.assertIsNotNone(first_layer_wq.grad, "Gradients were not computed")
    self.assertTrue(
        torch.is_tensor(first_layer_wq.grad), "Gradient is not a tensor"
    )


if __name__ == "__main__":
  absltest.main()
