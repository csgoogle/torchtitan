"""Device tests for the HF backend's Pallas expert-kernel selection.

Checks that the shape-aware choice between the paired and the K/N-tiled
unpaired kernels compiles within scoped VMEM at real expert widths (forcing
paired at the Mixtral shape fails to compile), and that
batches whose routed-row count is not a tile_m multiple are padded correctly.
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.transformers_modeling_backend import model as hf_model


def _reference(x, w1, w2, w3, group_sizes):
  """FP32 per-expert SwiGLU over native (E, K, N) weights."""
  outs, start = [], 0
  for e, size in enumerate(group_sizes.tolist()):
    xe = x[start:start + size]
    outs.append((F.silu(xe @ w1[e]) * (xe @ w3[e])) @ w2[e])
    start += size
  return torch.cat(outs)


def _rel_err(actual, expected):
  return ((actual.float() - expected).norm() / expected.norm()).item()


class PallasExpertsTest(base_device_test.BaseAcceleratorDeviceTest):

  @parameterized.named_parameters(
      # (dim, hidden, group sizes, expect paired)
      ("qwen3_30b", 2048, 768, [0, 300, 100, 112], True),
      ("olmoe", 2048, 1024, [256, 0, 200, 56], True),
      # Largest paired shape (29.5 MiB estimated).
      ("deepseek_v2_lite", 2048, 1408, [128, 128, 0, 256], True),
      ("mixtral_8x7b", 4096, 14336, [100, 156], False),
      # 16 tokens x top-k 8 = 128 rows: below one 256-row tile.
      ("short_batch_paired", 2048, 768, [16, 0, 40, 8, 24, 16, 8, 16], True),
      ("short_batch_unpaired", 1024, 8192, [64, 0, 64], False),
  )
  def test_forward_backward(self, dim, hidden, group_sizes, expect_paired):
    device = self.accelerator_device
    cfg = TpuGroupedExperts.Config(dim=dim, hidden_dim=hidden, num_experts=len(group_sizes))
    self.assertEqual(hf_model._configure_pallas_experts(cfg), expect_paired)
    experts = TpuGroupedExperts(cfg)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
      for name, k in (("w1_EDF", dim), ("w2_EFD", hidden), ("w3_EDF", dim)):
        param = getattr(experts, name)
        param.copy_(torch.randn(param.shape, generator=gen) / k**0.5)
    experts = experts.to(device=device, dtype=torch.bfloat16)

    rows = sum(group_sizes)
    x_cpu = torch.randn(rows, dim, generator=gen).bfloat16()
    dout_cpu = torch.randn(rows, dim, generator=gen).bfloat16()
    sizes = torch.tensor(group_sizes, dtype=torch.int32)

    x = x_cpu.to(device).requires_grad_()
    # Compiled: eager Pallas calls do not enforce the scoped VMEM limit.
    out = torch.compile(experts, backend="tpu")(x, sizes.to(device))
    params = (experts.w1_EDF, experts.w2_EFD, experts.w3_EDF)
    grads = torch.autograd.grad(out, (x, *params), dout_cpu.to(device))

    ref_inputs = [x_cpu.float().requires_grad_()] + [
        p.detach().cpu().float().requires_grad_() for p in params
    ]
    x_ref, w1, w2, w3 = ref_inputs
    ref = _reference(x_ref, w1, w2, w3, sizes)
    ref_grads = torch.autograd.grad(ref, ref_inputs, dout_cpu.float())

    self.assertEqual(tuple(out.shape), (rows, dim))
    self.assertLess(_rel_err(out.cpu(), ref.detach()), 1e-2)
    for name, g, rg in zip(("dx", "dw1", "dw2", "dw3"), grads, ref_grads):
      self.assertLess(_rel_err(g.cpu(), rg), 2e-2, name)


if __name__ == "__main__":
  absltest.main()
