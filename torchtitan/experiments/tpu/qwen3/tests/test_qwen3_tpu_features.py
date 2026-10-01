"""Single-chip numerics checks for the TPU features the Qwen3 MoE recipes enable.

Each case trains the small Qwen3 MoE twice on one TPU chip, from the same
initial weights and the same batches: once without a feature and once with it.
The second run's loss, gradients and parameters are checked step by step
against the first. Comparing TPU against TPU isolates the feature: a CPU
reference would differ by TPU matmul precision alone, which is larger than
most of what these features change.

Every case also asserts that the feature actually ran (its kernel was
dispatched, or its module was built). A feature that silently does nothing
passes any parity check, which is how a dropped kernel or a dropped aux loss
gets through.

The reference model runs its experts through the Pallas grouped matmul, as the
235B recipe does, not through `torch._grouped_mm`: on TPU the latter leaves
the weight gradient of an expert that received no tokens uninitialized (see
`test_grouped_mm_zeroes_empty_expert_gradients`), which makes any run that
uses it nondeterministic and eventually NaN.

Features expected to be exact are checked at zero tolerance. Kernels that
reorder floating-point accumulation are checked at tolerances measured on this
model, with headroom.
"""

import collections
import copy
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.utils import _python_dispatch
import torchtitan.config
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
import torchtitan.experiments.tpu.train_minimal  # pylint: disable=unused-import
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExpertsConverter
from torchtitan.experiments.tpu.qwen3 import config_registry
from torchtitan.experiments.tpu.token_dispatcher import TpuTokenDispatcherConverter
from torchtitan.models.common.moe import MoE


_TRAIN_ARGS = [
    "--module=torchtitan.experiments.tpu.qwen3",
    "--config=qwen3_moe_testmodel",
    "--optimizer.implementation=foreach",
    "--hf_assets_path=tests/assets/tokenizer",
    "--dataloader.dataset_path=tests/assets/c4_test",
    "--dataloader.dataset=c4_test",
    "--training.seq_len=128",
    "--training.local_batch_size=2",
    "--training.steps=4",
    "--training.mixed_precision_param=float32",
    "--training.mixed_precision_reduce=float32",
]

_EXACT = dict(
    loss_atol=0.0, loss_rtol=0.0,
    grad_atol=0.0, grad_rtol=0.0,
    param_atol=0.0, param_rtol=0.0,
)

# For kernels that reorder floating-point accumulation. Parameters get a fixed
# bound rather than a measured one: AdamW moves a weight by about the learning
# rate whenever its gradient flips sign, and a gradient near zero flips under
# any reordering, so two correct runs can differ by up to 2 x lr x steps.
_REORDER = dict(
    loss_atol=1e-3, loss_rtol=0.0,
    grad_atol=5e-3, grad_rtol=1e-2,
    param_atol=1e-2, param_rtol=0.0,
)


def _config(*, paired=False, native=False):
  """The small Qwen3 MoE with Pallas grouped-matmul experts."""
  config = torchtitan.config.ConfigManager().parse_args(list(_TRAIN_ARGS))
  # `qwen3_moe_testmodel` hands out a module-level model config, and the
  # converters below rewrite it in place; without a copy, one case's feature
  # leaks into every case built after it.
  config.model_spec.model = TpuGroupedExpertsConverter.Config().build().convert(
      copy.deepcopy(config.model_spec.model)
  )
  for _, experts, _, _ in config.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.use_pallas_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 128
    if paired:
      # The 235B recipe's tilings, scaled to this model's 128-wide experts:
      # the kernels need 128-aligned K tiles, so W1/W3 and W2 get the same
      # tiling here, but W2 still goes through its own tiling argument.
      experts.use_pallas_paired_grouped_mm = True
      experts.pallas_grouped_mm_tgmm_tiling = (128, 128, 0)
      experts.pallas_grouped_mm_w2_tgmm_tiling = (128, 128, 0)
    experts.use_kernel_native_weight_layout = native
  return config


class _OpCounter(_python_dispatch.TorchDispatchMode):
  """Counts the non-aten ops a run dispatches, by name."""

  def __init__(self):
    super().__init__()
    self.counts = collections.Counter()

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    if func.namespace not in ("aten", "prim", "prims", "profiler"):
      self.counts[str(func)] += 1
    return func(*args, **(kwargs or {}))

  def count(self, prefix: str) -> int:
    return sum(n for name, n in self.counts.items() if name.startswith(prefix))


class Qwen3TpuFeaturesTest(base_device_test.BaseAcceleratorDeviceTest):
  """Each Qwen3 MoE TPU feature against the same model without it."""

  def setUp(self):
    super().setUp()
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("The features under test are TPU kernels.")

  def _compare(self, reference, feature, **tolerances):
    ops = _OpCounter()
    trainer = self._run_trainer_device_parity_test(
        reference,
        validate_job_config=feature,
        validate_device=self.accelerator_device,
        validate_context=ops,
        **tolerances,
    )
    return trainer, ops

  def test_token_dispatcher(self):
    """The TPU reorder/combine VJPs compute exactly what the stock ones do."""
    feature = _config()
    feature.model_spec.model = (
        TpuTokenDispatcherConverter.Config().build().convert(
            feature.model_spec.model
        )
    )
    _, ops = self._compare(_config(), feature, **_EXACT)
    # The stock combine scatters through this op; the TPU one does not.
    self.assertEqual(ops.count("torchtitan.deterministic_scatter_add"), 0)

  def test_paired_grouped_mm(self):
    """W1 and W3 in one paired kernel compute exactly what two calls do."""
    _, ops = self._compare(_config(), _config(paired=True), **_EXACT)
    self.assertGreater(ops.count("tt_experts.paired_grouped_mm"), 0)

  def test_kernel_native_weight_layout(self):
    """Storing W1/W3 as the kernel reads them changes no arithmetic.

    The native layout renames the expert weights (`w1_EFD` -> `w1_EDF`), so
    the per-parameter comparison only covers the parameters that keep their
    name; the loss covers the experts.
    """
    trainer, _ = self._compare(
        _config(paired=True), _config(paired=True, native=True), **_EXACT
    )
    names = {name.rsplit(".", 1)[-1] for name, _ in trainer.model.named_parameters()}
    self.assertIn("w1_EDF", names)

  @parameterized.named_parameters(
      dict(
          testcase_name="splash",
          option="use_splash_attention_kernel",
          op="pallas.splash_attn_fwd",
      ),
      dict(
          testcase_name="tokamax_splash",
          option="use_tokamax_splash_attention_kernel",
          op="pallas.tokamax_splash_attn_fwd",
      ),
  )
  def test_splash_attention(self, option, op):
    """Splash attention against the SDPA it replaces.

    The tokamax backward's vmap/scan choice (`use_vmap_bwd`) is not covered:
    it is read from a process-global at trace time and the compiled kernel is
    cached by shape, so a second case at the same shape reuses the first's
    backward.
    """
    feature = _config()
    setattr(feature.splash_attention_kernel, option, True)
    _, ops = self._compare(_config(), feature, **_REORDER)
    self.assertGreater(ops.count(op), 0, f"{op} never ran")

  def test_global_batch_aux_loss(self):
    """The 235B recipe's aux loss must reach the router's gradients.

    The aux term is not part of the reported loss, so step 1's loss matches;
    the run must differ from the reference in the router gate's gradient.
    """
    # The recipe also turns off bias-based balancing; do the same in the
    # reference so the aux loss is the only difference between the runs.
    reference = _config()
    for _, moe, _, _ in reference.model_spec.model.traverse(MoE.Config):
      moe.load_balance_coeff = None
    feature = _config()
    config_registry._use_global_batch_aux_loss(feature)  # pylint: disable=protected-access
    with self.assertRaisesRegex(AssertionError, r"Gradient \(.*router\.gate"):
      self._compare(reference, feature, **_EXACT)

  # TODO: remove the decorator once torch_tpu zeroes the ragged-contraction
  # output of empty groups (csrc/ops/grouped_mm/grouped_mm_aten_kernels.cc,
  # case 3). Until then this documents why the reference above avoids
  # `torch._grouped_mm`; it will report an unexpected success once fixed.
  @unittest.expectedFailure
  def test_grouped_mm_zeroes_empty_expert_gradients(self):
    """An expert that receives no tokens must get an all-zero weight gradient.

    Its slice of `torch._grouped_mm`'s weight gradient sums over no rows. On
    TPU it is left uninitialized: denormals, ~3e38 or NaN, varying run to run.
    """
    counts = [0, 128, 0, 128, 0, 0, 0, 0]
    offs = torch.cumsum(torch.tensor(counts), 0).to(torch.int32)
    for trial in range(10):
      torch.manual_seed(trial)
      x = torch.randn(sum(counts), 128, device=self.accelerator_device)
      w = torch.randn(len(counts), 128, 128).mul(0.02).to(self.accelerator_device)
      w.requires_grad_(True)
      out = torch._grouped_mm(
          x.bfloat16(),
          w.bfloat16().transpose(-2, -1),
          offs=offs.to(self.accelerator_device),
      )
      out.float().sum().backward()
      grad = w.grad.cpu()
      for expert, count in enumerate(counts):
        if count == 0:
          self.assertTrue(
              torch.equal(grad[expert], torch.zeros_like(grad[expert])),
              f"trial {trial}: expert {expert} received no tokens but has a"
              " non-zero weight gradient",
          )


if __name__ == "__main__":
  absltest.main()
