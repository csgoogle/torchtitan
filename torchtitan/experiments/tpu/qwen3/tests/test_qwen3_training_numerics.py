"""Qwen3 MoE trained on TPU the way the 235B recipe runs it, against a CPU reference.

The small Qwen3 MoE is trained for a few steps on one TPU chip with the
recipe's TPU kernels and model options, then the same initial weights and
batches are replayed on CPU with the stock implementations. Forward (the loss),
backward (every gradient) and the optimizer update (every parameter) are
compared at each step.

What the TPU run enables, and the CPU reference replaces with stock code:

- experts: Pallas paired grouped matmul (W1/W3 in one kernel, W2 on its own
  dWeight tiling) with kernel-native weight layout
- token dispatch: the TPU reorder/combine VJPs
- attention: tokamax splash attention

Both runs use the recipe's global-batch auxiliary load-balancing loss.

The per-kernel tests own each kernel's exact numerics; this test checks that
the model they make up trains like the stock model.
"""

import collections
import copy

from absl.testing import absltest
import torch
from torch.utils import _python_dispatch
import torchtitan.config
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import numerical_validation
import torchtitan.experiments.tpu.train_minimal  # pylint: disable=unused-import
from torchtitan.experiments.tpu.grouped_experts import _CANONICAL_TO_NATIVE_WEIGHT
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExpertsConverter
from torchtitan.experiments.tpu.qwen3 import config_registry
from torchtitan.experiments.tpu.token_dispatcher import TpuTokenDispatcherConverter


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

_NATIVE_TO_CANONICAL_WEIGHT = {
    native: canonical for canonical, native in _CANONICAL_TO_NATIVE_WEIGHT.items()
}


def _stock_config():
  config = torchtitan.config.ConfigManager().parse_args(list(_TRAIN_ARGS))
  # `qwen3_moe_testmodel` hands out a module-level model config; copy it so
  # the converters below cannot leak into other configs.
  config.model_spec.model = copy.deepcopy(config.model_spec.model)
  config_registry._use_global_batch_aux_loss(config)  # pylint: disable=protected-access
  return config


def _tpu_recipe_config():
  """The stock config with the 235B recipe's TPU kernels and model options."""
  config = _stock_config()
  model = TpuTokenDispatcherConverter.Config().build().convert(
      config.model_spec.model
  )
  model = TpuGroupedExpertsConverter.Config().build().convert(model)
  for _, experts, _, _ in model.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    experts.use_pallas_paired_grouped_mm = True
    # The recipe's tile_m (256) and K tiles (512/2048 for W1/W3, 768 for W2)
    # exceed this model's 128-wide experts; the kernels need 128-aligned K
    # tiles, so both get (128, 128, 0) here.
    experts.pallas_grouped_mm_tile_m = 128
    experts.pallas_grouped_mm_tgmm_tiling = (128, 128, 0)
    experts.pallas_grouped_mm_w2_tgmm_tiling = (128, 128, 0)
    experts.use_kernel_native_weight_layout = True
  config.model_spec.model = model
  config.splash_attention_kernel.use_tokamax_splash_attention_kernel = True
  return config


def _capture_canonical_state(model, loss):
  """Captures state with kernel-native expert weights in the stock layout."""
  state = numerical_validation.capture_local_state(model, loss)
  for kind in ("params", "grads"):
    for name in list(state[kind]):
      prefix, _, leaf = name.rpartition(".")
      if leaf in _NATIVE_TO_CANONICAL_WEIGHT:
        state[kind][f"{prefix}.{_NATIVE_TO_CANONICAL_WEIGHT[leaf]}"] = (
            state[kind].pop(name).transpose(-2, -1).contiguous()
        )
  return state


class _OpCounter(_python_dispatch.TorchDispatchMode):
  """Counts the non-aten ops a run dispatches, by name."""

  def __init__(self):
    super().__init__()
    self.counts = collections.Counter()

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    if func.namespace not in ("aten", "prim", "prims", "profiler"):
      self.counts[str(func)] += 1
    return func(*args, **(kwargs or {}))


class Qwen3TrainingNumericsTest(base_device_test.BaseAcceleratorDeviceTest):
  """Qwen3 MoE recipe training on TPU against the stock model on CPU."""

  def setUp(self):
    super().setUp()
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("The TPU run uses Pallas kernels.")

  def test_moe_recipe_training_matches_cpu(self):
    ops = _OpCounter()
    self._run_trainer_device_parity_test(
        _tpu_recipe_config(),
        cpu_job_config=_stock_config(),
        device_capture_fn=_capture_canonical_state,
        device_context=ops,
        # Measured (deterministic run to run): loss <= 9e-5, gradients
        # <= 2.4e-3, parameters <= 5.7e-3.
        loss_atol=5e-4, loss_rtol=0.0,
        grad_atol=5e-3, grad_rtol=1e-2,
        # AdamW moves a weight by about the learning rate whenever its
        # gradient flips sign, which a near-zero gradient does under any
        # reordering: two correct runs can differ by up to 2 x lr x steps
        # (6.4e-3 here), so parameters only catch gross errors.
        param_atol=1e-2, param_rtol=0.0,
    )
    # A kernel that silently fell back to the stock op would still pass the
    # comparison, so check that the TPU run used the recipe's kernels.
    for op in (
        "tt_experts.paired_grouped_mm",
        "pallas.tokamax_splash_attn_fwd",
    ):
      self.assertTrue(
          any(name.startswith(op) for name in ops.counts), f"{op} never ran"
      )



if __name__ == "__main__":
  absltest.main()
