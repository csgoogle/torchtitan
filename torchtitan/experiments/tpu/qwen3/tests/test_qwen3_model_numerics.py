"""Qwen3 MoE model numerics on TPU with each recipe feature enabled.

Builds the small Qwen3 MoE once and runs one batch through the same weights in
several configurations, comparing what the model produces: the logits
(forward), the loss, and the parameter gradients (backward).

- `test_stock_matches_cpu`: the stock model on TPU against the stock model on
  CPU in float32, anchoring the TPU model to a ground truth. Its tolerance is
  TPU's default float32 matmul precision (bfloat16 inputs).
- `test_feature_matches_stock`: each feature the Qwen3 MoE recipes use, alone
  and all together as the 235B recipe runs them, against the stock model on
  TPU: same device and precision, so a numerics problem shows up against the
  feature that caused it.

Outputs are compared as wholes -- the loss, and the relative error norm of the
logits and of all gradients together -- not element by element. In an MoE a
perturbation as small as 0.1% anywhere upstream flips some tokens' expert
choice, and those few tokens dominate any elementwise maximum: a 0.1% error
and a swap to a different, correct attention kernel give the same maximum
logit difference. Whole-tensor norms and the loss average the flips out.

Resolution, measured by scaling a kernel's output: expert kernels are
bit-identical to the stock grouped matmul in the forward pass, so their cases
catch an error of 0.001%; attention kernels legitimately differ from TPU's
stock SDPA by about as much as TPU differs from CPU, so their cases catch a 1%
error but not 0.1%.

Every configuration also checks that its kernel ran -- a silent fallback to
the stock op would otherwise match the reference trivially -- and that every
output is finite. All of them use the recipe's global-batch auxiliary
load-balancing loss, which reaches the router's gradients.
"""

import collections
import contextlib
import copy

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.nn import functional as F
from torch.utils import _python_dispatch
import torchtitan.config
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import train_minimal
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
    "--training.mixed_precision_param=float32",
    "--training.mixed_precision_reduce=float32",
]

_NATIVE_TO_CANONICAL_WEIGHT = {
    native: canonical
    for canonical, native in _CANONICAL_TO_NATIVE_WEIGHT.items()
}


def _config():
  config = torchtitan.config.ConfigManager().parse_args(list(_TRAIN_ARGS))
  # `qwen3_moe_testmodel` hands out a module-level model config; copy it so
  # one case's converters cannot leak into the next.
  config.model_spec.model = copy.deepcopy(config.model_spec.model)
  config_registry._use_global_batch_aux_loss(config)  # pylint: disable=protected-access
  return config


def _token_dispatcher(config):
  converter = TpuTokenDispatcherConverter.Config().build()
  config.model_spec.model = converter.convert(config.model_spec.model)


def _pallas_experts(config, *, paired=False):
  """Pallas grouped-matmul experts; `paired` adds the recipe's options."""
  config.model_spec.model = TpuGroupedExpertsConverter.Config().build().convert(
      config.model_spec.model
  )
  for _, experts, _, _ in config.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.use_pallas_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 128
    if paired:
      experts.use_pallas_paired_grouped_mm = True
      # The recipe's tiles exceed this model's 128-wide experts, and the
      # kernels need 128-aligned K tiles, so W1/W3 and W2 both get 128 here.
      experts.pallas_grouped_mm_tgmm_tiling = (128, 128, 0)
      experts.pallas_grouped_mm_w2_tgmm_tiling = (128, 128, 0)
      experts.use_kernel_native_weight_layout = True


def _splash(config):
  config.splash_attention_kernel.use_splash_attention_kernel = True


def _tokamax_splash(config):
  config.splash_attention_kernel.use_tokamax_splash_attention_kernel = True


def _recipe(config):
  """Everything the 235B recipe enables that this model can run."""
  _token_dispatcher(config)
  _pallas_experts(config, paired=True)
  _tokamax_splash(config)


# Tolerances, from measured differences (bit-identical run to run): loss as an
# absolute difference, logits and gradients as relative error norms.
_EXACT = dict(loss_atol=0.0, logits_rtol=0.0, grads_rtol=0.0)
# Pallas expert kernels: forward bit-identical to the stock grouped matmul;
# gradients 1.0e-3. Scaling the expert output by 1.00001 moves the logits by
# 1.3e-3.
_EXPERTS = dict(loss_atol=1e-6, logits_rtol=1e-6, grads_rtol=3e-3)
# Attention kernels against TPU's stock SDPA: loss 1.2e-4, logits 9.2e-3,
# gradients 6.7e-3 (splash; tokamax is closer). Scaling the tokamax output by
# 1.01 moves the loss by 5.0e-4 and the gradients by 1.45e-2.
_ATTENTION = dict(loss_atol=2.5e-4, logits_rtol=1.5e-2, grads_rtol=1.2e-2)
# Stock TPU against CPU float32: loss 1.6e-4, logits 1.1e-2, gradients 8.9e-3.
_CPU = dict(loss_atol=4e-4, logits_rtol=2e-2, grads_rtol=2e-2)


def _relative_error(actual, expected):
  """||actual - expected|| / ||expected||."""
  error = torch.linalg.vector_norm(actual - expected)
  return (error / torch.linalg.vector_norm(expected)).item()


class _OpCounter(_python_dispatch.TorchDispatchMode):
  """Counts the non-aten ops dispatched while active, by name."""

  def __init__(self):
    super().__init__()
    self.counts = collections.Counter()

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    if func.namespace not in ("aten", "prim", "prims", "profiler"):
      self.counts[str(func)] += 1
    return func(*args, **(kwargs or {}))

  def ran(self, prefix: str) -> bool:
    return any(name.startswith(prefix) for name in self.counts)


def _forward_backward(model, tokens, labels):
  """Logits, loss, and canonical-layout parameter gradients for one batch."""
  model.zero_grad(set_to_none=True)
  logits = model(tokens)
  loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten())
  loss.backward()
  grads = {}
  for name, param in model.named_parameters():
    grad = param.grad.detach().cpu()
    prefix, _, leaf = name.rpartition(".")
    if leaf in _NATIVE_TO_CANONICAL_WEIGHT:
      name = f"{prefix}.{_NATIVE_TO_CANONICAL_WEIGHT[leaf]}"
      grad = grad.transpose(-2, -1)
    grads[name] = grad
  return logits.detach().cpu(), loss.detach().cpu(), grads


class Qwen3ModelNumericsTest(base_device_test.BaseAcceleratorDeviceTest):
  """The Qwen3 MoE's outputs and gradients with each TPU feature enabled."""

  def setUp(self):
    super().setUp()
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("The configurations under test are TPU kernels.")
    trainer = train_minimal.TrainerMinimal(
        torch.device("cpu"), rank=0, world_size=1, job_config=_config()
    )
    self.cpu_model = trainer.model
    batch, self.labels = next(iter(trainer.dataloader))
    self.tokens = batch["input"]

  def _outputs(self, config, device, counter=None):
    """Logits, loss and gradients of `config` on `device`, shared weights."""
    model = train_minimal.TrainerMinimal(
        device, rank=0, world_size=1, job_config=config
    ).model
    # Canonical names and layout in; kernel-native layouts convert on load.
    model.load_state_dict(self.cpu_model.state_dict())
    with counter or contextlib.nullcontext():
      return _forward_backward(
          model, self.tokens.to(device), self.labels.to(device)
      )

  def _check(self, actual, expected, *, loss_atol, logits_rtol, grads_rtol):
    logits, loss, grads = actual
    expected_logits, expected_loss, expected_grads = expected
    self.assertEqual(grads.keys(), expected_grads.keys())
    names = sorted(grads)
    grads = torch.cat([grads[name].flatten() for name in names])
    expected_grads = torch.cat(
        [expected_grads[name].flatten() for name in names]
    )
    for name, tensor in (
        ("logits", logits), ("loss", loss), ("gradients", grads)
    ):
      self.assertTrue(torch.isfinite(tensor).all(), f"non-finite {name}")
    self.assertLessEqual((loss - expected_loss).abs().item(), loss_atol, "loss")
    self.assertLessEqual(
        _relative_error(logits, expected_logits), logits_rtol, "logits"
    )
    self.assertLessEqual(
        _relative_error(grads, expected_grads), grads_rtol, "gradients"
    )

  def test_stock_matches_cpu(self):
    self._check(
        self._outputs(_config(), self.accelerator_device),
        self._outputs(_config(), torch.device("cpu")),
        **_CPU,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="token_dispatcher",
          enable=_token_dispatcher,
          ops=(),
          tolerances=_EXACT,
      ),
      dict(
          testcase_name="pallas_grouped_mm",
          enable=_pallas_experts,
          ops=("tt_experts.grouped_mm",),
          tolerances=_EXPERTS,
      ),
      dict(
          testcase_name="paired_grouped_mm_native_layout",
          enable=lambda config: _pallas_experts(config, paired=True),
          ops=("tt_experts.paired_grouped_mm",),
          tolerances=_EXPERTS,
      ),
      dict(
          testcase_name="splash",
          enable=_splash,
          ops=("pallas.splash_attn_fwd",),
          tolerances=_ATTENTION,
      ),
      dict(
          testcase_name="tokamax_splash",
          enable=_tokamax_splash,
          ops=("pallas.tokamax_splash_attn_fwd",),
          tolerances=_ATTENTION,
      ),
      dict(
          testcase_name="recipe",
          enable=_recipe,
          ops=(
              "tt_experts.paired_grouped_mm",
              "pallas.tokamax_splash_attn_fwd",
          ),
          tolerances=_ATTENTION,
      ),
  )
  def test_feature_matches_stock(self, enable, ops, tolerances):
    config = _config()
    enable(config)
    counter = _OpCounter()
    actual = self._outputs(config, self.accelerator_device, counter)
    for op in ops:
      self.assertTrue(counter.ran(op), f"{op} never ran")
    self._check(
        actual,
        self._outputs(_config(), self.accelerator_device),
        **tolerances,
    )


if __name__ == "__main__":
  absltest.main()
