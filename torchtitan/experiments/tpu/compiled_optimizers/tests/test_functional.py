# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the in-graph functional AdamW step.

`optimizer_in_graph` traces the optimizer into the same graph as fwd+loss+bwd,
so the step has to be a pure function of its inputs: it may not mutate the
tensors it is handed, and it must compute exactly what `torch.optim.AdamW`
would. A drift here changes what the model converges to while every speed
number still looks fine, which makes it a semantics bug rather than a
performance one.

The separate-graph path (`compile_optimizer`) is covered by
`test_separate_graph.py`.

`functional_adamw_step` defaults to `capturable=True`, which torch only permits
on an accelerator, so the arithmetic tests run on the device rather than CPU.
"""

import copy

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.experiments.tpu.compiled_optimizers import functional


_LR, _WD, _EPS = 1e-2, 0.01, 1e-8


class FunctionalAdamWTest(BaseAcceleratorDeviceTest):
  """The in-graph step must equal `torch.optim.AdamW`, step for step."""

  def _model(self) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2)).to(
        self.accelerator_device
    )

  def _fresh_state(self, params):
    return (
        [torch.zeros_like(p) for p in params],
        [torch.zeros_like(p) for p in params],
        [
            torch.zeros((), dtype=torch.float32, device=self.accelerator_device)
            for _ in params
        ],
    )

  @parameterized.named_parameters(("foreach", True), ("per_tensor", False))
  def test_matches_torch_adamw_over_multiple_steps(self, foreach):
    """Five steps, not one: a wrong bias correction only diverges over time."""
    reference = self._model()
    subject = copy.deepcopy(reference)

    opt = torch.optim.AdamW(
        reference.parameters(), lr=_LR, weight_decay=_WD, eps=_EPS
    )
    params = list(subject.parameters())
    exp_avgs, exp_avg_sqs, steps = self._fresh_state(params)
    lr = torch.tensor(_LR, device=self.accelerator_device)

    for i in range(5):
      torch.manual_seed(100 + i)
      grads = [torch.randn_like(p) for p in params]

      opt.zero_grad()
      for p, g in zip(reference.parameters(), grads, strict=True):
        p.grad = g.clone()
      opt.step()

      params, exp_avgs, exp_avg_sqs, steps = functional.functional_adamw_step(
          params,
          grads,
          exp_avgs,
          exp_avg_sqs,
          steps,
          lr,
          weight_decay=_WD,
          eps=_EPS,
          foreach=foreach,
      )

      for j, (want, got) in enumerate(
          zip(reference.parameters(), params, strict=True)
      ):
        torch.testing.assert_close(
            got.cpu(), want.cpu(), rtol=1e-5, atol=1e-5,
            msg=f"param {j} diverged from torch AdamW at step {i + 1}",
        )

  def test_does_not_mutate_its_inputs(self):
    """Purity is the tracing contract: in-place writes would escape the graph."""
    params = list(self._model().parameters())
    exp_avgs, exp_avg_sqs, steps = self._fresh_state(params)
    grads = [torch.randn_like(p) for p in params]
    before = [t.clone() for t in (*params, *exp_avgs, *exp_avg_sqs, *steps)]

    functional.functional_adamw_step(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        steps,
        torch.tensor(_LR, device=self.accelerator_device),
    )

    after = (*params, *exp_avgs, *exp_avg_sqs, *steps)
    for i, (want, got) in enumerate(zip(before, after, strict=True)):
      torch.testing.assert_close(
          got.cpu(), want.cpu(), msg=f"input tensor {i} was modified in place"
      )

  def test_returns_fresh_tensors_not_aliases(self):
    params = list(self._model().parameters())
    exp_avgs, exp_avg_sqs, steps = self._fresh_state(params)
    grads = [torch.randn_like(p) for p in params]

    new_params, new_eas, _, _ = functional.functional_adamw_step(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        steps,
        torch.tensor(_LR, device=self.accelerator_device),
    )

    for old, new in zip(params, new_params, strict=True):
      self.assertNotEqual(old.data_ptr(), new.data_ptr())
    for old, new in zip(exp_avgs, new_eas, strict=True):
      self.assertNotEqual(old.data_ptr(), new.data_ptr())

  def test_step_counter_advances_by_one(self):
    params = list(self._model().parameters())
    exp_avgs, exp_avg_sqs, steps = self._fresh_state(params)
    grads = [torch.randn_like(p) for p in params]

    _, _, _, new_steps = functional.functional_adamw_step(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        steps,
        torch.tensor(_LR, device=self.accelerator_device),
    )

    for step in new_steps:
      self.assertEqual(step.item(), 1)

  def test_casts_grads_to_param_dtype(self):
    """Mixed precision feeds bf16 grads against fp32 params."""
    params = [torch.randn(4, 4, device=self.accelerator_device)]
    grads = [torch.randn(4, 4, device=self.accelerator_device).bfloat16()]
    exp_avgs, exp_avg_sqs, steps = self._fresh_state(params)

    new_params, _, _, _ = functional.functional_adamw_step(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        steps,
        torch.tensor(_LR, device=self.accelerator_device),
    )

    self.assertEqual(new_params[0].dtype, torch.float32)


class _Parametrized(nn.Module):
  """Stands in for simple_fsdp: the attribute getter returns a fresh tensor."""

  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(3))
    self.getter_calls = 0

  def __getattr__(self, name):
    if name == "weight":
      object.__setattr__(self, "getter_calls", self.getter_calls + 1)
      return self._parameters["weight"] * 2.0  # an all-gathered copy
    return super().__getattr__(name)


class ParamLookupTest(parameterized.TestCase):
  """`_get_param_by_fqn` must reach the real parameter, not a gathered copy."""

  def test_bypasses_the_parametrization_getter(self):
    """Writing through the getter's result would be silently discarded.

    simple_fsdp exposes each shard through a property that all-gathers into a
    fresh transient tensor. Copying the optimizer's output into that tensor
    updates nothing the next step will read.
    """
    module = _Parametrized()
    got = functional._get_param_by_fqn(module, "weight")  # pylint: disable=protected-access

    self.assertIs(got, module._parameters["weight"])
    self.assertEqual(module.getter_calls, 0, "the getter must not be invoked")

  def test_resolves_nested_fqns(self):
    model = nn.Sequential(nn.Linear(4, 4))
    got = functional._get_param_by_fqn(model, "0.weight")  # pylint: disable=protected-access
    self.assertIs(got, model[0].weight)


class BufferWritebackTest(parameterized.TestCase):
  """Graph outputs have to land in the live params and optstate buffers."""

  def _model_with_optstate(self):
    model = nn.Linear(3, 2, bias=False)
    model.register_buffer("_optstate_weight_exp_avg", torch.zeros(2, 3))
    model.register_buffer("_optstate_weight_exp_avg_sq", torch.zeros(2, 3))
    return model

  def test_writes_params_and_both_optstate_buffers(self):
    model = self._model_with_optstate()
    new_p = torch.full((2, 3), 0.5)
    new_ea = torch.full((2, 3), 0.25)
    new_eas = torch.full((2, 3), 0.125)

    functional.update_model_and_optstate_buffers(
        model, ["weight"], [new_p], [new_ea], [new_eas]
    )

    torch.testing.assert_close(model.weight.data, new_p)
    torch.testing.assert_close(model._optstate_weight_exp_avg, new_ea)
    torch.testing.assert_close(model._optstate_weight_exp_avg_sq, new_eas)

  def test_writes_in_place_so_buffer_identity_survives(self):
    """The graph holds these buffers by identity across steps."""
    model = self._model_with_optstate()
    param_ptr = model.weight.data_ptr()
    buffer_ptr = model._optstate_weight_exp_avg.data_ptr()

    functional.update_model_and_optstate_buffers(
        model,
        ["weight"],
        [torch.ones(2, 3)],
        [torch.ones(2, 3)],
        [torch.ones(2, 3)],
    )

    self.assertEqual(model.weight.data_ptr(), param_ptr)
    self.assertEqual(model._optstate_weight_exp_avg.data_ptr(), buffer_ptr)

  def test_rejects_mismatched_lengths(self):
    model = self._model_with_optstate()
    with self.assertRaises(ValueError):
      functional.update_model_and_optstate_buffers(
          model, ["weight"], [], [torch.ones(2, 3)], [torch.ones(2, 3)]
      )


if __name__ == "__main__":
  absltest.main()
