"""The joint trace must retain GPT-OSS gradients and routing-buffer updates."""

import copy
import dataclasses
from unittest import mock

from absl.testing import absltest, parameterized
import torch
import torch.nn.functional as F

from torchtitan.components.optimizer.optimizer import (
    default_adamw,
    register_moe_load_balancing_hook,
)

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.experiments.tpu.gpt_oss.tests.test_gpt_oss import (
    _Bf16GroupedMmPrecision,
    assert_gradients,
    tiny_config,
)
from torchtitan.experiments.tpu.graph_trainer.gpt_oss import GraphTrainerGptOssModel
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len


class TestGptOssGraphTrainer(BaseAcceleratorDeviceTest):
    @parameterized.named_parameters(("traced", False), ("compiled", True))
    def test_joint_trace_updates(self, compiled):
        if compiled and self.accelerator_device.type != "tpu":
            self.skipTest("Validates the TPU joint-graph compiler")
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("medium" if compiled else "highest")
        self.addCleanup(torch.set_float32_matmul_precision, previous)
        if not compiled:
            self.enter_context(_Bf16GroupedMmPrecision())
        torch.manual_seed(42)
        cfg = tiny_config()
        cfg = GraphTrainerGptOssModel.Config(
            **{
                field.name: getattr(cfg, field.name)
                for field in dataclasses.fields(cfg)
            }
        )
        eager = cfg.build().train()
        eager.init_states(buffer_device=torch.device("cpu"))
        traced_model = copy.deepcopy(eager).to(self.accelerator_device)
        eager = eager.to(self.accelerator_device)
        optimizers = [
            default_adamw(lr=1e-4, implementation="foreach").build(model_parts=[m])
            for m in (eager, traced_model)
        ]
        parallel_dims = mock.Mock(ep_enabled=False)
        parallel_dims.get_optional_mesh.return_value = None
        for optimizer, model in zip(optimizers, (eager, traced_model), strict=True):
            register_moe_load_balancing_hook(optimizer, [model], parallel_dims)
        tokens = torch.randint(0, 2048, (32,)).to(self.accelerator_device)
        labels = torch.randint(0, 2048, (32,)).to(self.accelerator_device)
        positions = torch.tensor(list(range(6)) + list(range(3)) + list(range(7)))
        positions = positions.repeat(2).to(self.accelerator_device)

        def step(inputs, targets, positions):
            loss = F.cross_entropy(traced_model(inputs, positions).float(), targets)
            grads = torch.autograd.grad(loss, tuple(traced_model.parameters()))
            return loss, grads

        with set_unflatten_seq_len(16):
            traced = minimal_fx_tracer(step, module=traced_model)(
                tokens, labels, positions
            )
            if compiled:
                from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
                    tpu_backend_compile_pass,
                )

                traced.gm = tpu_backend_compile_pass(traced.gm, traced.example_inputs)
            run = run_traced(traced, module=traced_model)
            for _ in range(3):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                expected = F.cross_entropy(eager(tokens, positions).float(), labels)
                expected.backward()
                actual, grads = run(tokens, labels, positions)
                # Use the baseline full-model reference bound here. The
                # stricter ~1e-4 trainer-trajectory gate is recorded separately
                # from this trace/gradient/buffer regression.
                torch.testing.assert_close(
                    actual.cpu(), expected.cpu(), atol=5e-3, rtol=2e-3
                )
                for parameter, grad in zip(
                    traced_model.parameters(), grads, strict=True
                ):
                    parameter.grad = grad
                assert_gradients(
                    self,
                    {n: p.grad for n, p in traced_model.named_parameters()},
                    {n: p.grad.cpu() for n, p in eager.named_parameters()},
                )
                # A trace that drops or double-applies mutation is numerically
                # wrong even if the current loss still matches.
                for left, right in zip(
                    eager.layers.values(), traced_model.layers.values()
                ):
                    torch.testing.assert_close(
                        left.moe.tokens_per_expert_E.cpu(),
                        right.moe.tokens_per_expert_E.cpu(),
                        atol=0,
                        rtol=0,
                    )
                    self.assertEqual(right.moe.tokens_per_expert_E.sum().item(), 32 * 4)
                for optimizer in optimizers:
                    optimizer.step()
                for left, right in zip(
                    eager.layers.values(), traced_model.layers.values()
                ):
                    self.assertEqual(right.moe.tokens_per_expert_E.sum().item(), 0)
                    self.assertGreater(right.moe.expert_bias_E.abs().sum().item(), 0)
                    torch.testing.assert_close(
                        left.moe.expert_bias_E.cpu(),
                        right.moe.expert_bias_E.cpu(),
                        atol=0,
                        rtol=0,
                    )


if __name__ == "__main__":
    absltest.main()
