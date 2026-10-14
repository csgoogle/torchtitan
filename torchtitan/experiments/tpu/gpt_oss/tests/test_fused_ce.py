"""GPT-OSS fused LM-head loss and gradient parity against fp32 cross entropy."""

from absl.testing import absltest, parameterized
import torch
import torch.nn.functional as F

from torchtitan.config import ConfigManager
from torchtitan.config.override import apply_overrides
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.experiments.tpu.gpt_oss.config_registry import (
    gpt_oss_debugmodel_fused_ce,
)
from torchtitan.experiments.tpu.graph_trainer.config import to_tpu_graph_trainer_config
from torchtitan.experiments.tpu.graph_trainer.gpt_oss import model_registry
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.overrides.fused_loss import FusedLossLinearWrapper


class TestFusedCE(BaseAcceleratorDeviceTest):
    def test_recipe_and_checkpoint_contract(self):
        config = ConfigManager().parse_args(
            [
                "--module=torchtitan.experiments.tpu.gpt_oss",
                "--config=gpt_oss_debugmodel_fused_ce",
            ]
        )
        graph_config = to_tpu_graph_trainer_config(
            config, model_registry, target_cls=TpuGraphTrainer.Config
        )
        original = graph_config.model_spec.model.lm_head
        self.assertLen(apply_overrides(graph_config.override, graph_config), 5)
        head = graph_config.model_spec.model.lm_head
        self.assertIsInstance(head, FusedLossLinearWrapper.Config)
        self.assertIsInstance(graph_config.loss, PallasCrossEntropyLoss.Config)
        self.assertIsNone(graph_config.loss.soft_cap)
        self.assertEqual(graph_config.loss.output_multiplier, 1.0)
        self.assertFalse(head.bias)
        with torch.device("meta"):
            reference, fused = original.build(), head.build()
        self.assertEqual(set(reference.state_dict()), set(fused.state_dict()))
        self.assertEqual(reference.weight.shape, fused.weight.shape)
        self.assertEqual(head.param_init, original.param_init)

    @parameterized.named_parameters(
        ("fp32", torch.float32, "none"),
        ("bf16", torch.bfloat16, "none"),
        ("fp32_ignored", torch.float32, "mixed"),
        ("bf16_ignored", torch.bfloat16, "mixed"),
        ("all_ignored", torch.float32, "all"),
    )
    def test_loss_and_gradients(self, dtype, ignored):
        if self.accelerator_device.type != "tpu":
            self.skipTest("Mosaic fused cross entropy requires a TPU")
        torch.manual_seed(42)
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        self.addCleanup(torch.set_float32_matmul_precision, previous)
        config = gpt_oss_debugmodel_fused_ce()
        head_config = config.model_spec.model.lm_head
        apply_overrides(config.override, config)
        head = config.model_spec.model.lm_head.build().to(
            device=self.accelerator_device, dtype=dtype
        )
        # Perturbed, nonzero head weights and two batch rows test flattening too.
        with torch.no_grad():
            head.weight.copy_(
                torch.randn_like(head.weight) / head_config.in_features**0.5
            )
        hidden = torch.randn(
            2,
            256,
            head_config.in_features,
            device=self.accelerator_device,
            dtype=dtype,
            requires_grad=True,
        )
        ref_hidden = hidden.detach().float().requires_grad_(True)
        ref_weight = head.weight.detach().float().requires_grad_(True)
        labels = torch.randint(head_config.out_features, (2, 256), device=hidden.device)
        if ignored == "mixed":
            labels[:, ::3] = -100
        elif ignored == "all":
            labels.fill_(-100)
        # Emulate a global batch containing twice this rank's valid tokens.
        denominator = (2 * (labels != -100).sum()).clamp_min(1)
        expected = (
            F.cross_entropy(
                F.linear(ref_hidden, ref_weight).flatten(0, 1),
                labels.flatten(),
                reduction="sum",
                ignore_index=-100,
            )
            / denominator
        )
        result, _ = config.loss.build()(head(hidden), labels, denominator)
        torch.testing.assert_close(result.reshape(()), expected, rtol=2e-4, atol=2e-5)
        result.backward()
        expected.backward()
        errors = {}
        for name, actual, target in (
            ("hidden", hidden.grad, ref_hidden.grad),
            ("head.weight", head.weight.grad, ref_weight.grad),
        ):
            self.assertIsNotNone(actual)
            if ignored == "all":
                self.assertEqual(actual.count_nonzero().item(), 0)
                continue
            relative = (
                lambda a: (a.float() - target).norm().item() / target.norm().item()
            )
            errors[name] = relative(actual)
            self.assertLessEqual(errors[name], 1e-2, name)
            # Reject omitted, sign-flipped, and incorrectly normalized gradients.
            for bad in (torch.zeros_like(actual), -actual, actual * 2):
                self.assertGreater(relative(bad), 0.5)
        if ignored != "all":
            self.assertGreater(abs((2 * result - expected).item()), 1.0)
        self.assertEqual(hidden.grad[labels == -100].count_nonzero().item(), 0)
        print(
            "FUSED_CE_PARITY",
            dtype,
            ignored,
            "loss",
            result.item(),
            "reference",
            expected.item(),
            "gradients",
            errors,
            flush=True,
        )


if __name__ == "__main__":
    absltest.main()
