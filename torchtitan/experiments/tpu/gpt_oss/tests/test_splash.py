"""Packed global/local sink-attention parity against the GPT-OSS SDPA path."""

import copy

from absl.testing import absltest, parameterized
import torch

from torchtitan.config import ConfigManager, derive
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.experiments.tpu.gpt_oss import model_registry
from torchtitan.experiments.tpu.gpt_oss.splash_attention import SinkSplashAttention
from torchtitan.experiments.tpu.graph_trainer.config import to_tpu_graph_trainer_config
from torchtitan.experiments.tpu.graph_trainer.gpt_oss import (
    model_registry as gt_registry,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len


class TestSinkSplash(BaseAcceleratorDeviceTest):
    def test_recipe_preserves_all_layer_overrides(self):
        config = ConfigManager().parse_args(
            [
                "--module=torchtitan.experiments.tpu.gpt_oss",
                "--config=gpt_oss_debugmodel_splash",
            ]
        )
        graph_config = to_tpu_graph_trainer_config(
            config, gt_registry, target_cls=TpuGraphTrainer.Config
        )
        # Overrides are applied by the trainer just before model construction.
        from torchtitan.config.override import apply_overrides

        replacements = apply_overrides(graph_config.override, graph_config)
        self.assertLen(replacements, 4)
        model_config = graph_config.model_spec.model
        self.assertLen(model_config.layers, 4)
        self.assertTrue(
            all(
                isinstance(layer.attention, SinkSplashAttention.Config)
                for layer in model_config.layers
            )
        )
        self.assertEqual(
            [layer.attention.sliding_window_size for layer in model_config.layers],
            [128, None, 128, None],
        )

    @parameterized.named_parameters(
        ("global_fp32", None, False),
        ("local_fp32", 128, False),
        ("global_bf16", None, True),
        ("local_bf16", 128, True),
    )
    def test_packed_attention_and_gradients(self, window, amp):
        if self.accelerator_device.type != "tpu":
            self.skipTest("Tokamax Splash requires a TPU")
        torch.manual_seed(42)
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        self.addCleanup(torch.set_float32_matmul_precision, previous)
        set_unflatten_seq_len(512)
        self.addCleanup(set_unflatten_seq_len, None)
        config = copy.deepcopy(
            model_registry("debugmodel", seq_len=512).model.layers[0].attention
        )
        config.n_heads, config.n_kv_heads = 8, 2
        config.qkv_linear.n_heads, config.qkv_linear.n_kv_heads = 8, 2
        config.qkv_linear.wqkv.out_features = 12 * 64
        config.wo.in_features = 8 * 64
        config.sliding_window_size = window
        config.inner_attention.window_size = window
        reference = config.build().to(self.accelerator_device)
        reference.init_states(buffer_device=self.accelerator_device)
        actual = (
            derive(config, SinkSplashAttention.Config)
            .build()
            .to(self.accelerator_device)
        )
        actual.init_states(buffer_device=self.accelerator_device)
        with torch.no_grad():
            reference.sinks.copy_(
                torch.linspace(1, 5, 8, device=self.accelerator_device)
            )
        actual.load_state_dict(reference.state_dict(), strict=True)
        # Two rows, three documents each; boundaries fall inside 128-token tiles.
        positions = (
            torch.cat([torch.arange(n) for n in (193, 113, 206)])
            .repeat(2)
            .to(self.accelerator_device)
        )
        x = torch.randn(
            1024, config.dim, device=self.accelerator_device, requires_grad=True
        )
        y = x.detach().clone().requires_grad_(True)
        grad = torch.randn_like(x)
        with torch.autocast("tpu", dtype=torch.bfloat16, enabled=amp):
            expected = reference(x, None, positions)
            result = actual(y, None, positions)
            unmasked = reference(x, None, torch.arange(512, device=x.device).repeat(2))
        relative = (
            lambda a, b: (a.float() - b.float()).norm().item() / b.float().norm().item()
        )
        output_error = relative(result, expected)
        self.assertLessEqual(output_error, 1e-2)
        mask_control = relative(unmasked, expected)
        self.assertGreater(mask_control, 0.1)
        expected.backward(grad)
        result.backward(grad)
        expected_grads = {n: p.grad for n, p in reference.named_parameters()}
        actual_grads = {n: p.grad for n, p in actual.named_parameters()}
        expected_grads["input"], actual_grads["input"] = x.grad, y.grad
        self.assertEqual(set(expected_grads), set(actual_grads))
        errors = {}
        for name, target in expected_grads.items():
            self.assertIsNotNone(actual_grads[name], name)
            self.assertGreater(target.float().norm().item(), 1e-4, name)
            errors[name] = relative(actual_grads[name], target)
            self.assertLessEqual(errors[name], 2.5e-2, name)
            # The bound must reject missing and sign-flipped gradients.
            self.assertGreater(relative(torch.zeros_like(target), target), 2.5e-2)
            self.assertGreater(relative(-target, target), 2.5e-2)
        print(
            "SPLASH_PARITY",
            window,
            amp,
            "output",
            output_error,
            "mask_control",
            mask_control,
            "gradients",
            errors,
            flush=True,
        )


if __name__ == "__main__":
    absltest.main()
