"""GPT-OSS baseline: upstream model with CPU math attention as the reference.

FlexAttention has no CPU backward. Only the reference's inner attention is
replaced by explicit scores/softmax; its architecture, routing, experts, RoPE,
sink callback and checkpoint hooks are the upstream implementation.
"""

from absl.testing import absltest, parameterized
import torch
import torch.nn.functional as F

from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.experiments.tpu.gpt_oss import model_registry
from torchtitan.experiments.tpu.gpt_oss.attention import SinkSDPA
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
from torchtitan.models.gpt_oss import model_registry as upstream_registry
from torchtitan.models.gpt_oss.state_dict_adapter import GptOssStateDictAdapter


class _ReferenceAttention(torch.nn.Module):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def forward(self, q, k, v, *, positions, scale, out_transform, **kwargs):
        repeats = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
        scores = torch.einsum("thd,shd->hts", q, k) * scale
        ids = (positions == 0).long().cumsum(0)
        indices = torch.arange(q.shape[0])
        distance = indices[:, None] - indices[None, :]
        allowed = (distance >= 0) & (ids[:, None] == ids[None, :])
        if self.window is not None:
            allowed &= distance < self.window
        scores = scores.masked_fill(~allowed, float("-inf"))
        output = torch.einsum("hts,shd->thd", scores.softmax(-1), v)
        return out_transform(output, scores.logsumexp(-1).T)


def tiny_config(registry=model_registry, **kwargs):
    config = registry("debugmodel", seq_len=32, **kwargs).model
    config.layers = config.layers[:2]
    for layer in config.layers:
        attn = layer.attention
        attn.n_heads, attn.n_kv_heads, attn.head_dim = 8, 2, 32
        attn.qkv_linear.n_heads = 8
        attn.qkv_linear.n_kv_heads = 2
        attn.qkv_linear.head_dim = 32
        attn.qkv_linear.wqkv.out_features = 12 * 32
        attn.wo.in_features = 8 * 32
        attn.rope.dim = 32
        if attn.sliding_window_size is not None:
            attn.sliding_window_size = 4
        if isinstance(attn.inner_attention, SinkSDPA.Config):
            attn.inner_attention.window_size = attn.sliding_window_size
        layer.moe.routed_experts.inner_experts.hidden_dim = 128
    return config


def assert_gradients(test, actual, reference):
    test.assertEqual(set(actual), set(reference))
    worst = (0.0, "")
    for name, expected in reference.items():
        test.assertIsNotNone(actual[name], name)
        test.assertIsNotNone(expected, name)
        norm = expected.float().norm().item()
        error = (actual[name].cpu().float() - expected.float()).norm().item()
        if norm < 1e-4:
            test.assertLessEqual(error, 1e-6, name)
        else:
            relative = error / norm
            test.assertLessEqual(relative, 2e-2, name)
            worst = max(worst, (relative, name))
    print("worst gradient relative L2:", worst)


class TestGptOssReferenceParity(BaseAcceleratorDeviceTest):
    def test_upstream_reference(self):
        reference_cfg = tiny_config(upstream_registry, attn_backend="flex")
        reference = reference_cfg.build().eval()
        reference.init_states(buffer_device=torch.device("cpu"))
        for layer, layer_config in zip(reference.layers.values(), reference_cfg.layers):
            layer.attention.inner_attention = _ReferenceAttention(
                layer_config.attention.sliding_window_size
            )
            # Upstream returns bf16 experts into an fp32 scatter destination.
            # Restore the dispatch contract without changing expert arithmetic.
            layer.moe.routed_experts.inner_experts.register_forward_hook(
                lambda module, inputs, output: output.to(inputs[0].dtype)
            )
        with torch.no_grad():
            for name, value in reference.named_parameters():
                if "norm.weight" in name:
                    value.copy_(1 + 0.1 * torch.randn_like(value))
                elif name.endswith("sinks"):
                    value.copy_(torch.randn_like(value))
        cfg = tiny_config()
        model = cfg.build().to(self.accelerator_device).eval()
        model.init_states(buffer_device=self.accelerator_device)
        # Both sides traverse the production adapter and its QKV load/save hooks.
        source_adapter = GptOssStateDictAdapter(reference_cfg, None)
        target_adapter = GptOssStateDictAdapter(cfg, None)
        source = source_adapter.to_hf(reference.state_dict())
        mapped = target_adapter.from_hf(source)
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        self.assertEmpty(missing)
        self.assertEmpty(unexpected)
        self.assertEqual(set(source), set(target_adapter.to_hf(model.state_dict())))
        tokens = torch.randint(0, 2048, (32,))
        labels = torch.randint(0, 2048, (32,))
        # Two rows, each with three documents and boundaries inside the window.
        positions = torch.tensor(
            list(range(6)) + list(range(3)) + list(range(7))
        ).repeat(2)
        with set_unflatten_seq_len(16):
            expected = reference(tokens, positions)
            actual = model(
                tokens.to(self.accelerator_device),
                positions.to(self.accelerator_device),
            )
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-1, rtol=2e-2)
        expected_loss = F.cross_entropy(expected.float(), labels)
        actual_loss = F.cross_entropy(
            actual.float(), labels.to(self.accelerator_device)
        )
        torch.testing.assert_close(
            actual_loss.cpu(), expected_loss, atol=5e-3, rtol=2e-3
        )
        expected_loss.backward()
        actual_loss.backward()
        expected_grads = dict((n, p.grad) for n, p in reference.named_parameters())
        actual_grads = dict((n, p.grad) for n, p in model.named_parameters())
        assert_gradients(self, actual_grads, expected_grads)
        for multiplier in (0, -1):
            with self.assertRaises(AssertionError):
                assert_gradients(
                    self,
                    {n: g * multiplier for n, g in expected_grads.items()},
                    expected_grads,
                )


class TestGptOssConfig(absltest.TestCase):
    def test_checkpoint_roundtrip(self):
        cfg = tiny_config()
        model = cfg.build()
        model.init_states(buffer_device=torch.device("cpu"))
        state = model.state_dict()
        adapter = GptOssStateDictAdapter(cfg, None)
        restored = adapter.from_hf(adapter.to_hf(state))
        self.assertEqual(set(restored), set(state))
        for key in state:
            torch.testing.assert_close(restored[key], state[key], atol=0, rtol=0)

    def test_production_flavors(self):
        for flavor, layers, experts in (("20b", 24, 32), ("120b", 36, 128)):
            cfg = model_registry(flavor, seq_len=256).model
            self.assertLen(cfg.layers, layers)
            self.assertEqual(cfg.dim, 2880)
            self.assertEqual(cfg.vocab_size, 201088)
            self.assertFalse(cfg.enable_weight_tying)
            for i, layer in enumerate(cfg.layers):
                self.assertEqual(layer.moe.num_experts, experts)
                self.assertEqual(layer.moe.router.top_k, 4)
                self.assertEqual(layer.attention.n_heads, 64)
                self.assertEqual(layer.attention.n_kv_heads, 8)
                self.assertEqual(
                    layer.attention.inner_attention.window_size,
                    128 if i % 2 == 0 else None,
                )


class TestGptOssMasks(BaseAcceleratorDeviceTest):
    @parameterized.named_parameters(("window", 4), ("global", None))
    def test_sink_and_packed_mask(self, window):
        device = self.accelerator_device
        module = SinkSDPA.Config(window_size=window).build()
        q = torch.randn(1, 16, 4, 16, device=device, requires_grad=True)
        k = torch.randn(1, 16, 2, 16, device=device, requires_grad=True)
        v = torch.randn(1, 16, 2, 16, device=device, requires_grad=True)
        sinks = torch.randn(4, device=device, requires_grad=True)
        positions = torch.tensor(
            list(range(6)) + list(range(3)) + list(range(7)), device=device
        )

        def transform(out, lse):
            return out * torch.sigmoid(lse - sinks).unsqueeze(-1)

        actual = module(
            q, k, v, positions=positions, enable_gqa=True, out_transform=transform
        )
        # Independent explicit sink-softmax reference, including sink gradients.
        scores = torch.einsum("blhd,bshd->bhls", q, k.repeat_interleave(2, 2)) / 4
        distance = (
            torch.arange(16, device=device)[:, None]
            - torch.arange(16, device=device)[None, :]
        )
        segments = torch.tensor([0] * 6 + [1] * 3 + [2] * 7, device=device)
        allowed = (distance >= 0) & (segments[:, None] == segments[None, :])
        if window is not None:
            allowed &= distance < window
        scores = scores.masked_fill(~allowed, float("-inf"))
        probs = torch.cat(
            (scores, sinks[None, :, None, None].expand(1, 4, 16, 1)), -1
        ).softmax(-1)[..., :-1]
        expected = torch.einsum("bhls,bshd->blhd", probs, v.repeat_interleave(2, 2))
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=5e-3, rtol=1e-2)
        actual_grads = torch.autograd.grad(
            actual.square().sum(), (q, k, v, sinks), retain_graph=True
        )
        expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v, sinks))
        assert_gradients(
            self,
            dict(enumerate(actual_grads)),
            {i: g.cpu() for i, g in enumerate(expected_grads)},
        )
        unmasked = module(q, k, v, enable_gqa=True, out_transform=transform)
        self.assertGreater(((unmasked - expected).norm() / expected.norm()).item(), 0.1)


if __name__ == "__main__":
    absltest.main()
