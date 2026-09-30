# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F

from torchtitan.config import ConfigManager
from torchtitan.models.gemma4 import model_registry
from torchtitan.models.gemma4.model import (
    Gemma4Attention,
    Gemma4Model,
    Gemma4RoPE,
    QKVLinearKeqV,
)
from torchtitan.models.gemma4.state_dict_adapter import Gemma4StateDictAdapter


class TestGemma4(unittest.TestCase):
    def test_model_registry_and_config_manager(self) -> None:
        for flavor in ("debugmodel", "debugmodel_moe", "31B", "26B-A4B"):
            spec = model_registry(flavor)
            self.assertEqual(spec.name, "gemma4")
            self.assertEqual(spec.flavor, flavor)
            self.assertIsInstance(spec.model, Gemma4Model.Config)

        for config_name in (
            "gemma4_debugmodel",
            "gemma4_moe_debug",
            "gemma4_31b",
            "gemma4_26b_a4b",
        ):
            config_manager = ConfigManager()
            config = config_manager.parse_args(
                ["--module", "gemma4", "--config", config_name]
            )
            self.assertEqual(config.model_spec.name, "gemma4")

    def test_hybrid_attention_and_proportional_rope(self) -> None:
        spec = model_registry("debugmodel", attn_backend="sdpa")
        model = spec.model.build()
        model.init_states()

        self.assertEqual(len(model.layers), 6)
        # First 5 layers are local sliding window; 6th layer (index 5) is global.
        for idx in range(5):
            attn = model.layers[str(idx)].attention
            self.assertIsInstance(attn, Gemma4Attention)
            self.assertTrue(attn.is_sliding)
            self.assertFalse(attn.use_k_eq_v)
            self.assertEqual(attn.head_dim, 64)
            self.assertEqual(attn.n_kv_heads, 4)
            self.assertTrue(hasattr(attn.qkv_linear, "wv"))
            self.assertIsNone(attn.v_norm.weight)
            self.assertIsNotNone(attn.q_norm.weight)
            self.assertIsNotNone(attn.k_norm.weight)

        global_attn = model.layers["5"].attention
        self.assertIsInstance(global_attn, Gemma4Attention)
        self.assertFalse(global_attn.is_sliding)
        self.assertTrue(global_attn.use_k_eq_v)
        self.assertEqual(global_attn.head_dim, 128)
        self.assertEqual(global_attn.n_kv_heads, 2)
        self.assertIsInstance(global_attn.qkv_linear, QKVLinearKeqV)
        self.assertFalse(hasattr(global_attn.qkv_linear, "wv"))

        # Check proportional RoPE cache on global layer (partial_rotary_factor=0.25):
        # For head_dim=128, half_dim=64, rope_angles=16, nop_angles=48.
        # Unrotated angles must have cos=1 and sin=0 across all positions.
        rope = global_attn.rope
        self.assertIsInstance(rope, Gemma4RoPE)
        cos_cache = rope.cache[:, :128]
        sin_cache = rope.cache[:, 128:]
        self.assertTrue(torch.allclose(cos_cache[:, 16:64], torch.ones_like(cos_cache[:, 16:64])))
        self.assertTrue(torch.allclose(sin_cache[:, 16:64], torch.zeros_like(sin_cache[:, 16:64])))

    def test_dense_forward_backward_and_softcapping(self) -> None:
        torch.manual_seed(42)
        spec = model_registry("debugmodel", attn_backend="sdpa")
        model = spec.model.build()
        model.to_empty(device="cpu")
        model.init_states()

        # Verify weight tying and layer_scalar initialization after to_empty + init_states.
        self.assertIs(model.lm_head.weight, model.tok_embeddings.weight)
        for block in model.layers.values():
            self.assertTrue(torch.equal(block.layer_scalar, torch.ones(1)))

        tokens = torch.randint(0, spec.model.vocab_size, (2, 16))
        positions = torch.arange(16).unsqueeze(0).expand(2, -1)
        masks = model.get_attention_masks(positions)
        self.assertIsNone(masks)
        logits = model(tokens, positions=positions, attention_masks=masks)

        self.assertEqual(logits.shape, (2, 16, spec.model.vocab_size))
        self.assertLessEqual(logits.abs().max().item(), 30.0 + 1e-5)

        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(model.tok_embeddings.weight.grad)
        self.assertIsNotNone(model.layers["0"].feed_forward.w1.weight.grad)
        self.assertIsNotNone(model.layers["5"].attention.qkv_linear.wk.weight.grad)

    def test_moe_forward_backward(self) -> None:
        torch.manual_seed(42)
        spec = model_registry("debugmodel_moe", attn_backend="sdpa")
        model = spec.model.build()
        model.to_empty(device="cpu")
        model.init_states()

        block0 = model.layers["0"]
        self.assertTrue(block0.moe_enabled)
        self.assertIsNotNone(block0.moe)
        self.assertTrue(torch.equal(block0.moe.router.scale, torch.ones(256)))
        self.assertTrue(torch.equal(block0.moe.router.per_expert_scale, torch.ones(8)))

        tokens = torch.randint(0, spec.model.vocab_size, (2, 16))
        positions = torch.arange(16).unsqueeze(0).expand(2, -1)
        logits = model(tokens, positions=positions, attention_masks=None)

        self.assertEqual(logits.shape, (2, 16, spec.model.vocab_size))
        loss = logits.sum()
        loss.backward()

        # Verify both parallel branches (dense MLP + MoE) receive gradients.
        self.assertIsNotNone(block0.feed_forward.w1.weight.grad)
        self.assertIsNotNone(block0.post_feedforward_layernorm_1.weight.grad)
        self.assertIsNotNone(block0.post_feedforward_layernorm_2.weight.grad)
        self.assertIsNotNone(block0.moe.pre_feedforward_layernorm_2.weight.grad)
        self.assertIsNotNone(block0.moe.router.gate.weight.grad)
        self.assertIsNotNone(block0.moe.router.scale.grad)
        self.assertIsNotNone(block0.moe.router.per_expert_scale.grad)
        self.assertIsNotNone(block0.moe.experts.w1_EFD.grad)
        self.assertIsNotNone(block0.moe.experts.w2_EDF.grad)
        self.assertIsNotNone(block0.moe.experts.w3_EFD.grad)

    def test_state_dict_adapter_roundtrip_dense_and_moe(self) -> None:
        for flavor in ("debugmodel", "debugmodel_moe"):
            spec = model_registry(flavor, attn_backend="sdpa")
            model = spec.model.build()
            model.init_states()

            adapter = Gemma4StateDictAdapter(spec.model, hf_assets_path=None)
            tt_sd = model.state_dict()
            hf_sd = adapter.to_hf(tt_sd)

            # Tied lm_head.weight is omitted in HF state dict and restored in from_hf.
            self.assertNotIn("lm_head.weight", hf_sd)
            self.assertIn("model.language_model.embed_tokens.weight", hf_sd)

            if flavor == "debugmodel_moe":
                gate_up_key = "model.language_model.layers.0.experts.gate_up_proj"
                down_key = "model.language_model.layers.0.experts.down_proj"
                self.assertIn(gate_up_key, hf_sd)
                self.assertIn(down_key, hf_sd)
                self.assertEqual(hf_sd[gate_up_key].shape, (8, 256, 256))
                self.assertEqual(hf_sd[down_key].shape, (8, 256, 128))

            # Add dummy multimodal keys to verify they are ignored on load.
            hf_sd_with_mm = dict(hf_sd)
            hf_sd_with_mm["model.vision_tower.encoder.layers.0.weight"] = torch.randn(4, 4)
            hf_sd_with_mm["model.audio_tower.encoder.layers.0.weight"] = torch.randn(4, 4)

            roundtrip_sd = adapter.from_hf(hf_sd_with_mm)
            self.assertEqual(set(roundtrip_sd.keys()), set(tt_sd.keys()))
            for k, v in tt_sd.items():
                self.assertTrue(
                    torch.equal(v, roundtrip_sd[k]),
                    f"Mismatch after roundtrip for key {k} in {flavor}",
                )

    def test_update_from_config_and_nparams_flops(self) -> None:
        for config_name in ("gemma4_debugmodel", "gemma4_moe_debug"):
            config_manager = ConfigManager()
            trainer_cfg = config_manager.parse_args(
                ["--module", "gemma4", "--config", config_name]
            )
            model_cfg = trainer_cfg.model_spec.model
            assert isinstance(model_cfg, Gemma4Model.Config)
            model_cfg.update_from_config(config=trainer_cfg)

            model = model_cfg.build()
            model.init_states()
            nparams, flops = model_cfg.get_nparams_and_flops(model, seq_len=128)
            self.assertGreater(nparams, 0)
            self.assertGreater(flops, 0)

            # Verify TP degree validation rejects tp=4 when n_global_kv_heads=2.
            trainer_cfg.parallelism.tensor_parallel_degree = 4
            with self.assertRaises(ValueError):
                model_cfg.update_from_config(config=trainer_cfg)

    def test_fused_qkv_sliding_layers(self) -> None:
        spec = model_registry(
            "debugmodel", attn_backend="sdpa", use_fused_qkv=True
        )
        model = spec.model.build()
        model.init_states()
        tokens = torch.randint(0, spec.model.vocab_size, (2, 16))
        positions = torch.arange(16).unsqueeze(0).expand(2, -1)
        logits = model(tokens, positions=positions)
        self.assertEqual(logits.shape, (2, 16, spec.model.vocab_size))

    def test_hf_numerics_parity_dense_and_moe(self) -> None:
        try:
            from transformers import Gemma4ForCausalLM, Gemma4TextConfig
        except ImportError:
            self.skipTest("Gemma4ForCausalLM requires transformers v5")

        for flavor, enable_moe in (
            ("debugmodel", False),
            ("debugmodel_moe", True),
        ):
            torch.manual_seed(1234)
            hf_cfg = Gemma4TextConfig(
                vocab_size=2048,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=6,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=64,
                global_head_dim=128,
                num_global_key_value_heads=2,
                attention_k_eq_v=True,
                rms_norm_eps=1e-6,
                max_position_embeddings=4096,
                sliding_window=128,
                final_logit_softcapping=30.0,
                hidden_size_per_layer_input=0,
                tie_word_embeddings=True,
                enable_moe_block=enable_moe,
                num_experts=8 if enable_moe else None,
                top_k_experts=2 if enable_moe else None,
                moe_intermediate_size=128 if enable_moe else None,
                attn_implementation="eager",
            )
            hf_model = Gemma4ForCausalLM(hf_cfg).eval()
            # Initialize router/expert/norm weights with non-trivial random values
            # so every scale, norm, and expert path is exercised.
            with torch.no_grad():
                for name, param in hf_model.named_parameters():
                    if "norm" in name or "scale" in name:
                        param.copy_(torch.ones_like(param) + 0.1 * torch.randn_like(param))
                    elif "router" in name or "experts" in name:
                        param.copy_(0.02 * torch.randn_like(param))
                for layer in hf_model.model.layers:
                    layer.layer_scalar.copy_(torch.tensor([0.85]))

            spec = model_registry(flavor, attn_backend="sdpa")
            tt_model = spec.model.build().eval()
            tt_model.init_states()

            adapter = Gemma4StateDictAdapter(spec.model, hf_assets_path=None)
            tt_state_dict = adapter.from_hf(hf_model.state_dict())
            tt_model.load_state_dict(tt_state_dict, strict=True)

            input_ids = torch.randint(0, 2048, (2, 16))
            positions = torch.arange(16).unsqueeze(0).expand(2, -1)

            if enable_moe:
                # First verify Gemma4GroupedExperts._experts_forward (which uses
                # bfloat16 torch._grouped_mm with 1-token padding per expert)
                # directly against the exact float32 expert matmul reference,
                # including when some experts receive 0 tokens.
                sample_layer = next(iter(tt_model.layers.values()))
                exp = sample_layer.moe.experts
                x_sample = torch.randn(16, 256)
                counts_sample = torch.tensor([4, 0, 5, 0, 3, 2, 0, 2], dtype=torch.int64)

                def _fp32_experts_forward(
                    x_RD: torch.Tensor,
                    num_tokens_per_expert_E: torch.Tensor,
                    _exp=exp,
                ) -> torch.Tensor:
                    out = []
                    offset = 0
                    for e, count in enumerate(num_tokens_per_expert_E.tolist()):
                        if count > 0:
                            x_e = x_RD[offset : offset + count]
                            h = F.gelu(
                                F.linear(x_e, _exp.w1_EFD[e]),
                                approximate="tanh",
                            ) * F.linear(x_e, _exp.w3_EFD[e])
                            out.append(F.linear(h, _exp.w2_EDF[e]))
                            offset += count
                    return (
                        torch.cat(out, dim=0)
                        if out
                        else x_RD.new_zeros(x_RD.shape)
                    )

                with torch.no_grad():
                    bf16_grouped_out = exp._experts_forward(
                        x_sample, counts_sample
                    )
                    fp32_ref_out = _fp32_experts_forward(
                        x_sample, counts_sample
                    )
                torch.testing.assert_close(
                    bf16_grouped_out, fp32_ref_out, rtol=2e-2, atol=2e-2
                )

                # For the 6-layer end-to-end model parity check against unmodified
                # HuggingFace Gemma4ForCausalLM (which runs in float32), evaluate
                # expert matmuls in float32 so 1-ULP bfloat16 rounding in early
                # layers cannot flip a near-tied top-k router decision in deeper
                # layers across different CPU SIMD instruction sets.
                for layer in tt_model.layers.values():
                    layer_exp = layer.moe.experts
                    layer_exp._experts_forward = (
                        lambda x_RD, counts_E, _e=layer_exp: _fp32_experts_forward(
                            x_RD, counts_E, _exp=_e
                        )
                    )

            with torch.no_grad():
                hf_logits = hf_model(
                    input_ids=input_ids, position_ids=positions
                ).logits
                tt_logits = tt_model(input_ids, positions=positions)

            max_diff = (hf_logits - tt_logits).abs().max().item()
            kl = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(tt_logits, dim=-1),
                torch.nn.functional.softmax(hf_logits, dim=-1),
                reduction="batchmean",
            ).item()
            top1_match = (
                (hf_logits.argmax(dim=-1) == tt_logits.argmax(dim=-1))
                .float()
                .mean()
                .item()
            )
            print(
                f"[Numerics {flavor}] max_abs_diff={max_diff:.6e}, "
                f"kl_div={kl:.6e}, top1_match={top1_match:.4f}"
            )
            torch.testing.assert_close(
                tt_logits, hf_logits, rtol=1e-4, atol=1e-4
            )
            self.assertLess(abs(kl), 1e-4)
            self.assertEqual(top1_match, 1.0)

    def test_loss_convergence_50_steps_dense_and_moe(self) -> None:
        from torchtitan.components.loss import CrossEntropyLoss

        for flavor in ("debugmodel", "debugmodel_moe"):
            torch.manual_seed(42)
            spec = model_registry(flavor, attn_backend="sdpa")
            model = spec.model.build()
            model.to_empty(device="cpu")
            model.init_states()
            model.train()

            loss_fn = CrossEntropyLoss(
                CrossEntropyLoss.Config(global_vocab_size=spec.model.vocab_size)
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

            batch_size, seq_len = 2, 32
            tokens = torch.randint(
                0, spec.model.vocab_size, (batch_size, seq_len + 1)
            )
            input_ids = tokens[:, :-1]
            labels = tokens[:, 1:]
            positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            valid_tokens = float(labels.numel())

            losses: list[float] = []
            for step in range(1, 51):
                optimizer.zero_grad()
                logits = model(input_ids, positions=positions)
                loss, _ = loss_fn(
                    logits, labels, global_valid_tokens=valid_tokens
                )
                self.assertFalse(torch.isnan(loss).item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())

            print(
                f"[Convergence 50-step {flavor}] "
                f"step 1={losses[0]:.4f}, step 10={losses[9]:.4f}, "
                f"step 25={losses[24]:.4f}, step 50={losses[49]:.4f}"
            )
            self.assertLess(losses[-1], losses[0] * 0.5)
            self.assertLess(losses[24], losses[0])
            self.assertLess(losses[-1], losses[24])

    def test_grouped_experts_zero_token_experts_backward(self) -> None:
        from torchtitan.models.common.moe import LocalTokenDispatcher
        from torchtitan.models.gemma4.model import Gemma4GroupedExperts

        torch.manual_seed(42)
        E, D, F_dim = 4, 32, 64
        experts_cfg = Gemma4GroupedExperts.Config(
            dim=D,
            hidden_dim=F_dim,
            num_experts=E,
            token_dispatcher=LocalTokenDispatcher.Config(
                num_experts=E, top_k=2
            ),
        )
        experts = experts_cfg.build()
        with torch.no_grad():
            for p in (experts.w1_EFD, experts.w2_EDF, experts.w3_EFD):
                torch.nn.init.normal_(p, std=0.02)

        # Experts 1 and 3 receive 0 tokens: counts = [3, 0, 2, 0], total R = 5
        counts = torch.tensor([3, 0, 2, 0], dtype=torch.int32)
        x = torch.randn(5, D, dtype=torch.bfloat16, requires_grad=True)
        out = experts._experts_forward(x, counts)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        # Reference per-expert computation for the active experts (0 and 2)
        x_ref = x.detach().clone().requires_grad_(True)
        w1_ref = experts.w1_EFD.detach().bfloat16().clone().requires_grad_(True)
        w2_ref = experts.w2_EDF.detach().bfloat16().clone().requires_grad_(True)
        w3_ref = experts.w3_EFD.detach().bfloat16().clone().requires_grad_(True)

        out_e0 = (
            torch.nn.functional.gelu(
                x_ref[:3] @ w1_ref[0].T, approximate="tanh"
            )
            * (x_ref[:3] @ w3_ref[0].T)
        ) @ w2_ref[0].T
        out_e2 = (
            torch.nn.functional.gelu(
                x_ref[3:] @ w1_ref[2].T, approximate="tanh"
            )
            * (x_ref[3:] @ w3_ref[2].T)
        ) @ w2_ref[2].T
        out_ref = torch.cat([out_e0, out_e2], dim=0)
        out_ref.backward(grad_out)

        torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(
            experts.w1_EFD.grad.bfloat16(), w1_ref.grad, atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            experts.w2_EDF.grad.bfloat16(), w2_ref.grad, atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            experts.w3_EFD.grad.bfloat16(), w3_ref.grad, atol=1e-3, rtol=1e-3
        )
        # Zero-token experts (1 and 3) must have strictly zero weight gradients
        self.assertEqual(experts.w1_EFD.grad[1].abs().max().item(), 0.0)
        self.assertEqual(experts.w1_EFD.grad[3].abs().max().item(), 0.0)
        self.assertEqual(experts.w2_EDF.grad[1].abs().max().item(), 0.0)
        self.assertEqual(experts.w2_EDF.grad[3].abs().max().item(), 0.0)
        self.assertEqual(experts.w3_EFD.grad[1].abs().max().item(), 0.0)
        self.assertEqual(experts.w3_EFD.grad[3].abs().max().item(), 0.0)

    def test_moe_ep2_fsdp_multi_step_training(self) -> None:
        import socket
        import torch.multiprocessing as mp

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        mp.spawn(
            _run_ep2_fsdp_worker,
            args=(2, port),
            nprocs=2,
            join=True,
        )


def _run_ep2_fsdp_worker(rank: int, world_size: int, port: int) -> None:
    from unittest.mock import patch
    import torch.distributed as dist
    from torchtitan.components.loss import CrossEntropyLoss
    from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.distributed import ParallelDims
    from torchtitan.models.gemma4.config_registry import gemma4_moe_debug
    from torchtitan.models.gemma4.parallelize import parallelize_gemma4

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )
    try:
        with patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
            parallel_dims = ParallelDims(
                dp_replicate=1,
                dp_shard=world_size,
                cp=1,
                tp=1,
                pp=1,
                ep=world_size,
                world_size=world_size,
            )
            parallel_dims.build_mesh()

        trainer_cfg = gemma4_moe_debug()
        trainer_cfg.model_spec = model_registry(
            "debugmodel_moe", attn_backend="sdpa"
        )
        trainer_cfg.parallelism = ParallelismConfig(
            data_parallel_shard_degree=world_size,
            expert_parallel_degree=world_size,
        )
        trainer_cfg.training = TrainingConfig(
            local_batch_size=2,
            seq_len=32,
            steps=8,
        )
        model_cfg = trainer_cfg.model_spec.model
        model_cfg.update_from_config(config=trainer_cfg)

        with torch.device("meta"):
            model = model_cfg.build()

        model = parallelize_gemma4(
            model,
            parallel_dims=parallel_dims,
            training=trainer_cfg.training,
            parallelism=trainer_cfg.parallelism,
            compile_config=CompileConfig(enable=False),
            ac_config=None,
            dump_folder="/tmp",
        )
        model.to_empty(device="cpu")
        with torch.no_grad():
            model.init_states()
        model.train()

        loss_fn = CrossEntropyLoss(
            CrossEntropyLoss.Config(global_vocab_size=model_cfg.vocab_size)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

        torch.manual_seed(100 + rank)
        tokens = torch.randint(0, model_cfg.vocab_size, (2, 33))
        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]
        positions = torch.arange(32).unsqueeze(0).expand(2, -1)
        valid_tokens = float(labels.numel())

        losses: list[float] = []
        for step in range(1, 9):
            optimizer.zero_grad()
            logits = model(input_ids, positions=positions)
            loss, _ = loss_fn(logits, labels, global_valid_tokens=valid_tokens)
            assert not torch.isnan(loss).item(), f"NaN loss at step {step}"
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if rank == 0:
            print(
                f"[EP=2 FSDP=2 8-step MoE] "
                f"step 1={losses[0]:.4f}, step 5={losses[4]:.4f}, step 8={losses[7]:.4f}"
            )
        assert losses[-1] < losses[0], (
            f"Expected EP=2 FSDP=2 loss to decrease: {losses[0]} -> {losses[-1]}"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()

