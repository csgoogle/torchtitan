"""Tests for Qwen3.5 (text-only Dense and MoE) on CPU and TPU devices.

Covers:
1. HuggingFace reference parity (``Qwen3_5ForCausalLM`` and
   ``Qwen3_5MoeForCausalLM``) through ``Qwen35StateDictAdapter`` for forward
   logits, cross-entropy loss, and every parameter gradient (with negative
   controls), for dense, MoE, and MoE with ``TpuGroupedExpertsConverter``.
2. ``Qwen35StateDictAdapter`` round-trip (``from_hf(to_hf(sd)) == sd``).
3. ``chunked_gated_delta`` parity against the exact per-token recurrent gated
   delta rule, the slice-based causal convolution against ``conv1d``, and
   batch-sequence isolation (``B=2`` vs two ``B=1`` runs).
4. Packed-document isolation in ``TpuQwen35Attention`` and
   ``PackedScaledDotProductInnerAttention``, plus Splash parity on TPU.
5. Flavor, recipe and MFU-accounting checks.
"""

import copy

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F
from torchtitan.config.manager import ConfigManager
from torchtitan.config.override import apply_overrides
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.graph_trainer import qwen3_5 as gt_qwen3_5
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
import torchtitan.experiments.tpu.qwen3_5 as qwen3_5_tpu
from torchtitan.experiments.tpu.qwen3_5 import config_registry
from torchtitan.experiments.tpu.qwen3_5 import gated_delta
from torchtitan.experiments.tpu.qwen3_5.infra.parallelize import parallelize_qwen3_5
from torchtitan.experiments.tpu.token_dispatcher import TpuAllToAllTokenDispatcher
from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch


def _recurrent_gated_delta_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
  """Per-token reference for the gated delta rule recurrence in fp32."""
  q = gated_delta._l2norm(q.float()) * (q.shape[-1] ** -0.5)  # pylint: disable=protected-access
  k = gated_delta._l2norm(k.float())  # pylint: disable=protected-access
  v, g, beta = v.float(), g.float(), beta.float()
  bs, seq_len, n_heads, d_k = k.shape
  state = torch.zeros(
      bs, n_heads, d_k, v.shape[-1], dtype=torch.float32, device=q.device
  )
  outs = []
  for t in range(seq_len):
    state = state * g[:, t].exp()[..., None, None]
    kv_mem = (state * k[:, t, :, :, None]).sum(-2)
    delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
    state = state + k[:, t, :, :, None] * delta[:, :, None, :]
    outs.append((state * q[:, t, :, :, None]).sum(-2))
  return torch.stack(outs, dim=1)


def _relative_l2_error(test: torch.Tensor, reference: torch.Tensor) -> float:
  test_f = test.detach().float().cpu()
  ref_f = reference.detach().float().cpu()
  denom = torch.linalg.vector_norm(ref_f).clamp_min(1e-6)
  return (torch.linalg.vector_norm(test_f - ref_f) / denom).item()


HF_PARITY_CONFIGS = [
    dict(
        testcase_name="_dense",
        flavor="testmodel",
        is_moe=False,
        use_tpu_grouped_experts=False,
    ),
    dict(
        testcase_name="_moe",
        flavor="testmodel_moe",
        is_moe=True,
        use_tpu_grouped_experts=False,
    ),
    dict(
        testcase_name="_moe_tpu_grouped_experts",
        flavor="testmodel_moe",
        is_moe=True,
        use_tpu_grouped_experts=True,
    ),
]


class Qwen35Test(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for Qwen3.5 hybrid GatedDeltaNet + Attention + MoE on CPU and TPU."""

  def setUp(self):
    super().setUp()
    use_cpu_safe_histc_patch()
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    self.addCleanup(torch.set_float32_matmul_precision, previous_precision)

  def _is_tpu(self) -> bool:
    return self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU

  def _build_hf_reference(self, is_moe: bool) -> torch.nn.Module:
    try:
      from transformers import (  # pylint: disable=g-import-not-at-top
          Qwen3_5ForCausalLM,
          Qwen3_5MoeForCausalLM,
          Qwen3_5MoeTextConfig,
          Qwen3_5TextConfig,
      )
    except ImportError as exc:
      self.skipTest(f"transformers lacks Qwen3.5 classes: {exc}")

    common_kwargs = dict(
        vocab_size=2048,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
        },
        attn_implementation="sdpa",
    )
    torch.manual_seed(42)
    if is_moe:
      hf_cfg = Qwen3_5MoeTextConfig(
          **common_kwargs,
          moe_intermediate_size=64,
          shared_expert_intermediate_size=64,
          num_experts=8,
          num_experts_per_tok=2,
          tie_word_embeddings=False,
      )
      hf_model = Qwen3_5MoeForCausalLM(hf_cfg).float().eval()
    else:
      hf_cfg = Qwen3_5TextConfig(
          **common_kwargs,
          intermediate_size=256,
          tie_word_embeddings=True,
      )
      hf_model = Qwen3_5ForCausalLM(hf_cfg).float().eval()

    # Perturb unit/zero-initialized parameters so dropped scales cannot hide.
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
      for name, param in hf_model.named_parameters():
        if "norm" in name or "A_log" in name or "dt_bias" in name:
          param.copy_(0.2 * torch.randn(param.shape, generator=gen))
        elif "experts" in name or "gate" in name or "proj" in name:
          param.copy_(0.05 * torch.randn(param.shape, generator=gen))
    return hf_model

  @parameterized.named_parameters(*HF_PARITY_CONFIGS)
  def test_qwen3_5_hf_reference_parity(
      self,
      flavor: str,
      is_moe: bool,
      use_tpu_grouped_experts: bool,
  ):
    """Tests forward logits, loss, and all gradients against HF Qwen3.5."""
    hf_model = self._build_hf_reference(is_moe=is_moe)
    converters = (
        [TpuGroupedExpertsConverter.Config()] if use_tpu_grouped_experts else []
    )
    spec = qwen3_5_tpu.model_registry(flavor, converters=converters)
    tt_config = spec.model
    adapter = qwen3_5_tpu.Qwen35StateDictAdapter(tt_config, None)

    tt_model = tt_config.build()
    tt_model.init_states(buffer_device=torch.device("cpu"))

    # Check state_dict mapping both ways.
    tt_sd = adapter.from_hf(copy.deepcopy(hf_model.state_dict()))
    missing, unexpected = tt_model.load_state_dict(tt_sd, strict=False)
    self.assertEmpty(missing)
    self.assertEmpty(unexpected)

    if not use_tpu_grouped_experts:
      round_trip = adapter.from_hf(adapter.to_hf(tt_model.state_dict()))
      self.assertEqual(
          set(round_trip.keys()), set(tt_model.state_dict().keys())
      )
      for k, v in tt_model.state_dict().items():
        torch.testing.assert_close(round_trip[k], v, msg=f"Round-trip key {k}")

    tt_model = tt_model.to(self.accelerator_device)
    if tt_config.enable_weight_tying:
      tt_model.tok_embeddings.weight = tt_model.lm_head.weight
      self.assertIs(tt_model.tok_embeddings.weight, tt_model.lm_head.weight)

    batch, seq_len = 2, 16
    gen = torch.Generator().manual_seed(99)
    tokens_2d = torch.randint(
        0, tt_config.vocab_size, (batch, seq_len), generator=gen
    )
    labels_2d = torch.randint(
        0, tt_config.vocab_size, (batch, seq_len), generator=gen
    )
    positions_1d = torch.arange(seq_len, dtype=torch.long).repeat(batch)

    # Reference forward + backward on CPU in fp32.
    hf_model.zero_grad()
    hf_logits = hf_model(tokens_2d).logits
    hf_loss = F.cross_entropy(
        hf_logits.view(-1, tt_config.vocab_size), labels_2d.view(-1)
    )
    hf_loss.backward()

    # TorchTitan forward + backward on accelerator_device in packed [T] layout.
    tt_model.zero_grad()
    tokens_dev = tokens_2d.view(-1).to(self.accelerator_device)
    labels_dev = labels_2d.view(-1).to(self.accelerator_device)
    positions_dev = positions_1d.to(self.accelerator_device)

    # libtpu 0.0.48 rejects HIGHEST matmul precision for bf16/fp32 ragged dot
    # inside torch._grouped_mm on TPU; use "high" precision when running MoE on
    # TPU.
    precision = "high" if (self._is_tpu() and is_moe) else "highest"
    prev_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
      with set_unflatten_seq_len(seq_len):
        tt_logits = tt_model(tokens_dev, positions_dev)
        tt_loss = F.cross_entropy(
            tt_logits.float().view(-1, tt_config.vocab_size), labels_dev
        )
        tt_loss.backward()
    finally:
      torch.set_float32_matmul_precision(prev_prec)

    # GroupedExperts runs expert matmuls in bfloat16 even when the surrounding
    # model is float32 on CPU, so MoE uses bfloat16 tolerances on both CPU and
    # TPU.
    logit_atol = 5e-2 if (self._is_tpu() or is_moe) else 5e-3
    logit_rtol = 1e-2 if (self._is_tpu() or is_moe) else 5e-3
    loss_atol = 5e-3 if (self._is_tpu() or is_moe) else 1e-5
    loss_rtol = 1e-3 if (self._is_tpu() or is_moe) else 1e-5
    grad_rel_bound = 6e-2 if (self._is_tpu() or is_moe) else 5e-4

    test_utils.check_equivalence(
        tt_logits.cpu().view(batch, seq_len, -1),
        hf_logits.detach(),
        atol=logit_atol,
        rtol=logit_rtol,
        check_name="HF Parity Logits",
    )
    test_utils.check_equivalence(
        tt_loss.cpu(),
        hf_loss.detach(),
        atol=loss_atol,
        rtol=loss_rtol,
        check_name="HF Parity Loss",
    )

    hf_grads = {
        name: param.grad.detach().clone()
        for name, param in hf_model.named_parameters()
        if param.grad is not None
    }
    expected_tt_grads = adapter.from_hf(hf_grads)
    if tt_config.enable_weight_tying:
      expected_tt_grads.pop("lm_head.weight", None)

    actual_tt_grads = {
        name: param.grad.detach().float().cpu()
        for name, param in tt_model.named_parameters()
        if param.grad is not None
    }

    self.assertEqual(set(actual_tt_grads.keys()), set(expected_tt_grads.keys()))
    for name, ref_grad in expected_tt_grads.items():
      rel_err = _relative_l2_error(actual_tt_grads[name], ref_grad)
      self.assertLessEqual(
          rel_err,
          grad_rel_bound,
          msg=f"Gradient relative L2 error too high for {name}: {rel_err:.3e}",
      )
      # Negative controls: zeroed and sign-flipped gradients must exceed bound.
      if torch.linalg.vector_norm(ref_grad).item() > 1e-4:
        self.assertGreater(
            _relative_l2_error(torch.zeros_like(ref_grad), ref_grad),
            grad_rel_bound,
        )
        self.assertGreater(
            _relative_l2_error(-ref_grad, ref_grad),
            grad_rel_bound,
        )

  def test_chunked_gated_delta_matches_recurrent(self):
    """Tests chunked_gated_delta against the per-token recurrent reference."""
    bs, seq_len, n_heads, d_k, d_v = 2, 50, 4, 16, 16
    q, q_dev = self._create_input_tensor_cpu_device(
        (bs, seq_len, n_heads, d_k), requires_grad=True
    )
    k, k_dev = self._create_input_tensor_cpu_device(
        (bs, seq_len, n_heads, d_k), requires_grad=True
    )
    v, v_dev = self._create_input_tensor_cpu_device(
        (bs, seq_len, n_heads, d_v), requires_grad=True
    )
    g_raw, g_dev_raw = self._create_input_tensor_cpu_device(
        (bs, seq_len, n_heads), requires_grad=True
    )
    beta_raw, beta_dev_raw = self._create_input_tensor_cpu_device(
        (bs, seq_len, n_heads), requires_grad=True
    )

    g = -F.softplus(g_raw)
    beta = torch.sigmoid(beta_raw)
    g_dev = -F.softplus(g_dev_raw)
    beta_dev = torch.sigmoid(beta_dev_raw)

    expected = _recurrent_gated_delta_reference(q, k, v, g, beta)
    out_dev = gated_delta.chunked_gated_delta(
        q_dev, k_dev, v_dev, g_dev, beta_dev, chunk_size=16
    )
    tol = 5e-3 if self._is_tpu() else 1e-4
    test_utils.check_equivalence(
        out_dev.cpu(),
        expected,
        atol=tol,
        rtol=tol,
        check_name="Chunked vs Recurrent GDN Forward",
    )

    expected.sum().backward()
    out_dev.sum().backward()
    grad_tol = 2e-2 if self._is_tpu() else 5e-4
    for name, g_cpu, g_device in (
        ("q", q.grad, q_dev.grad),
        ("k", k.grad, k_dev.grad),
        ("v", v.grad, v_dev.grad),
        ("g", g_raw.grad, g_dev_raw.grad),
        ("beta", beta_raw.grad, beta_dev_raw.grad),
    ):
      self.assertLessEqual(
          _relative_l2_error(g_device, g_cpu),
          grad_tol,
          msg=f"GDN {name} grad mismatch",
      )

  @parameterized.named_parameters(
      dict(testcase_name="_chunk64", chunk_size=64),
      dict(testcase_name="_chunk128", chunk_size=128),
  )
  def test_chunked_gated_delta_matches_recurrent_correlated_keys(
      self, chunk_size: int
  ):
    """Tests the chunked UT transform with highly correlated keys.

    Nearly parallel keys with beta near 1 and little decay push the
    triangular system (I + tril(diag(beta) K K^T * decay)) toward its worst
    conditioning, which is where an inaccurate inverse would show up.
    """
    torch.manual_seed(0)
    bs, seq_len, n_heads, d_k, d_v = 1, 160, 2, 32, 32
    base = torch.randn(bs, 1, n_heads, d_k)
    k_cpu = (
        base + 0.05 * torch.randn(bs, seq_len, n_heads, d_k)
    ).requires_grad_()
    q_cpu = torch.randn(bs, seq_len, n_heads, d_k, requires_grad=True)
    v_cpu = torch.randn(bs, seq_len, n_heads, d_v, requires_grad=True)
    g_raw_cpu = torch.full((bs, seq_len, n_heads), -4.0, requires_grad=True)
    beta_raw_cpu = torch.full((bs, seq_len, n_heads), 4.0, requires_grad=True)
    leaves_cpu = (q_cpu, k_cpu, v_cpu, g_raw_cpu, beta_raw_cpu)
    leaves_dev = tuple(
        x.detach().to(self.accelerator_device).requires_grad_()
        for x in leaves_cpu
    )

    def run(fn, q, k, v, g_raw, beta_raw, **kwargs):
      return fn(q, k, v, -F.softplus(g_raw), torch.sigmoid(beta_raw), **kwargs)

    expected = run(_recurrent_gated_delta_reference, *leaves_cpu)
    out_dev = run(
        gated_delta.chunked_gated_delta,
        *leaves_dev,
        chunk_size=chunk_size,
    )
    tol = 5e-3 if self._is_tpu() else 1e-4
    test_utils.check_equivalence(
        out_dev.cpu(),
        expected,
        atol=tol,
        rtol=tol,
        check_name="Chunked vs Recurrent GDN Forward (correlated keys)",
    )
    grad_out = torch.randn_like(expected)
    expected.backward(grad_out)
    out_dev.backward(grad_out.to(self.accelerator_device))
    grad_tol = 2e-2 if self._is_tpu() else 5e-4
    for name, g_cpu, g_dev in zip(
        ("q", "k", "v", "g", "beta"), leaves_cpu, leaves_dev, strict=True
    ):
      self.assertLessEqual(
          _relative_l2_error(g_dev.grad, g_cpu.grad),
          grad_tol,
          msg=f"GDN {name} grad mismatch (correlated keys)",
      )

  def test_causal_conv_silu_matches_autograd(self):
    """Tests the slice-based causal conv backward against autograd/conv1d."""
    torch.manual_seed(0)
    channels, kernel_size = 8, 4
    x = torch.randn(3, 37, channels, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(
        channels, 1, kernel_size, dtype=torch.float64, requires_grad=True
    )
    grad = torch.randn(3, 37, channels, dtype=torch.float64)
    # pylint: disable=protected-access
    conv = gated_delta._causal_conv(x, weight)
    expected_conv = F.conv1d(
        F.pad(x.transpose(1, 2), (kernel_size - 1, 0)),
        weight,
        groups=channels,
    ).transpose(1, 2)
    # pylint: enable=protected-access
    torch.testing.assert_close(conv, expected_conv)
    out = gated_delta.causal_conv_silu(x, weight)
    expected = F.silu(expected_conv)
    torch.testing.assert_close(out, expected)
    grads = torch.autograd.grad(out, (x, weight), grad)
    expected_grads = torch.autograd.grad(expected, (x, weight), grad)
    for actual_grad, expected_grad in zip(grads, expected_grads):
      torch.testing.assert_close(actual_grad, expected_grad)

    # On the accelerator in bf16, as the model runs it.
    dev = self.accelerator_device
    x_bf16 = x.detach().to(dev, torch.bfloat16).requires_grad_()
    w_bf16 = weight.detach().to(dev, torch.bfloat16).requires_grad_()
    out_bf16 = gated_delta.causal_conv_silu(x_bf16, w_bf16)
    grads_bf16 = torch.autograd.grad(
        out_bf16, (x_bf16, w_bf16), grad.to(dev, torch.bfloat16)
    )
    self.assertLess(_relative_l2_error(out_bf16.double().cpu(), expected), 1e-2)
    for actual_grad, expected_grad in zip(grads_bf16, expected_grads):
      self.assertEqual(actual_grad.dtype, torch.bfloat16)
      self.assertLess(
          _relative_l2_error(actual_grad.double().cpu(), expected_grad), 2e-2
      )

  def test_gated_delta_net_isolates_batch_sequences(self):
    """Tests that B=2 packed sequences in GatedDeltaNet equal two B=1 runs."""
    config = qwen3_5_tpu.qwen3_5_configs["testmodel"]
    gdn = config.layers[0].delta_net.build()
    gdn.init_states(buffer_device=torch.device("cpu"))
    gdn = gdn.to(self.accelerator_device)

    seq_len = 16
    x0 = torch.randn(seq_len, config.dim, device=self.accelerator_device)
    x1 = torch.randn(seq_len, config.dim, device=self.accelerator_device)
    x_b2 = torch.cat([x0, x1], dim=0)

    with torch.no_grad():
      with set_unflatten_seq_len(seq_len):
        out0 = gdn(x0)
        out1 = gdn(x1)
        out_b2 = gdn(x_b2)
      # Negative control: if treated as 1 sequence of length 2*seq_len, the
      # second half changes significantly because conv1d and state bleed across.
      with set_unflatten_seq_len(2 * seq_len):
        out_bleed = gdn(x_b2)

    tol = 5e-3 if self._is_tpu() else 1e-5
    test_utils.check_equivalence(
        out_b2[:seq_len].cpu(),
        out0.cpu(),
        atol=tol,
        rtol=tol,
        check_name="GDN Batch Slice 0",
    )
    test_utils.check_equivalence(
        out_b2[seq_len:].cpu(),
        out1.cpu(),
        atol=tol,
        rtol=tol,
        check_name="GDN Batch Slice 1",
    )
    self.assertGreater(
        _relative_l2_error(out_bleed[seq_len:], out1),
        0.05,
        msg="Negative control failed: GDN bleed did not change slice 1",
    )

  def test_attention_isolates_packed_documents(self):
    """Tests that TpuQwen35Attention isolates packed documents via positions."""
    config = qwen3_5_tpu.qwen3_5_configs["testmodel"]
    attn = config.layers[3].attention.build()
    attn.init_states(buffer_device=torch.device("cpu"))
    attn = attn.to(self.accelerator_device)

    len_a, len_b = 7, 9
    seq_len = len_a + len_b
    x = torch.randn(seq_len, config.dim, device=self.accelerator_device)
    positions_packed = torch.cat([
        torch.arange(len_a, device=self.accelerator_device),
        torch.arange(len_b, device=self.accelerator_device),
    ])
    positions_unpacked = torch.arange(seq_len, device=self.accelerator_device)

    with torch.no_grad():
      with set_unflatten_seq_len(seq_len):
        out_packed = attn(x, None, positions_packed)
        out_unpacked = attn(x, None, positions_unpacked)
      with set_unflatten_seq_len(len_a):
        out_a = attn(
            x[:len_a],
            None,
            torch.arange(len_a, device=self.accelerator_device),
        )
      with set_unflatten_seq_len(len_b):
        out_b = attn(
            x[len_a:],
            None,
            torch.arange(len_b, device=self.accelerator_device),
        )

    tol = 5e-3 if self._is_tpu() else 1e-5
    test_utils.check_equivalence(
        out_packed[:len_a].cpu(),
        out_a.cpu(),
        atol=tol,
        rtol=tol,
        check_name="Packed Doc A",
    )
    test_utils.check_equivalence(
        out_packed[len_a:].cpu(),
        out_b.cpu(),
        atol=tol,
        rtol=tol,
        check_name="Packed Doc B",
    )
    self.assertGreater(
        _relative_l2_error(out_unpacked[len_a:], out_b),
        0.1,
        msg="Negative control failed: unpacked positions did not change Doc B",
    )

  def test_splash_attention_matches_sdpa_on_packed_documents(self):
    """Tests Splash Attention patch on TpuQwen35Attention vs CPU SDPA."""
    if not self._is_tpu():
      self.skipTest("Splash Attention is a TPU-only Pallas kernel.")
    config = qwen3_5_tpu.qwen3_5_configs["debugmodel"]
    batch, seq_len = 2, 256
    attn_cpu = config.layers[3].attention.build()
    attn_cpu.init_states(buffer_device=torch.device("cpu"))
    attn_cpu = attn_cpu.cpu()
    attn_tpu = copy.deepcopy(attn_cpu).to(self.accelerator_device)

    sa_config = SplashAttentionKernelConfig(
        use_splash_attention_kernel=True,
        sa_block_q=128,
        sa_block_kv=128,
        sa_block_kv_compute=128,
        sa_block_q_dkv=128,
        sa_block_kv_dkv=128,
        sa_block_kv_dkv_compute=128,
        sa_block_q_dq=128,
        sa_block_kv_dq=128,
    )
    workarounds.use_splash_attention_patch(attn_tpu, sa_config)

    x_cpu, x_tpu = self._create_input_tensor_cpu_device(
        (batch * seq_len, config.dim), requires_grad=True
    )
    positions_cpu = torch.cat([
        torch.arange(100),
        torch.arange(seq_len - 100),
        torch.arange(seq_len),
    ])
    positions_tpu = positions_cpu.to(self.accelerator_device)

    with set_unflatten_seq_len(seq_len):
      out_cpu = attn_cpu(x_cpu, None, positions_cpu)
      out_tpu = attn_tpu(x_tpu, None, positions_tpu)
      with torch.no_grad():
        out_cpu_unpacked = attn_cpu(
            x_cpu, None, torch.arange(seq_len).repeat(batch)
        )

    self.assertGreater(_relative_l2_error(out_cpu_unpacked, out_cpu), 0.1)
    self.assertLessEqual(_relative_l2_error(out_tpu, out_cpu), 1e-2)

    grad_out = torch.randn_like(out_cpu)
    out_cpu.backward(grad_out)
    out_tpu.backward(grad_out.to(self.accelerator_device))
    self.assertLessEqual(_relative_l2_error(x_tpu.grad, x_cpu.grad), 2.5e-2)

  def test_qwen3_5_configs_and_recipes(self):
    """Tests flavor geometries, TPU converters, and recipe configurations."""
    moe_35b = qwen3_5_tpu.qwen3_5_configs["35B-A3B"]
    self.assertIsNone(moe_35b.vision_encoder)
    self.assertLen(moe_35b.layers, 40)
    self.assertEqual(moe_35b.dim, 2048)
    self.assertEqual(moe_35b.vocab_size, 248320)
    gdn_layers = [l for l in moe_35b.layers if l.delta_net is not None]
    attn_layers = [l for l in moe_35b.layers if l.attention is not None]
    self.assertLen(gdn_layers, 30)
    self.assertLen(attn_layers, 10)
    self.assertEqual(attn_layers[0].attention.head_dim, 256)
    self.assertEqual(attn_layers[0].attention.n_heads, 16)
    self.assertEqual(attn_layers[0].attention.n_kv_heads, 2)
    self.assertEqual(moe_35b.layers[0].moe.num_experts, 256)
    self.assertEqual(moe_35b.layers[0].moe.router.top_k, 8)
    self.assertLen(qwen3_5_tpu.qwen3_5_configs["35B-A3B-proxy"].layers, 4)

    spec = qwen3_5_tpu.model_registry("35B-A3B")
    self.assertEqual(spec.name, "qwen3_5")
    self.assertIs(spec.parallelize_fn, parallelize_qwen3_5)
    gt_spec = gt_qwen3_5.model_registry("35B-A3B")
    self.assertEqual(gt_spec.name, "qwen3_5")
    self.assertIs(gt_spec.parallelize_fn, gt_qwen3_5.parallelize_qwen3_5)

    def gdn_backends(cfg):
      return [
          kernel.backend
          for _, kernel, _, _ in cfg.model_spec.model.traverse(
              gated_delta.TpuGatedDeltaKernel.Config
          )
      ]

    def moe_kernels(cfg):
      experts = [
          (
              e.use_pallas_grouped_mm,
              e.use_pallas_paired_grouped_mm,
              e.use_kernel_native_weight_layout,
          )
          for _, e, _, _ in cfg.model_spec.model.traverse(
              TpuGroupedExperts.Config
          )
      ]
      fused = [
          d.fuse_route_score_backward
          for _, d, _, _ in cfg.model_spec.model.traverse(
              TpuAllToAllTokenDispatcher.Config
          )
      ]
      return experts, fused

    self.assertEqual(
        gdn_backends(config_registry.qwen35_moe_35b()), ["xla"] * 30
    )
    with self.assertRaisesRegex(ValueError, "unknown TpuGatedDeltaKernel"):
      gated_delta.TpuGatedDeltaKernel.Config(backend="pallas").build()

    # The benchmark recipe: ZeRO-3, graph split (2 layers per executable),
    # fused AdamW outside the graph, Tokamax GatedDeltaNet, Pallas MoE / CE
    # kernels, and Splash attention.
    bench_cfg = config_registry.qwen35_moe_35b_split_tokamax_benchmark()
    self.assertEqual(bench_cfg.model_spec.flavor, "35B-A3B")
    self.assertEqual(gdn_backends(bench_cfg), ["tokamax"] * 30)
    self.assertEqual(
        moe_kernels(bench_cfg), ([(True, True, True)] * 40, [True] * 40)
    )
    self.assertEqual(bench_cfg.parallelism.fsdp_reshard_after_forward, "always")
    self.assertEqual(bench_cfg.parallelism.expert_parallel_degree, 1)
    self.assertEqual(
        bench_cfg.training.num_tokens_per_microbatch_per_dp_rank, 16 * 2048
    )
    self.assertEqual(bench_cfg.training.dtype, "bfloat16")
    self.assertEqual(bench_cfg.lr_scheduler.warmup_steps, 200)
    self.assertEqual(bench_cfg.metrics.log_freq, 5)
    tpu_config = bench_cfg.tpu_config
    self.assertTrue(tpu_config.use_graph_trainer)
    self.assertEqual(tpu_config.graph_trainer_memory_policy, "full")
    self.assertTrue(tpu_config.use_graph_split)
    self.assertEqual(tpu_config.graph_split.forward_layers_per_executable, 2)
    self.assertEqual(tpu_config.graph_split.backward_layers_per_executable, 2)
    self.assertEqual(
        tpu_config.graph_split.backward_partition_mode, "fsdp_pipeline"
    )
    self.assertFalse(tpu_config.optimizer_in_graph)
    self.assertTrue(tpu_config.compile_grad_norm)
    self.assertEqual(bench_cfg.optimizer.implementation, "fused")
    self.assertTrue(
        bench_cfg.splash_attention_kernel.use_splash_attention_kernel
    )
    self.assertEqual(bench_cfg.splash_attention_kernel.sa_block_q_dkv, 1024)
    self.assertIsInstance(bench_cfg.loss, PallasCrossEntropyLoss.Config)

    # The 4-layer proxy keeps the recipe, with 1 layer per executable.
    proxy_cfg = config_registry.qwen35_moe_35b_proxy_split_tokamax_benchmark()
    self.assertEqual(proxy_cfg.model_spec.flavor, "35B-A3B-proxy")
    self.assertEqual(gdn_backends(proxy_cfg), ["tokamax"] * 3)
    self.assertEqual(
        moe_kernels(proxy_cfg), ([(True, True, True)] * 4, [True] * 4)
    )
    self.assertEqual(proxy_cfg.training, bench_cfg.training)
    self.assertEqual(
        proxy_cfg.tpu_config.graph_split.forward_layers_per_executable, 1
    )

    apply_overrides(bench_cfg.override, bench_cfg)
    with torch.device("meta"):
      meta_model = bench_cfg.model_spec.model.build()
    workarounds._check_fused_loss_logit_transform(meta_model, bench_cfg)  # pylint: disable=protected-access
    inner_experts = meta_model.layers["0"].moe.routed_experts.inner_experts
    self.assertIsInstance(inner_experts, TpuGroupedExperts)

    # 35B-A3B parameter and MFU accounting. Active params exclude the input
    # embedding and count the top-8 of 256 routed experts. FLOPs per token are
    # 6 * active params, plus 10 quadratic attention layers
    # (6 * 16 heads * (256 + 256) * seq_len each), plus 30 delta-rule layers
    # (6 * 3 * 32 value heads * 128 * 128 each).
    seq_len = bench_cfg.training.max_context_length
    self.assertEqual(seq_len, 2048)
    total_params, flops_per_token = (
        bench_cfg.model_spec.model.get_nparams_and_flops(meta_model, seq_len)
    )
    self.assertEqual(total_params, 34_660_610_688)
    self.assertEqual(
        flops_per_token,
        6 * 2_946_429_568
        + 10 * 6 * 16 * (256 + 256) * seq_len
        + 30 * 6 * 3 * 32 * 128 * 128,
    )

    # The recipes parse from the command line.
    parsed = ConfigManager().parse_args([
        "--module=torchtitan.experiments.tpu.qwen3_5",
        "--config=qwen35_moe_35b_split_tokamax_benchmark",
        "--training.steps=20",
        "--training.num_tokens_per_microbatch_per_dp_rank=65536",
    ])
    self.assertEqual(parsed.training.steps, 20)
    self.assertEqual(
        parsed.training.num_tokens_per_microbatch_per_dp_rank, 65536
    )
    self.assertEqual(parsed.model_spec.flavor, "35B-A3B")


if __name__ == "__main__":
  absltest.main()
