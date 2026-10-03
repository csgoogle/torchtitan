"""Unit and numerical tests for Gemma 4 in torchtitan.experiments.tpu.examples.gemma4."""

import copy
from functools import partial
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest as BaseDeviceTest
from torchtitan.experiments.tpu.examples.gemma4 import (
    _build_flavor_config,
    gemma4_configs,
)
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch
from torchtitan.experiments.tpu.examples.gemma4.config_registry import (
    gemma4_26b_a4b,
    gemma4_31b,
    gemma4_debugmodel,
    gemma4_moe_debug,
    gemma4_moe_testmodel,
)
from torchtitan.experiments.tpu.examples.gemma4.model import (
    Gemma4Attention,
    Gemma4Embedding,
    Gemma4FeedForward,
    Gemma4MoE,
    Gemma4Model,
    Gemma4RoPE,
    Gemma4Router,
    Gemma4TransformerBlock,
    QKVLinearKeqV,
    SlidingWindowScaledDotProductAttention,
)
from torchtitan.experiments.tpu.examples.gemma4.state_dict_adapter import (
    Gemma4StateDictAdapter,
)
from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.models.common.attention import (
    QKVLinear,
    ScaledDotProductAttention,
)
from torchtitan.models.common.config_utils import make_token_dispatcher_config
from torchtitan.models.common.param_init import skip_param_init
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm


_NORM_INIT = {"weight": nn.init.ones_}
_LINEAR_INIT = {"weight": partial(nn.init.trunc_normal_, std=0.02)}
_EXPERTS_INIT = {
    "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
    "w2_EDF": partial(nn.init.trunc_normal_, std=0.02),
    "w3_EFD": partial(nn.init.trunc_normal_, std=0.02),
}


class TestTpuGroupedExpertsActivation(BaseDeviceTest):
  """Tests gelu_tanh activation support in TpuGroupedExperts."""

  def test_gelu_tanh_vs_silu_forward(self):
    num_experts, dim, hidden_dim = 4, 16, 32
    x = torch.randn(8, dim, device=self.accelerator_device, dtype=torch.float32)
    num_tokens_per_expert = torch.tensor(
        [2, 2, 2, 2], device=self.accelerator_device, dtype=torch.int32
    )

    cfg_gelu = TpuGroupedExperts.Config(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        use_pallas_grouped_mm=False,
        activation="gelu_tanh",
        token_dispatcher=make_token_dispatcher_config(
            num_experts=num_experts,
            top_k=2,
            comm_backend="standard",
            non_blocking_capacity_factor=None,
            hidden_dim=dim,
        ),
        param_init=_EXPERTS_INIT,
    )
    experts_gelu = cfg_gelu.build().to(self.accelerator_device)
    experts_gelu.init_states()

    out_gelu = experts_gelu._experts_forward(x, num_tokens_per_expert)

    # Compute reference gelu_tanh in bfloat16 (matching GroupedExperts compute)
    expected_chunks = []
    for i, chunk in enumerate(x.split([2, 2, 2, 2], dim=0)):
      c_bf16 = chunk.bfloat16()
      w1 = experts_gelu.w1_EFD[i].bfloat16().t()
      w2 = experts_gelu.w2_EDF[i].bfloat16().t()
      w3 = experts_gelu.w3_EFD[i].bfloat16().t()
      h = F.gelu(c_bf16 @ w1, approximate="tanh") * (c_bf16 @ w3)
      expected_chunks.append((h @ w2).type_as(x))
    expected = torch.cat(expected_chunks, dim=0)
    torch.testing.assert_close(out_gelu, expected, atol=2e-2, rtol=2e-2)

  def test_converter_propagates_gemma4_gelu_tanh(self):
    config = copy.deepcopy(gemma4_configs["testmodel_moe"])
    converter = TpuGroupedExpertsConverter(TpuGroupedExpertsConverter.Config())
    config = converter.convert(config)
    for layer in config.layers:
      if layer.moe is not None:
        self.assertIsInstance(layer.moe.experts, TpuGroupedExperts.Config)
        self.assertEqual(layer.moe.experts.activation, "gelu_tanh")

  def test_config_registry_valid(self):
    for fn in (
        gemma4_debugmodel,
        gemma4_moe_testmodel,
        gemma4_moe_debug,
        gemma4_31b,
        gemma4_26b_a4b,
    ):
      cfg = fn()
      self.assertEqual(cfg.model_spec.name, "gemma4")
      self.assertIn(cfg.model_spec.flavor, gemma4_configs)
      self.assertFalse(cfg.model_spec.model.enable_weight_tying)

  def test_build_flavor_config_weight_tying(self):
    tied = _build_flavor_config("testmodel")
    untied = _build_flavor_config("testmodel", enable_weight_tying=False)
    self.assertTrue(tied.enable_weight_tying)
    self.assertFalse(untied.enable_weight_tying)
    with torch.device("meta"):
      tied_model = tied.build()
      untied_model = untied.build()
    self.assertIs(tied_model.tok_embeddings.weight, tied_model.lm_head.weight)
    self.assertIsNot(
        untied_model.tok_embeddings.weight, untied_model.lm_head.weight
    )
    # An untied embedding must get a real init, not skip_param_init.
    self.assertIsNot(
        untied.tok_embeddings.param_init["weight"], skip_param_init
    )


class TestGemma4StateDictAdapter(absltest.TestCase):
  """Tests HF <-> TorchTitan state_dict conversion for Gemma 4."""

  def test_dense_state_dict_roundtrip(self):
    config = _build_flavor_config("testmodel")
    with torch.device("meta"):
      model = config.build()
    tt_sd = {}
    for k, v in model.state_dict().items():
      tt_sd[k] = torch.randn(v.shape, dtype=torch.float32)
    if config.enable_weight_tying:
      tt_sd["lm_head.weight"] = tt_sd["tok_embeddings.weight"]

    adapter = Gemma4StateDictAdapter(config, hf_assets_path=None)
    hf_sd = adapter.to_hf(tt_sd)

    self.assertIn("model.language_model.embed_tokens.weight", hf_sd)
    self.assertIn("model.language_model.norm.weight", hf_sd)
    self.assertIn(
        "model.language_model.layers.0.self_attn.q_proj.weight", hf_sd
    )
    self.assertIn(
        "model.language_model.layers.0.self_attn.k_proj.weight", hf_sd
    )
    self.assertIn(
        "model.language_model.layers.0.self_attn.v_proj.weight", hf_sd
    )
    self.assertIn("model.language_model.layers.0.layer_scalar", hf_sd)
    # Global K=V layer (layer 1 in testmodel)
    self.assertIn(
        "model.language_model.layers.1.self_attn.k_proj.weight", hf_sd
    )
    self.assertNotIn(
        "model.language_model.layers.1.self_attn.v_proj.weight", hf_sd
    )

    roundtrip_sd = adapter.from_hf(hf_sd)
    self.assertEqual(set(roundtrip_sd.keys()), set(tt_sd.keys()))
    for k in tt_sd:
      torch.testing.assert_close(roundtrip_sd[k], tt_sd[k])

  def test_moe_state_dict_roundtrip_and_vision_skip(self):
    config = _build_flavor_config("testmodel_moe")
    with torch.device("meta"):
      model = config.build()
    tt_sd = {}
    for k, v in model.state_dict().items():
      tt_sd[k] = torch.randn(v.shape, dtype=torch.float32)
    if config.enable_weight_tying:
      tt_sd["lm_head.weight"] = tt_sd["tok_embeddings.weight"]

    adapter = Gemma4StateDictAdapter(config, hf_assets_path=None)
    hf_sd = adapter.to_hf(tt_sd)

    self.assertIn(
        "model.language_model.layers.0.experts.gate_up_proj", hf_sd
    )
    self.assertIn("model.language_model.layers.0.experts.down_proj", hf_sd)
    self.assertIn("model.language_model.layers.0.router.proj.weight", hf_sd)
    self.assertIn("model.language_model.layers.0.router.scale", hf_sd)
    self.assertIn(
        "model.language_model.layers.0.router.per_expert_scale", hf_sd
    )
    self.assertIn(
        "model.language_model.layers.0.post_feedforward_layernorm_1.weight",
        hf_sd,
    )
    self.assertIn(
        "model.language_model.layers.0.post_feedforward_layernorm_2.weight",
        hf_sd,
    )
    self.assertIn(
        "model.language_model.layers.0.pre_feedforward_layernorm_2.weight",
        hf_sd,
    )

    # Add dummy vision/audio keys and verify they are ignored on from_hf
    hf_sd["model.vision_tower.encoder.layers.0.weight"] = torch.randn(4, 4)
    hf_sd["model.audio_tower.encoder.layers.0.weight"] = torch.randn(4, 4)
    hf_sd["model.embed_vision.weight"] = torch.randn(4, 4)
    hf_sd["model.embed_audio.weight"] = torch.randn(4, 4)
    hf_sd["multi_modal_projector.weight"] = torch.randn(4, 4)

    roundtrip_sd = adapter.from_hf(hf_sd)
    self.assertEqual(set(roundtrip_sd.keys()), set(tt_sd.keys()))
    for k in tt_sd:
      torch.testing.assert_close(roundtrip_sd[k], tt_sd[k])


class TestGemma4SdpaSlidingWindow(absltest.TestCase):
  """Sliding layers must apply the HF window on the plain SDPA backend."""

  _WINDOW = 4
  _SEQ_LEN = 12  # Longer than the window, so the window actually masks keys.
  _DIM = 32

  def _attention_with_mask(self, attn, mask_LL):
    """Copy of ``attn`` whose inner attention uses an explicit boolean mask."""
    ref = copy.deepcopy(attn)

    def masked_sdpa(
        q, k, v, *, scale=None, enable_gqa=False, **unused_kwargs
    ):
      return F.scaled_dot_product_attention(
          q.transpose(1, 2),
          k.transpose(1, 2),
          v.transpose(1, 2),
          attn_mask=mask_LL,
          scale=scale,
          enable_gqa=enable_gqa,
      ).transpose(1, 2)

    ref.inner_attention.forward = masked_sdpa
    return ref

  def test_sliding_window_applied_beyond_window(self):
    head_dim = 8
    cfg = Gemma4Attention.Config(
        dim=self._DIM,
        n_heads=4,
        n_kv_heads=2,
        head_dim=head_dim,
        is_sliding=True,
        sliding_window_size=self._WINDOW,
        qk_norm=RMSNorm.Config(
            normalized_shape=head_dim, eps=1e-6, param_init=_NORM_INIT
        ),
        v_norm=RMSNorm.Config(
            normalized_shape=head_dim, eps=1e-6, elementwise_affine=False
        ),
        qkv_linear=QKVLinear.Config(
            head_dim=head_dim,
            wq=Linear.Config(
                in_features=self._DIM, out_features=32, param_init=_LINEAR_INIT
            ),
            wkv=Linear.Config(
                in_features=self._DIM, out_features=16, param_init=_LINEAR_INIT
            ),
        ),
        wo=Linear.Config(
            in_features=32, out_features=self._DIM, param_init=_LINEAR_INIT
        ),
        inner_attention=ScaledDotProductAttention.Config(),
        rope=Gemma4RoPE.Config(
            dim=head_dim, max_seq_len=self._SEQ_LEN, theta=10000.0
        ),
    )
    torch.manual_seed(0)
    attn = cfg.build()
    attn.init_states()

    # HF Gemma 4: query q attends to key kv iff 0 <= q - kv < sliding_window.
    dist = (
        torch.arange(self._SEQ_LEN)[:, None]
        - torch.arange(self._SEQ_LEN)[None, :]
    )
    hf_mask = (dist >= 0) & (dist < self._WINDOW)
    causal_mask = dist >= 0

    x = torch.randn(2, self._SEQ_LEN, self._DIM)
    with torch.no_grad():
      actual = attn(x, attention_masks=None)
      expected = self._attention_with_mask(attn, hf_mask)(
          x, attention_masks=None
      )
      full_causal = self._attention_with_mask(attn, causal_mask)(
          x, attention_masks=None
      )

    torch.testing.assert_close(actual, expected)
    # Rows that see fewer than W tokens are unaffected by the window; every
    # later row must differ from full causal attention.
    w = self._WINDOW
    torch.testing.assert_close(actual[:, :w], full_causal[:, :w])
    row_diff = (actual - full_causal).abs().amax(dim=(0, 2))
    self.assertTrue(
        bool((row_diff[w:] > 1e-4).all()),
        f"Rows >= window match full causal attention: {row_diff[w:]}",
    )
    self.assertIsInstance(
        attn.inner_attention, SlidingWindowScaledDotProductAttention
    )

  def test_flavor_sliding_layers_use_windowed_sdpa(self):
    config = _build_flavor_config("testmodel", attn_backend="sdpa")
    with torch.device("meta"):
      model = config.build()
    for layer in model.layers.values():
      inner = layer.attention.inner_attention
      if layer.attention.is_sliding:
        self.assertIsInstance(inner, SlidingWindowScaledDotProductAttention)
        self.assertEqual(
            inner.sliding_window_size,
            layer.attention.sliding_window_size - 1,
        )
      else:
        self.assertIs(type(inner), ScaledDotProductAttention)


class TestGemma4Components(BaseDeviceTest):
  """Tests individual Gemma 4 modules and numerical parity on CPU/TPU."""

  def test_gemma4_embedding_scaling(self):
    vocab_size, dim = 64, 16
    cfg = Gemma4Embedding.Config(
        num_embeddings=vocab_size,
        embedding_dim=dim,
        param_init={"weight": partial(nn.init.normal_, mean=0.0, std=1.0)},
    )
    emb = cfg.build().to(self.accelerator_device)
    emb.init_states()

    tokens = torch.tensor(
        [[1, 2, 3], [4, 5, 6]], device=self.accelerator_device, dtype=torch.long
    )
    out = emb(tokens)
    expected = F.embedding(tokens, emb.weight) * (dim**0.5)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

  def test_gemma4_rope_proportional(self):
    dim = 16
    max_seq_len = 32
    theta = 1000000.0
    partial_rotary_factor = 0.25
    rope_dims = int(dim * partial_rotary_factor)  # 4
    nope_dims = dim - rope_dims  # 12

    cfg = Gemma4RoPE.Config(
        dim=dim,
        max_seq_len=max_seq_len,
        theta=theta,
        partial_rotary_factor=partial_rotary_factor,
    )
    rope = cfg.build().to(self.accelerator_device)
    rope.init_states()

    self.assertEqual(rope.cache.shape, (max_seq_len, dim * 2))
    cos, sin = rope.cache.chunk(2, dim=-1)
    torch.testing.assert_close(
        cos[:, rope_dims // 2 : rope_dims // 2 + nope_dims // 2],
        torch.ones(max_seq_len, nope_dims // 2, device=self.accelerator_device),
    )
    torch.testing.assert_close(
        sin[:, rope_dims // 2 : rope_dims // 2 + nope_dims // 2],
        torch.zeros(max_seq_len, nope_dims // 2, device=self.accelerator_device),
    )

    q = torch.randn(2, 8, 4, dim, device=self.accelerator_device)
    k = torch.randn(2, 8, 2, dim, device=self.accelerator_device)
    q_rot, k_rot = rope(q, k)

    torch.testing.assert_close(
        q_rot[..., rope_dims // 2 : rope_dims // 2 + nope_dims // 2],
        q[..., rope_dims // 2 : rope_dims // 2 + nope_dims // 2],
    )
    torch.testing.assert_close(
        q_rot[..., dim // 2 + rope_dims // 2 :],
        q[..., dim // 2 + rope_dims // 2 :],
    )
    self.assertFalse(
        torch.allclose(
            q_rot[:, 1:, :, : rope_dims // 2],
            q[:, 1:, :, : rope_dims // 2],
        )
    )

  def test_gemma4_rope_full(self):
    dim = 16
    max_seq_len = 32
    cfg = Gemma4RoPE.Config(
        dim=dim,
        max_seq_len=max_seq_len,
        theta=10000.0,
        partial_rotary_factor=1.0,
    )
    rope = cfg.build().to(self.accelerator_device)
    rope.init_states()

    freqs = 1.0 / (
        10000.0
        ** (
            torch.arange(
                0, dim, 2, dtype=torch.float32, device=self.accelerator_device
            )
            / dim
        )
    )
    t = torch.arange(max_seq_len, dtype=torch.float32, device=self.accelerator_device)
    freqs_outer = torch.outer(t, freqs)
    emb = torch.cat((freqs_outer, freqs_outer), dim=-1)
    expected_cache = torch.cat((emb.cos(), emb.sin()), dim=-1)
    torch.testing.assert_close(rope.cache, expected_cache, atol=1e-5, rtol=1e-5)

  def test_qkv_linear_keqv_shapes(self):
    dim = 32
    head_dim = 8
    n_heads = 4
    n_kv_heads = 2

    cfg_keqv = QKVLinearKeqV.Config(
        head_dim=head_dim,
        wq=Linear.Config(
            in_features=dim,
            out_features=n_heads * head_dim,
            param_init=_LINEAR_INIT,
        ),
        wkv=Linear.Config(
            in_features=dim,
            out_features=n_kv_heads * head_dim,
            param_init=_LINEAR_INIT,
        ),
    )
    qkv_keqv = cfg_keqv.build().to(self.accelerator_device)
    qkv_keqv.init_states()
    self.assertFalse(hasattr(qkv_keqv, "wv"))

    x = torch.randn(2, 6, dim, device=self.accelerator_device)
    xq, xk, xv = qkv_keqv(x)
    self.assertEqual(xq.shape, (2, 6, n_heads, head_dim))
    self.assertEqual(xk.shape, (2, 6, n_kv_heads, head_dim))
    self.assertIs(xk, xv)

  def test_gemma4_attention_sliding_and_global(self):
    dim = 32
    sliding_cfg = Gemma4Attention.Config(
        dim=dim,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        is_sliding=True,
        sliding_window_size=4,
        qk_norm=RMSNorm.Config(
            normalized_shape=8, eps=1e-6, param_init=_NORM_INIT
        ),
        v_norm=RMSNorm.Config(
            normalized_shape=8, eps=1e-6, elementwise_affine=False
        ),
        qkv_linear=QKVLinear.Config(
            head_dim=8,
            wq=Linear.Config(
                in_features=dim, out_features=32, param_init=_LINEAR_INIT
            ),
            wkv=Linear.Config(
                in_features=dim, out_features=16, param_init=_LINEAR_INIT
            ),
        ),
        wo=Linear.Config(
            in_features=32, out_features=dim, param_init=_LINEAR_INIT
        ),
        inner_attention=ScaledDotProductAttention.Config(),
        rope=Gemma4RoPE.Config(
            dim=8, max_seq_len=16, theta=10000.0, partial_rotary_factor=1.0
        ),
    )
    sliding_attn = sliding_cfg.build().to(self.accelerator_device)
    sliding_attn.init_states()

    x = torch.randn(2, 10, dim, device=self.accelerator_device)
    out_sliding = sliding_attn(x, attention_masks=None)
    self.assertEqual(out_sliding.shape, (2, 10, dim))

    # Global attention with K=V
    global_cfg = Gemma4Attention.Config(
        dim=dim,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        is_sliding=False,
        sliding_window_size=None,
        qk_norm=RMSNorm.Config(
            normalized_shape=16, eps=1e-6, param_init=_NORM_INIT
        ),
        v_norm=RMSNorm.Config(
            normalized_shape=16, eps=1e-6, elementwise_affine=False
        ),
        qkv_linear=QKVLinearKeqV.Config(
            head_dim=16,
            wq=Linear.Config(
                in_features=dim, out_features=64, param_init=_LINEAR_INIT
            ),
            wkv=Linear.Config(
                in_features=dim, out_features=32, param_init=_LINEAR_INIT
            ),
        ),
        wo=Linear.Config(
            in_features=64, out_features=dim, param_init=_LINEAR_INIT
        ),
        inner_attention=ScaledDotProductAttention.Config(),
        rope=Gemma4RoPE.Config(
            dim=16,
            max_seq_len=16,
            theta=1000000.0,
            partial_rotary_factor=0.25,
        ),
    )
    global_attn = global_cfg.build().to(self.accelerator_device)
    global_attn.init_states()

    out_global = global_attn(x, attention_masks=None)
    self.assertEqual(out_global.shape, (2, 10, dim))

  def test_gemma4_router_math(self):
    dim = 16
    num_experts = 8
    top_k = 2
    eps = 1e-6
    cfg = Gemma4Router.Config(
        dim=dim,
        num_experts=num_experts,
        top_k=top_k,
        score_func="softmax",
        route_norm=True,
        norm=RMSNorm.Config(
            normalized_shape=dim, eps=eps, elementwise_affine=False
        ),
        gate=Linear.Config(
            in_features=dim,
            out_features=num_experts,
            param_init=_LINEAR_INIT,
        ),
        param_init={"scale": nn.init.ones_, "per_expert_scale": nn.init.ones_},
    )
    router = cfg.build().to(self.accelerator_device)
    router.init_states()
    with torch.no_grad():
      router.scale.copy_(torch.linspace(0.5, 1.5, dim, device=self.accelerator_device))
      router.per_expert_scale.copy_(
          torch.linspace(0.8, 1.2, num_experts, device=self.accelerator_device)
      )

    x = torch.randn(2, 5, dim, device=self.accelerator_device)
    top_scores, selected_experts_indices, _ = router(x)

    # Reference calculation
    x_normed = F.rms_norm(x.float(), (dim,), weight=None, eps=eps).to(x.dtype)
    x_scaled = x_normed * (dim**-0.5) * router.scale.to(x.dtype)
    logits = router.gate(x_scaled)
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    ref_weights, ref_indices = torch.topk(probs, k=top_k, dim=-1)
    ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)
    ref_weights = (
        ref_weights.to(x.dtype)
        * router.per_expert_scale[ref_indices].to(x.dtype)
    )

    torch.testing.assert_close(
        top_scores.to(x.dtype), ref_weights, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(selected_experts_indices, ref_indices)

  def test_gemma4_grouped_experts_gelu_tanh_with_empty_experts(self):
    dim, hidden_dim, num_experts = 16, 32, 4
    cfg = TpuGroupedExperts.Config(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        activation="gelu_tanh",
        use_pallas_grouped_mm=False,
        token_dispatcher=make_token_dispatcher_config(
            num_experts=num_experts,
            top_k=2,
            comm_backend="standard",
            non_blocking_capacity_factor=None,
            hidden_dim=dim,
        ),
        param_init=_EXPERTS_INIT,
    )
    experts = cfg.build().to(self.accelerator_device)
    experts.init_states()

    # Experts 0 and 2 receive no tokens.
    counts = [0, 5, 0, 3]
    x = torch.randn(sum(counts), dim, device=self.accelerator_device)
    num_tokens_per_expert = torch.tensor(
        counts, device=self.accelerator_device, dtype=torch.int32
    )
    out = experts._experts_forward(x, num_tokens_per_expert)
    self.assertEqual(out.shape, (sum(counts), dim))

    expected_chunks = []
    for i, chunk in enumerate(x.split(counts, dim=0)):
      c_bf16 = chunk.bfloat16()
      w1 = experts.w1_EFD[i].bfloat16().t()
      w2 = experts.w2_EDF[i].bfloat16().t()
      w3 = experts.w3_EFD[i].bfloat16().t()
      h = F.gelu(c_bf16 @ w1, approximate="tanh") * (c_bf16 @ w3)
      expected_chunks.append((h @ w2).type_as(x))
    expected = torch.cat(expected_chunks, dim=0)
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)

  def test_gemma4_transformer_block_parallel_moe_and_layer_scalar(self):
    config = _build_flavor_config("testmodel_moe")
    block = config.layers[0].build().to(self.accelerator_device)
    block.init_states()

    x = torch.randn(2, 8, config.dim, device=self.accelerator_device)
    out = block(x, attention_masks=None)
    self.assertEqual(out.shape, (2, 8, config.dim))

    # If layer_scalar is set to 0, output must be all zeros
    with torch.no_grad():
      block.layer_scalar.zero_()
    out_zero = block(x, attention_masks=None)
    torch.testing.assert_close(out_zero, torch.zeros_like(out_zero))

  @parameterized.named_parameters(
      ("dense_testmodel", "testmodel"),
      ("moe_testmodel", "testmodel_moe"),
  )
  def test_gemma4_model_forward_backward_and_softcap(self, flavor):
    config: Gemma4Model.Config = _build_flavor_config(flavor)
    model = config.build().to(self.accelerator_device)
    model.init_states()

    tokens = torch.randint(
        0, config.vocab_size, (2, 16), device=self.accelerator_device
    )
    logits = model(tokens)
    self.assertEqual(logits.shape, (2, 16, config.vocab_size))
    self.assertLessEqual(
        logits.abs().max().item(), config.final_logit_softcapping + 1e-4
    )

    loss = logits.sum()
    loss.backward()
    for name, param in model.named_parameters():
      self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
      self.assertTrue(
          torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
      )


  @parameterized.named_parameters(
      ("dense_testmodel", "testmodel", False),
      ("moe_testmodel", "testmodel_moe", True),
  )
  def test_gemma4_transformer_block_cpu_device_parity(self, flavor, is_moe):
    """Tests CPU vs. DEVICE forward and backward parity of Gemma4TransformerBlock."""
    use_cpu_safe_histc_patch()
    config = _build_flavor_config(flavor)
    batch, seq_len = 2, 8
    for layer_id, block_config in enumerate(config.layers):
      with self.subTest(layer_id=layer_id):
        block_cpu, block_device = self._setup_module_cpu_device(
            Gemma4TransformerBlock,
            lambda m: m.init_states(buffer_device=torch.device("cpu")),
            config=block_config,
        )
        x_cpu, x_device = self._create_input_tensor_cpu_device(
            (batch, seq_len, config.dim), requires_grad=True
        )
        positions_cpu = (
            torch.arange(seq_len, dtype=torch.long, device="cpu")
            .unsqueeze(0)
            .expand(batch, seq_len)
        )
        positions_device = positions_cpu.to(self.accelerator_device)

        out_cpu = block_cpu(x_cpu, None, positions_cpu)
        out_device = block_device(x_device, None, positions_device)
        test_utils.check_equivalence(
            out_device.cpu(),
            out_cpu,
            atol=1e-1,
            rtol=1e-1,
            check_name=f"TransformerBlock Layer {layer_id} Forward",
        )

        out_cpu.sum().backward()
        out_device.sum().backward()
        test_utils.check_equivalence(
            x_device.grad.cpu(),
            x_cpu.grad,
            atol=4e-1,
            rtol=1e-1,
            check_name=f"TransformerBlock Layer {layer_id} Input Grad",
        )
        for (name, p_cpu), p_device in zip(
            block_cpu.named_parameters(),
            block_device.parameters(),
        ):
          param_atol = 6.5
          param_rtol = 1e-1
          if "moe" in name:
            param_atol = 6.5
            param_rtol = 2e-1
          test_utils.check_equivalence(
              p_device.grad.cpu(),
              p_cpu.grad,
              atol=param_atol,
              rtol=param_rtol,
              check_name=f"TransformerBlock Layer {layer_id} Param Grad: {name}",
          )

  @parameterized.named_parameters(
      dict(
          testcase_name="dense_testmodel",
          flavor="testmodel",
          loss_atol=1e-1,
          loss_rtol=1e-2,
          grad_atol=2.5,
          grad_rtol=1e-2,
          param_atol=8e-2,
          param_rtol=1e-2,
      ),
      dict(
          testcase_name="moe_testmodel",
          flavor="testmodel_moe",
          loss_atol=1e-1,
          loss_rtol=1e-2,
          grad_atol=2.5,
          grad_rtol=1e-2,
          param_atol=8e-2,
          param_rtol=1e-2,
      ),
  )
  def test_gemma4_full_model_training_steps_cpu_device_parity(
      self,
      flavor,
      loss_atol,
      loss_rtol,
      grad_atol,
      grad_rtol,
      param_atol,
      param_rtol,
  ):
    """Tests CPU vs. DEVICE parity of full Gemma4 model after 3 training steps."""
    use_cpu_safe_histc_patch()
    config = _build_flavor_config(flavor)
    batch, seq_len = 2, 8

    model_cpu = Gemma4Model(config)
    model_cpu.init_weights(buffer_device=torch.device("cpu"))
    model_cpu = model_cpu.cpu()

    self._run_full_model_training_steps_parity_test(
        model_cpu=model_cpu,
        loss_atol=loss_atol,
        loss_rtol=loss_rtol,
        grad_atol=grad_atol,
        grad_rtol=grad_rtol,
        param_atol=param_atol,
        param_rtol=param_rtol,
        vocab_size=config.vocab_size,
        batch=batch,
        seq_len=seq_len,
    )


class TestGemma4EndToEndLossConvergence(BaseDeviceTest):
  """Tests multi-step training loss convergence for Dense and MoE Gemma 4."""

  @parameterized.named_parameters(
      ("dense_testmodel", "testmodel", False),
      ("moe_testmodel", "testmodel_moe", True),
  )
  def test_dense_and_moe_loss_decreases(self, flavor, use_tpu_experts):
    torch.manual_seed(42)
    config: Gemma4Model.Config = _build_flavor_config(flavor)
    if use_tpu_experts:
      converter = TpuGroupedExpertsConverter(
          TpuGroupedExpertsConverter.Config()
      )
      config = converter.convert(config)
      for layer in config.layers:
        if layer.moe is not None:
          layer.moe.experts.use_pallas_grouped_mm = False

    model = config.build().to(self.accelerator_device)
    model.init_states()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    tokens = torch.randint(
        0, config.vocab_size, (2, 16), device=self.accelerator_device
    )
    labels = torch.randint(
        0, config.vocab_size, (2, 16), device=self.accelerator_device
    )

    losses = []
    for _ in range(8):
      optimizer.zero_grad()
      logits = model(tokens)
      loss = cross_entropy_loss(logits, labels)
      loss.backward()
      optimizer.step()
      losses.append(loss.item())

    self.assertLess(
        losses[-1],
        losses[0],
        f"Expected loss to decrease over 8 steps for {flavor}, got {losses}",
    )

    for i, layer in enumerate(model.layers.values()):
      if layer.moe is not None:
        for attr in ("scale", "per_expert_scale"):
          grad = getattr(layer.moe.router, attr).grad
          self.assertIsNotNone(grad, f"Layer {i} router.{attr} grad is None")
          self.assertTrue(torch.isfinite(grad).all())
        self.assertIsNotNone(layer.moe.router.gate.weight.grad)


if __name__ == "__main__":
  absltest.main()
