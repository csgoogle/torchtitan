"""Tests for OLMo 2 in torchtitan.experiments.tpu.examples.olmo2.

Run as CI does: ``python -m <this module> --device=tpu`` (or ``--device=cpu``
with ``TORCH_DEVICE_BACKEND_AUTOLOAD=0``).
"""

import types

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F

from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.experiments.tpu.examples.olmo2 import build_olmo2_config
from torchtitan.experiments.tpu.examples.olmo2 import model_registry
from torchtitan.experiments.tpu.examples.olmo2 import Olmo2Attention
from torchtitan.experiments.tpu.examples.olmo2 import Olmo2StateDictAdapter
from torchtitan.experiments.tpu.examples.olmo2.config_registry import olmo2_1b
from torchtitan.experiments.tpu.examples.olmo2.config_registry import olmo2_debugmodel
from torchtitan.experiments.tpu.examples.olmo2.graph_trainer import GraphTrainerOlmo2Model
from torchtitan.experiments.tpu.examples.olmo2.graph_trainer import model_registry as graph_trainer_model_registry
from torchtitan.experiments.tpu.examples.olmo2.graph_trainer import parallelize_olmo2 as graph_trainer_parallelize_olmo2
from torchtitan.experiments.tpu.examples.olmo2.infra.parallelize import parallelize_olmo2
from torchtitan.experiments.tpu.graph_trainer.layer_stack import check_layers_are_stackable
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
import transformers

# allenai/OLMo-2-0425-1B config.json (transformers 4.50.3 export).
_OLMO2_1B_HF_CONFIG = dict(
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=16,
    num_attention_heads=16,
    num_key_value_heads=16,
    vocab_size=100352,
    rope_theta=500000,
    rms_norm_eps=1e-6,
    max_position_embeddings=4096,
    tie_word_embeddings=False,
)
# model.safetensors.index.json total_size (fp32 bytes) / 4.
_OLMO2_1B_NPARAMS = 1_484_916_736

# Tiny reference geometry: every OLMo 2 layer is the same type, so two layers
# suffice; head_dim 32.
_TINY = dict(dim=256, n_layers=2, n_heads=8, head_dim=32, hidden_dim=512)
_TINY_VOCAB = 2048


def _tiny_tt_config(n_kv_heads: int):
  return build_olmo2_config(
      **_TINY, n_kv_heads=n_kv_heads, vocab_size=_TINY_VOCAB, seq_len=64
  )


def _tiny_hf_model(n_kv_heads: int) -> transformers.PreTrainedModel:
  hf_config = transformers.Olmo2Config(
      vocab_size=_TINY_VOCAB,
      hidden_size=_TINY["dim"],
      intermediate_size=_TINY["hidden_dim"],
      num_hidden_layers=_TINY["n_layers"],
      num_attention_heads=_TINY["n_heads"],
      num_key_value_heads=n_kv_heads,
      max_position_embeddings=64,
      rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
      rms_norm_eps=1e-6,
      tie_word_embeddings=False,
      pad_token_id=None,
      attn_implementation="eager",
  )
  hf_model = transformers.Olmo2ForCausalLM(hf_config).float().eval()
  # Every norm weight starts at 1.0, where a dropped or misplaced norm scale is
  # invisible; perturb them.
  with torch.no_grad():
    for name, param in hf_model.named_parameters():
      if "norm" in name:
        param.copy_(1.0 + 0.1 * torch.randn_like(param))
  return hf_model


def _grad_errors(
    grads: dict[str, torch.Tensor], ref_grads: dict[str, torch.Tensor]
) -> dict[str, float]:
  """Per-tensor relative L2 error; absolute L2 where ``||ref|| < 1e-4``."""
  errors = {}
  for name, ref in ref_grads.items():
    diff = torch.linalg.vector_norm(grads[name] - ref).item()
    ref_norm = torch.linalg.vector_norm(ref).item()
    errors[name] = diff / ref_norm if ref_norm >= 1e-4 else diff
  return errors


def _assert_grads_close(
    grads: dict[str, torch.Tensor],
    ref_grads: dict[str, torch.Tensor],
    *,
    rel_l2: float,
    abs_floor: float = 1e-6,
) -> dict[str, float]:
  errors = _grad_errors(grads, ref_grads)
  failed = {}
  for name, error in errors.items():
    small = torch.linalg.vector_norm(ref_grads[name]).item() < 1e-4
    if error > (abs_floor if small else rel_l2):
      failed[name] = error
  if failed:
    raise AssertionError(f"Gradients outside bound {rel_l2}: {failed}")
  return errors


class TestOlmo2HFReferenceParity(BaseAcceleratorDeviceTest):
  """One forward + backward of Olmo2Model vs HF Olmo2ForCausalLM (CPU, fp32).

  Runs at fp32 matmul precision "highest", so the bounds can be tight: on v6e
  the measured errors are ~1.7e-6 (logits, gradients). At the TPU default
  precision (bf16 passes) the same comparison measures logits 1.6e-2 max abs
  and gradient relative L2 up to 1.85e-2, too close to any bound that would
  still catch a small scale bug.
  """

  def setUp(self):
    super().setUp()
    if not hasattr(transformers, "Olmo2ForCausalLM"):
      self.skipTest("transformers lacks Olmo2ForCausalLM")
    self.addCleanup(
        torch.set_float32_matmul_precision,
        torch.get_float32_matmul_precision(),
    )
    torch.set_float32_matmul_precision("highest")

  @parameterized.named_parameters(
      # MHA, as in the 1B; one document per row.
      dict(testcase_name="mha", n_kv_heads=8, doc_lengths=((16,), (16,))),
      # GQA (OLMo 2 32B is GQA): q_norm and k_norm have different widths.
      dict(testcase_name="gqa", n_kv_heads=2, doc_lengths=((16,), (16,))),
      # Packed rows: each document must see only itself, with RoPE positions
      # restarting at its start, i.e. equal HF run on each document alone.
      dict(testcase_name="packed", n_kv_heads=8, doc_lengths=((5, 11), (9, 7))),
  )
  def test_one_step_matches_hf(self, n_kv_heads, doc_lengths):
    torch.manual_seed(42)
    seq_len = sum(doc_lengths[0])
    batch_size = len(doc_lengths)

    hf_model = _tiny_hf_model(n_kv_heads)
    tt_config = _tiny_tt_config(n_kv_heads)
    tt_model = tt_config.build().to(self.accelerator_device)
    tt_model.init_states()

    adapter = Olmo2StateDictAdapter(tt_config, hf_assets_path=None)
    converted = adapter.from_hf(
        {k: v.detach().clone() for k, v in hf_model.state_dict().items()}
    )
    missing, unexpected = tt_model.load_state_dict(converted, strict=False)
    self.assertEmpty(missing)
    self.assertEmpty(unexpected)

    tokens = torch.randint(0, _TINY_VOCAB, (batch_size, seq_len))
    labels = torch.randint(0, _TINY_VOCAB, (batch_size, seq_len))
    positions = torch.cat(
        [torch.arange(n) for row in doc_lengths for n in row]
    )
    num_tokens = labels.numel()

    with set_unflatten_seq_len(seq_len):
      tt_logits = tt_model(
          tokens.view(-1).to(self.accelerator_device),
          positions=positions.to(self.accelerator_device),
      ).view(batch_size, seq_len, -1)
    tt_loss = (
        F.cross_entropy(
            tt_logits.flatten(0, 1).float(),
            labels.view(-1).to(self.accelerator_device),
            reduction="sum",
        )
        / num_tokens
    )
    tt_loss.backward()

    # The reference runs each document on its own.
    rows = []
    for row_tokens, row in zip(tokens, doc_lengths):
      docs = torch.split(row_tokens, list(row))
      rows.append(
          torch.cat([hf_model(input_ids=d[None]).logits[0] for d in docs])
      )
    hf_logits = torch.stack(rows)
    hf_loss = (
        F.cross_entropy(
            hf_logits.flatten(0, 1), labels.view(-1), reduction="sum"
        )
        / num_tokens
    )
    hf_loss.backward()

    names = dict(
        test_label=f"Olmo2Model ({self.accelerator_device.type})",
        ref_label="HF Olmo2ForCausalLM (CPU)",
    )
    tt_logits_cpu = tt_logits.detach().float().cpu()
    test_utils.check_equivalence(
        tt_logits_cpu,
        hf_logits.detach(),
        atol=1e-4,
        rtol=1e-4,
        check_name="Logits",
        **names,
    )
    test_utils.check_equivalence(
        tt_loss.detach().float().cpu(),
        hf_loss.detach(),
        atol=1e-5,
        rtol=1e-5,
        check_name="Loss",
        **names,
    )

    tt_grads = {
        n: p.grad.detach().float().cpu()
        for n, p in tt_model.named_parameters()
        if p.grad is not None
    }
    hf_grads = adapter.from_hf({
        n: p.grad.detach().float()
        for n, p in hf_model.named_parameters()
        if p.grad is not None
    })
    self.assertEqual(set(tt_grads), set(hf_grads))
    self.assertEqual(set(tt_grads), set(converted))
    errors = _assert_grads_close(tt_grads, hf_grads, rel_l2=1e-4)
    print(
        "measured: logits max abs"
        f" {(tt_logits_cpu - hf_logits.detach()).abs().max().item():.3e},"
        f" loss abs {abs(tt_loss.item() - hf_loss.item()):.3e},"
        f" grad rel L2 max {max(errors.values()):.3e}"
        f" ({max(errors, key=errors.get)})"
    )

    # Negative control: the bound must reject a zeroed and a sign-flipped
    # reference gradient.
    with self.assertRaises(AssertionError):
      _assert_grads_close(
          tt_grads,
          {n: torch.zeros_like(g) for n, g in hf_grads.items()},
          rel_l2=1e-4,
      )
    with self.assertRaises(AssertionError):
      _assert_grads_close(
          tt_grads, {n: -g for n, g in hf_grads.items()}, rel_l2=1e-4
      )

  def test_packed_rows_need_document_mask(self):
    """Control for the ``packed`` case: masking documents changes the output.

    Without ``positions`` the row is one causal sequence, so the second
    document's logits move; the packed parity case would fail if the mask
    were dropped.
    """
    torch.manual_seed(0)
    tt_config = _tiny_tt_config(n_kv_heads=8)
    tt_model = tt_config.build().to(self.accelerator_device)
    tt_model.init_states()
    tokens = torch.randint(0, _TINY_VOCAB, (16,)).to(self.accelerator_device)
    positions = torch.cat([torch.arange(5), torch.arange(11)]).to(
        self.accelerator_device
    )
    with torch.no_grad(), set_unflatten_seq_len(16):
      masked = tt_model(tokens, positions=positions).float().cpu()
      unmasked = tt_model(tokens).float().cpu()
    torch.testing.assert_close(masked[:5], unmasked[:5], atol=1e-4, rtol=1e-4)
    relative_change = (
        torch.linalg.vector_norm(masked[5:] - unmasked[5:])
        / torch.linalg.vector_norm(unmasked[5:])
    ).item()
    self.assertGreater(relative_change, 0.1)


class TestOlmo2StateDictAdapter(parameterized.TestCase):
  """HF <-> TorchTitan key mapping for OLMo 2."""

  @parameterized.named_parameters(
      dict(testcase_name="mha", n_kv_heads=8),
      dict(testcase_name="gqa", n_kv_heads=2),
  )
  def test_keys_match_hf_both_ways(self, n_kv_heads):
    tt_config = _tiny_tt_config(n_kv_heads)
    with torch.device("meta"):
      tt_model = tt_config.build()
    hf_sd = _tiny_hf_model(n_kv_heads).state_dict()
    adapter = Olmo2StateDictAdapter(tt_config, hf_assets_path=None)

    self.assertEqual(set(adapter.from_hf(hf_sd)), set(tt_model.state_dict()))
    self.assertEqual(set(adapter.to_hf(tt_model.state_dict())), set(hf_sd))
    for key, value in adapter.from_hf(hf_sd).items():
      self.assertEqual(value.shape, tt_model.state_dict()[key].shape, key)

  def test_roundtrip(self):
    config = model_registry("debugmodel").model
    with torch.device("meta"):
      model = config.build()
    tt_sd = {k: torch.randn(v.shape) for k, v in model.state_dict().items()}
    adapter = Olmo2StateDictAdapter(config, hf_assets_path=None)
    roundtrip = adapter.from_hf(adapter.to_hf(tt_sd))
    self.assertEqual(set(roundtrip), set(tt_sd))
    for key, value in tt_sd.items():
      torch.testing.assert_close(roundtrip[key], value, atol=0, rtol=0)

  def test_unknown_key_rejected(self):
    config = model_registry("debugmodel").model
    adapter = Olmo2StateDictAdapter(config, hf_assets_path=None)
    with self.assertRaises(KeyError):
      adapter.from_hf({"model.layers.0.input_layernorm.weight": torch.ones(1)})

  def test_weight_tying_rejected(self):
    config = model_registry("debugmodel").model
    config.enable_weight_tying = True
    with self.assertRaises(ValueError):
      Olmo2StateDictAdapter(config, hf_assets_path=None)


class TestOlmo2Flavors(absltest.TestCase):
  """Flavor constants, parameter counts and config validation."""

  def test_1b_matches_checkpoint_config(self):
    config = model_registry("1B").model
    hf = _OLMO2_1B_HF_CONFIG
    self.assertEqual(config.dim, hf["hidden_size"])
    self.assertEqual(config.vocab_size, hf["vocab_size"])
    self.assertLen(config.layers, hf["num_hidden_layers"])
    self.assertEqual(config.enable_weight_tying, hf["tie_word_embeddings"])
    self.assertEqual(config.norm.eps, hf["rms_norm_eps"])
    for layer in config.layers:
      attention = layer.attention
      self.assertIsInstance(attention, Olmo2Attention.Config)
      self.assertEqual(attention.n_heads, hf["num_attention_heads"])
      self.assertEqual(attention.n_kv_heads, hf["num_key_value_heads"])
      self.assertEqual(
          attention.head_dim, hf["hidden_size"] // hf["num_attention_heads"]
      )
      self.assertEqual(attention.rope.theta, hf["rope_theta"])
      self.assertEqual(
          attention.rope.max_context_length, hf["max_position_embeddings"]
      )
      self.assertEqual(attention.q_norm.normalized_shape, hf["hidden_size"])
      self.assertEqual(attention.k_norm.normalized_shape, hf["hidden_size"])
      self.assertEqual(layer.feed_forward.w1.out_features, hf["intermediate_size"])
      for norm in (attention.q_norm, layer.attention_norm, layer.ffn_norm):
        self.assertEqual(norm.eps, hf["rms_norm_eps"])

  def test_1b_parameter_count_matches_checkpoint(self):
    config = model_registry("1B").model
    with torch.device("meta"):
      model = config.build()
    nparams, _ = config.get_nparams_and_flops(model, seq_len=4096)
    self.assertEqual(nparams, _OLMO2_1B_NPARAMS)

  def test_layers_are_stackable(self):
    for flavor in ("debugmodel", "1B"):
      with torch.device("meta"):
        model = model_registry(flavor).model.build()
      check_layers_are_stackable(model.layers)

  def test_recipes_build(self):
    for recipe in (olmo2_debugmodel, olmo2_1b):
      config = recipe()
      self.assertEqual(config.model_spec.name, "olmo2")
      self.assertEqual(
          config.model_spec.max_context_length,
          config.training.max_context_length,
      )

  def test_qk_norm_rejected(self):
    attention = model_registry("debugmodel").model.layers[0].attention
    with self.assertRaises(ValueError):
      Olmo2Attention.Config(
          **{
              name: getattr(attention, name)
              for name in ("n_heads", "dim", "qkv_linear", "wo", "q_norm",
                           "k_norm", "n_kv_heads", "head_dim",
                           "inner_attention", "rope")
          },
          qk_norm=attention.q_norm,
      )

  def test_unsupported_parallelisms_rejected(self):
    parallelism = types.SimpleNamespace(spmd_backend="partial_dtensor")
    for parallelize in (parallelize_olmo2, graph_trainer_parallelize_olmo2):
      for enabled in ("tp_enabled", "cp_enabled", "pp_enabled"):
        dims = types.SimpleNamespace(
            tp_enabled=False, cp_enabled=False, pp_enabled=False
        )
        setattr(dims, enabled, True)
        with self.assertRaises(NotImplementedError, msg=enabled):
          parallelize(
              torch.nn.Module(),
              parallel_dims=dims,
              parallelism=parallelism,
              training=None,
              compile_config=None,
              dump_folder="",
          )

  def test_graph_trainer_registry_keeps_the_model(self):
    for flavor in ("debugmodel", "1B"):
      spec = model_registry(flavor)
      graph_spec = graph_trainer_model_registry(flavor)
      self.assertIs(type(graph_spec.model), GraphTrainerOlmo2Model.Config)
      self.assertIs(graph_spec.parallelize_fn, graph_trainer_parallelize_olmo2)
      self.assertIsNone(graph_spec.pipelining_fn)
      self.assertEqual(graph_spec.model.layers, spec.model.layers)
      self.assertEqual(graph_spec.model.lm_head, spec.model.lm_head)


if __name__ == "__main__":
  absltest.main()
