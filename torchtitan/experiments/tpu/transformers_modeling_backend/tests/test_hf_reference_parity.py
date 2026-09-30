"""Single-chip parity of the TPU lane's HF backend against the raw HF model.

For each model family the lane builds the wrapped model on one TPU chip with
its production choices -- splash attention (``attn_implementation=splash_tpu``)
and the fused Pallas cross entropy -- runs one forward/backward on a real
batch, and compares the loss and every parameter gradient against the
*original* ``transformers`` model class on CPU: same weights (copied from the
chip), same tokens, positions expressed as a dense document mask, plain
cross entropy over the same valid tokens. This is the check that the lane
computes what HF computes -- mask semantics, positions, head, loss -- with
nothing from torchtitan on the reference side but the batch.

    python -m torchtitan.experiments.tpu.transformers_modeling_backend.tests.test_hf_reference_parity

The multi-chip check (SimpleFSDP vs a CPU run of the same wrapper) is
``test_hf_backend_distributed``.
"""

import copy
import os
import socket
import tempfile

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

import torchtitan.config
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import gdist
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu import train_minimal
from torchtitan.experiments.tpu.transformers_modeling_backend import (
    attention as splash_adapter,
)
from torchtitan.experiments.tpu.transformers_modeling_backend.model import (
    TPUHFTransformerModel,
)
from torchtitan.tools import utils

_MODULE = "torchtitan.experiments.tpu.transformers_modeling_backend"


def _document_mask(positions: torch.Tensor, causal: bool) -> torch.Tensor:
  """Dense ``[1, 1, T, T]`` additive mask: same document, causal if asked.

  ``positions`` restart at 0 at every packed-sample boundary; this is the
  mask the lane hands splash as segment ids (plus its causal flag), written
  out for an attention implementation that takes a 4D mask as-is.
  """
  doc = torch.cumsum((positions == 0).int(), dim=0)
  allowed = doc[:, None] == doc[None, :]
  if causal:
    idx = torch.arange(positions.shape[0])
    allowed &= idx[:, None] >= idx[None, :]
  mask = torch.zeros(allowed.shape, dtype=torch.float32)
  mask.masked_fill_(~allowed, float("-inf"))
  return mask[None, None]


def _reference_logits(reference, wrapper, input_ids, kwargs):
  """The logits the raw HF model produces for the wrapper's inputs.

  Mirrors what the wrapper trains through: the model's own head when it maps
  to the vocabulary, otherwise the trunk's hidden states through the head the
  lane attached (tied to the token embedding).
  """
  if wrapper._added_lm_head is None:
    return reference(input_ids=input_ids, **kwargs).logits
  hidden = reference.base_model(input_ids=input_ids, **kwargs).last_hidden_state
  return hidden @ reference.get_input_embeddings().weight.t()


def _single_chip_env() -> dict[str, str]:
  """The TPU runtime environment for one chip of this host.

  ``cloud_distributed_utils`` derives these from LOCAL_RANK / WORLD_SIZE for
  a multi-chip torchrun and declines a world of one, so a single-chip worker
  states them itself: chip 0, a 1x1x1 topology, and a free port for the
  slice builder so a concurrent multi-chip job on the host is not disturbed.
  """
  with socket.socket() as sock:
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
  address = f"localhost:{port}"
  return {
      "CLOUD_TPU_TASK_ID": "0",
      "TPU_VISIBLE_CHIPS": "0",
      "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
      "TPU_HOST_BOUNDS": "1,1,1",
      "TPU_PROCESS_ADDRESSES": address,
      "TPU_PROCESS_PORT": str(port),
      "TORCH_TPU_SLICEBUILDER_ADDRESSES": address,
      "TORCH_TPU_TOPOLOGY": "1,1,1",
  }


def _run_on_device(config, out_path: str):
  """Worker: one step on the accelerator, the same step on the raw HF model."""
  # The TPU runtime comes up with the tpu_dist process group; TrainerMinimal
  # only opens one for world_size > 1, so open a world of one here.
  dist_utils.init_distributed(config.comm, base_folder=config.dump_folder)
  torch.manual_seed(config.debug.seed or 42)  # reproducible init; TrainerMinimal seeds only multi-process runs
  device = utils.get_local_device()
  trainer = train_minimal.TrainerMinimal(
      device=device, rank=0, world_size=1, job_config=config
  )
  wrapper = trainer.model
  hf = wrapper.model
  assert isinstance(wrapper, TPUHFTransformerModel), type(wrapper)

  # One real batch, prepared exactly as the trainer prepares it.
  batch = dict(next(iter(trainer.dataloader)))
  valid = batch.pop("num_valid_tokens", None)
  for key, value in batch.items():
    if isinstance(value, torch.Tensor):
      batch[key] = value.to(device)
  tokens, labels, extra, = wrapper.preprocess_inputs(
      batch, parallel_dims=trainer.parallel_dims, parallelism=config.parallelism
  )
  if valid is None:
    valid = (labels != IGNORE_INDEX).sum()
  valid = torch.as_tensor(valid, dtype=torch.int64, device=device)

  wrapper.train()
  trainer.optimizers.zero_grad()
  pred = wrapper(tokens, **extra)
  fused_operands = isinstance(pred, tuple)  # (hidden, head weight) for the fused CE
  loss, _ = trainer.loss_fn(pred, labels, valid)
  loss.backward()
  # The adapter reports once per process: "splash_tpu fired" when the kernel
  # ran, and a reason string for every distinct fallback to sdpa. Eager mode
  # (compile is off in this test) is what makes these reports fire.
  reports = set(splash_adapter._reported)
  splash_fired = "splash_tpu fired" in reports
  fallbacks = sorted(r for r in reports if r not in ("splash_tpu fired", "splash_tpu args"))
  device_loss = loss.detach().float().cpu()
  device_grads = {
      name: p.grad.detach().float().cpu()
      for name, p in hf.named_parameters()
      if p.grad is not None
  }
  attn_impl = getattr(hf.config, "_attn_implementation", None)
  fused = wrapper._fused_loss

  # --- the reference: the same HF class, on CPU, with the portable attention.
  ref_config = copy.deepcopy(hf.config)
  TPUHFTransformerModel._set_attn_everywhere(ref_config, "sdpa")
  reference = type(hf)(config=ref_config)
  state = {k: v.detach().float().cpu() for k, v in hf.state_dict().items()}
  missing, unexpected = reference.load_state_dict(state, strict=False)
  assert not unexpected, f"unexpected keys in the reference: {unexpected[:5]}"
  # Tied weights are absent from a state_dict by design; anything else
  # missing means the copy is not the same model.
  tied = set(reference._tied_weights_keys or []) if hasattr(reference, "_tied_weights_keys") else set()
  missing = [m for m in missing if m not in tied and not any(m.endswith(t) for t in tied)]
  assert not missing, f"missing keys in the reference: {missing[:5]}"
  reference.float().eval()  # dropout is zero on the device side; eval() is that with no RNG

  positions = extra.get("positions")
  positions = positions.cpu() if positions is not None else torch.arange(tokens.shape[0])
  causal = any(getattr(m, "is_causal", False) for m in hf.modules())
  ref_kwargs = {
      "position_ids": positions[None],
      "attention_mask": _document_mask(positions, causal),
  }
  if wrapper._forward_kwargs is not None:
    ref_kwargs = {k: v for k, v in ref_kwargs.items() if k in wrapper._forward_kwargs}
  input_ids = tokens.cpu()[None]
  logits = _reference_logits(reference, wrapper, input_ids, ref_kwargs)
  ref_loss = F.cross_entropy(
      logits.float().squeeze(0), labels.cpu(), ignore_index=IGNORE_INDEX, reduction="sum"
  ) / valid.cpu()
  ref_loss.backward()
  ref_grads = {
      name: p.grad.detach().float()
      for name, p in reference.named_parameters()
      if p.grad is not None
  }

  torch.save(
      {
          "device_loss": device_loss,
          "ref_loss": ref_loss.detach(),
          "device_grads": device_grads,
          "ref_grads": ref_grads,
          "attn_impl": attn_impl,
          "splash_fired": splash_fired,
          "fallbacks": fallbacks,
          "fused_loss": fused,
          "fused_operands": fused_operands,
          "model_class": type(hf).__name__,
          "n_valid": int(valid.cpu()),
      },
      out_path,
  )


class HFReferenceParityTest(parameterized.TestCase):
  """One chip, splash + fused CE, vs the raw transformers model on CPU."""

  def setUp(self):
    super().setUp()
    torch.compiler.reset()
    self.accelerator_device_type = test_utils.resolve_device_type()
    if self.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
      self.skipTest("This parity test targets the TPU kernels.")
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(tempfile.gettempdir(), "hf_datasets"))

  @parameterized.named_parameters([
      # Decoders: GQA + RoPE (Qwen3), the lane's frontier family (Llama).
      dict(testcase_name="qwen3", config="hf_debugmodel_max",
           hf_model="Qwen/Qwen3-0.6B", args=["--loss.b_block_size=256"]),
      dict(testcase_name="llama3", config="hf_debugmodel_max",
           hf_model="meta-llama/Llama-3.2-1B-Instruct", args=["--loss.b_block_size=256"]),
      # Encoder at full size: bidirectional splash, native MLM head, learned
      # positions; dropout zeroed so the reference is deterministic.
      dict(testcase_name="bert_base", config="hf_full_splash",
           hf_model="google-bert/bert-base-uncased",
           args=["--hf_config_overrides=attention_probs_dropout_prob=0,hidden_dropout_prob=0"]),
      # GPT-2 at full size: learned positions, tied head, `attn` block naming.
      dict(testcase_name="distilgpt2", config="hf_full_splash",
           hf_model="distilbert/distilgpt2",
           args=["--hf_config_overrides=attn_pdrop=0,resid_pdrop=0,embd_pdrop=0"]),
      # Encoder whose position ids start at padding_idx + 1, so the lane grows
      # the position table (RoBERTa); native MLM head.
      # DistilBERT is deliberately absent: its HF config aliases
      # hidden_size -> dim while the backend Config aliases dim -> hidden_size,
      # and the two recurse (a shared-backend issue, not a TPU one).
      dict(testcase_name="roberta_base", config="hf_full_splash",
           hf_model="FacebookAI/roberta-base",
           args=["--hf_config_overrides=attention_probs_dropout_prob=0,hidden_dropout_prob=0"]),
      # MoE checkpoint through splash. Note what this covers: the Titan MoE
      # swap happens in parallelize_fn, which TrainerMinimal runs only for
      # world_size > 1, so on one chip HF's own MixtralSparseMoeBlock runs on
      # the device too -- this is splash + HF MoE vs HF on CPU, not the Titan
      # MoE. Routing is top-k on a random-init model, so a bf16 attention
      # difference can flip a near-tie assignment and move a handful of
      # gradient entries by ~1e-2 (measured 4 of 2M embedding entries); the
      # loss itself agrees to ~2e-3.
      dict(testcase_name="tiny_mixtral_moe", config="hf_moe_auto",
           hf_model="hf-internal-testing/tiny-random-MixtralForCausalLM",
           grad_atol=2e-2, grad_rtol=1e-1),
  ])
  def test_one_step_matches_hf(
      self, config, hf_model, args=None, loss_atol=1e-2, loss_rtol=1e-3,
      grad_atol=5e-3, grad_rtol=5e-3,
  ):
    config_args = [
        f"--module={_MODULE}",
        f"--config={config}",
        f"--hf_model={hf_model}",
        "--optimizer.implementation=foreach",
        "--hf_assets_path=tests/assets/tokenizer",
        "--training.max_context_length=128",
        "--training.steps=1",
        "--training.mixed_precision_param=float32",
        "--training.mixed_precision_reduce=float32",
        "--training.num_tokens_per_microbatch_per_dp_rank=256",  # 2 x 128
        "--parallelism.data_parallel_shard_degree=1",
    ] + (args or [])
    job_config = torchtitan.config.ConfigManager().parse_args(config_args)
    job_config.dump_folder = tempfile.gettempdir()
    # Eager on one chip: the adapter's fallback report fires in eager, so a
    # kernel that did not run is visible in the log rather than silently
    # measured as sdpa.
    job_config.compile.enable = False

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
      out_path = tmp.name
    try:
      os.environ.update(_single_chip_env())
      gdist.torchrun(_run_on_device, nproc_per_node=1)(job_config, out_path)
      result = torch.load(out_path)
    finally:
      if os.path.exists(out_path):
        os.remove(out_path)

    self.assertEqual(result["attn_impl"], "splash_tpu", "the device run did not select splash")
    self.assertEqual(result["fallbacks"], [], "the splash adapter fell back to sdpa")
    self.assertTrue(result["splash_fired"], "the splash kernel never ran")
    self.assertEqual(
        result["fused_operands"], result["fused_loss"],
        "fused-loss config and the model's output disagree",
    )
    logging.info(
        "%s: device loss %.6f, reference loss %.6f (fused_loss=%s, %d valid tokens)",
        result["model_class"], result["device_loss"], result["ref_loss"],
        result["fused_loss"], result["n_valid"],
    )
    torch.testing.assert_close(
        result["device_loss"], result["ref_loss"], atol=loss_atol, rtol=loss_rtol,
        msg=lambda m: f"loss vs raw {result['model_class']}: {m}",
    )
    device_grads, ref_grads = result["device_grads"], result["ref_grads"]
    self.assertEqual(set(device_grads), set(ref_grads), "gradient sets differ")
    worst = max(
        (float((device_grads[n] - ref_grads[n]).abs().max()), n) for n in ref_grads
    )
    logging.info("largest gradient difference %.3e on %s", *worst)
    for name in sorted(ref_grads):
      torch.testing.assert_close(
          device_grads[name], ref_grads[name], atol=grad_atol, rtol=grad_rtol,
          msg=lambda m, name=name: f"grad {name}: {m}",
      )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
