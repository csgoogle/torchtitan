"""Equivalence test: JAX Conformer (Torchax) vs. PyTorch Conformer on TPU.

Methodology & Strategy:
  1. **Golden Reference Baseline**: We instantiate a native PyTorch Conformer
     model on CPU eager mode, running in full float32. This serves as our
     mathematically exact mathematical benchmark.
  2. **Weights Synchronization**: Because our Torchax conformer implements
     surgical in-place norm layer patching, it preserves 100% strict state_dict
     key parity. We instantiate a Torchax-compatible Conformer model and direct
     load the exact parameters from the PyTorch CPU baseline.
  4. **JAX Model Translation**: Inside the Torchax context block (`with env:`),
     we call `env.to_xla(model)` to compile the PyTorch module graph into an
     HLO (High-Level Optimizer) JAX trace, wrapped as a `JittableModule`.
  5. **JAX TPU Backpropagation**: We wrap our JAX CTC loss function inside
     `torchax.interop.jax_value_and_grad`. JAX traces and compiles the forward
     and
     backward step, executing it natively on the physical TPU device.
  6. **CPU Eager Backpropagation**: The golden reference PyTorch model executes
  the forward
     step and `loss.backward()` entirely on CPU in eager mode.
  7. **Numerical Assertions**: NumPy snapshots of forward logits, CTC loss
  values,
     and layer gradients are returned from both lanes and validated on CPU.
     Due to the physical TPU MXU's automatic downcasting of float32 matmuls to
     bfloat16 compute under the hood, a minor drift up to 5e-2 is expected and
     tolerated.
"""

from absl import logging
from absl.testing import absltest

# pylint: disable=g-import-not-at-top
try:
  import libtpu

  libtpu.configure_library_path()
except ImportError:
  pass

import contextlib
import jax
import torch
import torchax
import torchtitan.experiments.torchax.conformer as torchax_model
import torchtitan.experiments.torchax.conformer.loss as loss_module
import torchtitan.experiments.tpu.conformer.model as tpu_model

# pylint: enable=g-import-not-at-top

_BATCH_SIZE = 2
_SEQ_LEN = 16
_TARGET_LEN = 10

# Both lanes execute in float32 at the frontend level.
# However, the physical TPU MXU automatically downcasts float32 matmuls to bfloat16 compute
# under the hood, introducing a small numerical drift up to 4.3e-2 against CPU IEEE math.
_LOGITS_ATOL = 1e-4
_LOGITS_RTOL = 1e-4
_LOSS_DELTA = 1e-3
_GRADS_ATOL = 1e-3
_GRADS_RTOL = 1e-3


class ConformerEquivalenceTest(absltest.TestCase):
  """Parity check between TPU JAX/Torchax Conformer and CPU PyTorch Conformer."""

  def test_loss_and_gradients_equivalence(self):
    # Enable accuracy mode to force full Float32 precision on TPU.
    # Without this, the TPU MXU automatically downcasts Float32 matmuls to Bfloat16 compute under the hood,
    # which introduces significant numerical drift and forces us to use relaxed tolerances.
    torchax.enable_accuracy_mode()

    # Do NOT call torchax.enable_globally() to avoid backend conflicts with PyTorch/XLA.
    # Use the explicit torch_xla2 environment to translate modules.
    env = torchax.default_env()

    # Verify TPU device status under JAX tpu platform backend
    try:
      tpu_devices = jax.devices("tpu")
      is_tpu = len(tpu_devices) > 0
    except (ValueError, RuntimeError):
      tpu_devices = []
      is_tpu = False

    self.assertTrue(is_tpu, "TPU hardware device not initialized under JAX.")
    logging.info("TPU successfully initialized under JAX: %s", tpu_devices)

    tpu_args = torchax_model.args["debugmodel"]

    # Generate inputs on CPU
    torch.manual_seed(42)
    inputs = torch.randint(
        0,
        tpu_args.vocab_size,
        (_BATCH_SIZE, _SEQ_LEN),
        dtype=torch.long,
        device="cpu",
    )
    targets = torch.randint(
        1,
        tpu_args.vocab_size,
        (_BATCH_SIZE, _TARGET_LEN),
        dtype=torch.long,
        device="cpu",
    )

    # ==========================================================================
    # 1. Golden Reference: Native PyTorch on CPU (Eager)
    # ==========================================================================
    # Execute entirely on CPU in full float32
    model_tpu = tpu_model.Conformer(tpu_args).float().cpu()
    model_tpu.init_weights()
    model_tpu.train()

    # CPU Forward pass
    logits_tpu = model_tpu(inputs)

    # CPU CTC Loss computation
    input_lengths = torch.full(
        (_BATCH_SIZE,), _SEQ_LEN, dtype=torch.long, device="cpu"
    )
    target_lengths = torch.full(
        (_BATCH_SIZE,), _TARGET_LEN, dtype=torch.long, device="cpu"
    )
    logits_tpu_tnc = logits_tpu.permute(1, 0, 2).log_softmax(2)
    loss_tpu = torch.nn.functional.ctc_loss(
        logits_tpu_tnc,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="sum",
    )

    loss_tpu.backward()
    grads_tpu = {
        name: p.grad.clone().cpu()
        for name, p in model_tpu.named_parameters()
        if p.grad is not None
    }

    tpu_loss_numpy = float(loss_tpu.detach().cpu().numpy())
    logits_tpu_numpy = logits_tpu.detach().cpu().numpy()

    # ==========================================================================
    # 2. JAX Lane: Torchax JAX Compiled on TPU
    # ==========================================================================
    torchax_args = tpu_args
    model_torchax = torchax_model.model(torchax_args).float()
    model_torchax.train()

    # Sync weights from CPU state dict (extracted from PyTorch CPU Model)
    state_dict = model_tpu.state_dict()
    # Move state dict to CPU before loading into Torchax
    cpu_sd = {k: v.cpu() for k, v in state_dict.items()}
    model_torchax.load_state_dict(cpu_sd, strict=True)

    # Convert model and inputs to JAX via XLA inside the environment context
    with env:
      model_torchax_jax = env.to_xla(model_torchax)
      jittable_torchax = torchax.interop.JittableModule(model_torchax_jax)

      inputs_xla = env.to_xla(inputs)
      targets_xla = env.to_xla(targets)

      def loss_fn_torchax(params, buffers, x, y):
        logits = jittable_torchax.functional_call("forward", params, buffers, x)
        loss_val = loss_module.conformer_jax_ctc_loss(logits, y)
        return loss_val

      jitted_val_and_grad_torchax = torchax.interop.jax_value_and_grad(
          loss_fn_torchax
      )
      loss_torchax, grads_torchax_raw = jitted_val_and_grad_torchax(
          jittable_torchax.params,
          jittable_torchax.buffers,
          inputs_xla,
          targets_xla,
      )

      def forward_fn_torchax(params, buffers, x):
        return jittable_torchax.functional_call("forward", params, buffers, x)

      jitted_forward_torchax = torchax.interop.jax_jit(forward_fn_torchax)
      logits_torchax = jitted_forward_torchax(
          jittable_torchax.params, jittable_torchax.buffers, inputs_xla
      )

      logits_torchax_numpy = logits_torchax.detach().cpu().numpy()
      torchax_loss_numpy = float(loss_torchax.detach().cpu().numpy())

      # Extract grads back to CPU PyTorch tensors inside the active context
      grads_torchax = {
          k: v.detach().cpu() for k, v in grads_torchax_raw.items()
      }

    # ==========================================================================
    # 3. Compare Parity
    # ==========================================================================
    # A. Verify Logits
    logging.info(
        "Logits PyTorch shape: %s, Torchax JAX shape: %s",
        logits_tpu_numpy.shape,
        logits_torchax_numpy.shape,
    )
    torch.testing.assert_close(
        torch.from_numpy(logits_tpu_numpy),
        torch.from_numpy(logits_torchax_numpy),
        rtol=_LOGITS_RTOL,
        atol=_LOGITS_ATOL,
    )
    logging.info("Parity check successful: Logits match perfectly.")

    # B. Verify Loss
    logging.info("PyTorch CPU CTC Loss:      %.6f", tpu_loss_numpy)
    logging.info("Torchax JAX TPU CTC Loss:  %.6f", torchax_loss_numpy)
    self.assertAlmostEqual(
        tpu_loss_numpy, torchax_loss_numpy, delta=_LOSS_DELTA
    )
    logging.info("Parity check successful: Loss values match perfectly.")

    # C. Verify Gradients
    logging.info("Verifying backward pass gradients parity...")
    for name, tpu_grad in grads_tpu.items():
      self.assertIn(
          name,
          grads_torchax,
          f"JAX gradient missing for {name}",
      )

      tpu_grad_numpy = tpu_grad.detach().numpy()
      torchax_grad_numpy = grads_torchax[name].detach().cpu().numpy()

      torch.testing.assert_close(
          torch.from_numpy(tpu_grad_numpy),
          torch.from_numpy(torchax_grad_numpy),
          rtol=_GRADS_RTOL,
          atol=_GRADS_ATOL,
      )
    logging.info("Parity check successful: Gradients match perfectly.")


if __name__ == "__main__":
  absltest.main()
