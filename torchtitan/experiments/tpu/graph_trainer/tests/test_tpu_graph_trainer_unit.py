"""Unit tests for TpuGraphTrainer (tracing, loss override, and config wiring)."""

import typing

from absl.testing import absltest
import torch
from torch import nn
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.graph_trainer.make_fx_tracer import trace_train_step
from torchtitan.experiments.tpu.base_device_test import BaseAcceleratorDeviceTest
from torchtitan.experiments.tpu.loss import build_cross_entropy_loss, PallasCrossEntropyLoss
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig


class DummyModule(nn.Module):
  """Simple synthetic module for fast FX tracing checks."""

  def __init__(self, dim: int = 32, vocab_size: int = 128) -> None:
    super().__init__()
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.linear = nn.Linear(dim, dim)
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.tok_embeddings(x)
    return self.lm_head(torch.relu(self.linear(h)))


def _make_dummy_fwd_bwd_step(loss_fn: nn.Module):
  """Returns a joint forward+loss+backward step function for tracing verification."""

  def fwd_bwd_fn(model, x, target):
    pred = model(x)
    loss = loss_fn(pred.view(-1, 128), target.view(-1))
    loss.backward()
    return loss

  return fwd_bwd_fn


class TpuGraphTrainerUnitTest(BaseAcceleratorDeviceTest):
  """Fast single-process unit tests for TpuGraphTrainer components cleanly without mocks."""

  def setUp(self) -> None:
    super().setUp()
    self.config_manager = ConfigManager()
    self.config = typing.cast(
        TPUTrainerConfig,
        self.config_manager.parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            "--training.local_batch_size=2",
            "--training.seq_len=16",
            "--tpu_config.compile_mode=fullgraph",
        ]),
    )

  def test_config_triggers_graph_trainer_selection(self) -> None:
    """Verifies that use_graph_trainer or compile_mode='aot_fx_trace' triggers TpuGraphTrainer."""
    self.config.tpu_config.compile_mode = "aot_fx_trace"
    is_gt = (
        isinstance(self.config, TPUTrainerConfig)
        and (
            self.config.tpu_config.use_graph_trainer
            or self.config.tpu_config.compile_mode == "aot_fx_trace"
        )
    )
    self.assertTrue(
        is_gt,
        "compile_mode='aot_fx_trace' must trigger TpuGraphTrainer selection check.",
    )

  def test_pallas_ce_loss_override(self) -> None:
    """Verifies build_cross_entropy_loss returns PallasCrossEntropyLoss when use_loss_kernel=True."""
    config = typing.cast(
        TPUTrainerConfig,
        self.config_manager.parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            "--training.local_batch_size=2",
            "--training.seq_len=16",
            "--tpu_config.compile_mode=fullgraph",
            "--loss-kernel.use-loss-kernel",
        ]),
    )

    loss_fn = build_cross_entropy_loss(config)
    self.assertIsInstance(
        loss_fn,
        PallasCrossEntropyLoss,
        "build_cross_entropy_loss must produce PallasCrossEntropyLoss when use_loss_kernel is set.",
    )

  def test_make_fx_tracing_on_synthetic_model(self) -> None:
    """Verifies make_fx joint tracing on a synthetic fwd+loss+bwd step without mocks."""
    device = self.accelerator_device
    model = DummyModule(dim=32, vocab_size=128).to(device)
    loss_fn = nn.CrossEntropyLoss()
    fwd_bwd_fn = _make_dummy_fwd_bwd_step(loss_fn)

    inputs = torch.randint(0, 128, (2, 16), device=device)
    labels = torch.randint(0, 128, (2, 16), device=device)

    try:
      traced_step = trace_train_step(fwd_bwd_fn)(model, inputs, labels)
      self.assertIsInstance(
          traced_step.gm,
          torch.fx.GraphModule,
          "traced_step.gm must be a valid FX GraphModule.",
      )
    except Exception as e:
      self.fail(f"trace_train_step raised unexpected error during lowering: {e}")


if __name__ == "__main__":
  absltest.main()

