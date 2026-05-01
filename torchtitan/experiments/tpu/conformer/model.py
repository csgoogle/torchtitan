"""Conformer model for TorchTitan."""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torchtitan.protocols.model import BaseModelArgs, ModelProtocol
from torchtitan.tools.logging import logger


@dataclass
class ConformerModelArgs(BaseModelArgs):
  """Arguments for Conformer model configuration."""

  vocab_size: int = 64
  hidden_dim: int = 512
  num_layers: int = 17
  num_heads: int = 8
  kernel_size: int = 31
  max_seq_len: int = 2048  # Added to match train_minimal expectations

  def update_from_config(self, job_config, **kwargs):
    # Update from job_config if needed
    if hasattr(job_config, "training"):
      self.max_seq_len = job_config.training.seq_len

  def get_nparams_and_flops(
      self, model: nn.Module, seq_len: int
  ) -> tuple[int, int]:
    nparams = sum(p.numel() for p in model.parameters())
    # Dummy flops for now to avoid crash in metrics processor
    # redacted uses 6 * N * seq_len as rough estimate
    flops_per_token = 6 * nparams
    return nparams, flops_per_token


# --- TPU-compatible GLU Fix ---
def tpu_glu(input, dim=-1):
  chunks = input.chunk(2, dim)
  return chunks[0] * torch.sigmoid(chunks[1])


class TPUGLU(torch.nn.Module):

  def __init__(self, dim=-1):
    super().__init__()
    self.dim = dim

  def forward(self, x):
    return tpu_glu(x, self.dim)


# ------------------------------


class _FeedForwardModule(nn.Module):

  def __init__(self, dim, hidden_dim, dropout_p=0.0):
    super().__init__()
    self.sequential = nn.Sequential(
        nn.LayerNorm(dim, eps=1e-05),
        nn.Linear(dim, hidden_dim, bias=True),
        nn.SiLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(hidden_dim, dim, bias=True),
        nn.Dropout(p=dropout_p),
    )

  def forward(self, x):
    return self.sequential(x)


class _ConvolutionModule(nn.Module):

  def __init__(self, dim, kernel_size=31, dropout_p=0.0):
    super().__init__()
    padding = kernel_size // 2
    self.layer_norm = nn.LayerNorm(dim, eps=1e-05)
    self.sequential = nn.Sequential(
        nn.Conv1d(dim, 2 * dim, kernel_size=1, stride=1),
        TPUGLU(dim=1),
        nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=dim,
        ),
        nn.BatchNorm1d(dim, eps=1e-05, momentum=0.1),
        nn.SiLU(),
        nn.Conv1d(dim, dim, kernel_size=1, stride=1),
        nn.Dropout(p=dropout_p),
    )

  def forward(self, x):
    x = self.layer_norm(x)
    x = x.transpose(1, 2)
    x = self.sequential(x)
    x = x.transpose(1, 2)
    return x


class ConformerLayer(nn.Module):

  def __init__(self, dim, num_heads, ffn_dim, kernel_size=31, dropout_p=0.0):
    super().__init__()
    self.ffn1 = _FeedForwardModule(dim, ffn_dim, dropout_p)
    self.self_attn_layer_norm = nn.LayerNorm(dim, eps=1e-05)
    self.self_attn = nn.MultiheadAttention(
        dim, num_heads, dropout=dropout_p, batch_first=True
    )
    self.self_attn_dropout = nn.Dropout(p=dropout_p)
    self.conv_module = _ConvolutionModule(dim, kernel_size, dropout_p)
    self.ffn2 = _FeedForwardModule(dim, ffn_dim, dropout_p)
    self.final_layer_norm = nn.LayerNorm(dim, eps=1e-05)

  def forward(self, x):
    x = x + 0.5 * self.ffn1(x)
    residual = x
    x = self.self_attn_layer_norm(x)
    x, _ = self.self_attn(x, x, x)
    x = residual + self.self_attn_dropout(x)
    x = x + self.conv_module(x)
    x = x + 0.5 * self.ffn2(x)
    x = self.final_layer_norm(x)
    return x


class Conformer(ModelProtocol):
  """Build a conformer model matching torchaudio.models.Conformer architecture."""

  def __init__(self, model_args: ConformerModelArgs) -> None:
    super().__init__(model_args)
    self._model_args = model_args

    logger.info("Building Conformer model from scratch...")
    self.conformer_layers = nn.ModuleList([
        ConformerLayer(
            dim=model_args.hidden_dim,
            num_heads=model_args.num_heads,
            ffn_dim=4 * model_args.hidden_dim,
            kernel_size=model_args.kernel_size,
        )
        for _ in range(model_args.num_layers)
    ])
    self.fc = nn.Linear(model_args.hidden_dim, model_args.vocab_size)
    logger.info("Model built successfully.")

  def forward(self, inputs: torch.Tensor, **kwargs):
    x = inputs
    for layer in self.conformer_layers:
      x = layer(x)
    logits = self.fc(x)
    return logits

  def init_weights(self, buffer_device: torch.device | None = None) -> None:
    logger.info("Initializing Conformer weights...")
    with torch.no_grad():
      for module in [self.conformer_layers, self.fc]:
        for name, param in module.named_parameters():
          local_tensor = (
              param.to_local() if hasattr(param, "to_local") else param
          )
          local_tensor.uniform_(-0.01, 0.01)
        for buffer in module.buffers():
          buffer.fill_(0)
    logger.info("Conformer weights initialized.")
