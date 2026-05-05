"""Conformer model for TorchTitan."""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa
from torchtitan.experiments.tpu.tpu_job_config import TPUJobConfig
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
  use_splash: bool = False
  use_vmap_bwd: bool = False

  def update_from_config(self, job_config, **kwargs):
    # Update from job_config if needed
    if hasattr(job_config, "training"):
      self.max_seq_len = job_config.training.seq_len
    if isinstance(job_config, TPUJobConfig):
      self.use_splash = (
          job_config.splash_attention_kernel.use_splash_attention_kernel
      )
      self.use_vmap_bwd = (
          job_config.splash_attention_kernel.use_vmap_bwd
      )

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


# TPU Patch: PatchedMHA to force Flash Attention
class PatchedMHA(nn.Module):

  def __init__(self, embed_dim, num_heads, dropout=0.0, use_splash=False, use_vmap_bwd=False):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.dropout = dropout
    self.head_dim = embed_dim // num_heads
    self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
    self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
    self.use_splash = use_splash
    self.use_vmap_bwd = use_vmap_bwd

  def forward(
      self,
      query,
      key,
      value,
      key_padding_mask=None,
      need_weights=True,
      attn_mask=None,
      average_attn_weights=True,
      is_causal=False,
  ):
    bsz, tgt_len, _ = query.shape
    qkv = self.in_proj(query)
    q, k, v = qkv.chunk(3, dim=-1)

    if self.use_splash:
      q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
      k = k.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
      v = v.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
      attn_output = splash_sdpa(q, k, v, is_causal=is_causal, use_vmap_bwd=self.use_vmap_bwd)
    else:
      q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
      k = k.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
      v = v.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
      attn_output = torch.nn.functional.scaled_dot_product_attention(
          q,
          k,
          v,
          attn_mask=None,
          dropout_p=self.dropout if self.training else 0.0,
          is_causal=is_causal,
      )

    attn_output = attn_output.transpose(1, 2).contiguous().flatten(start_dim=-2)
    attn_output = self.out_proj(attn_output)
    return attn_output, None


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

  def __init__(
      self,
      dim,
      num_heads,
      ffn_dim,
      kernel_size=31,
      dropout_p=0.0,
      use_splash=False,
      use_vmap_bwd=False,
  ):
    super().__init__()
    self.ffn1 = _FeedForwardModule(dim, ffn_dim, dropout_p)
    self.self_attn_layer_norm = nn.LayerNorm(dim, eps=1e-05)
    # Use PatchedMHA instead of nn.MultiheadAttention
    self.self_attn = PatchedMHA(
        dim, num_heads, dropout=dropout_p, use_splash=use_splash, use_vmap_bwd=use_vmap_bwd
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

    self.conformer_layers = nn.ModuleList([
        ConformerLayer(
            dim=model_args.hidden_dim,
            num_heads=model_args.num_heads,
            ffn_dim=4 * model_args.hidden_dim,
            kernel_size=model_args.kernel_size,
            use_splash=model_args.use_splash,
            use_vmap_bwd=model_args.use_vmap_bwd,
        )
        for _ in range(model_args.num_layers)
    ])
    self.fc = nn.Linear(model_args.hidden_dim, model_args.vocab_size)
    logger.info("Conformer model built successfully.")

  def forward(self, inputs: torch.Tensor, **kwargs):
    x = inputs
    for layer in self.conformer_layers:
      x = layer(x)
    logits = self.fc(x)
    return logits

  def init_weights(self, buffer_device: torch.device | None = None) -> None:
    logger.info("Initializing Conformer weights mimicking PyTorch defaults ...")
    with torch.no_grad():
      for module in self.modules():
        if isinstance(module, nn.Linear):
          fan_in = module.in_features
          bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
          for param in module.parameters():
            local_tensor = (
                param.to_local() if hasattr(param, "to_local") else param
            )
            cpu_tensor = torch.empty_like(local_tensor, device="cpu")
            cpu_tensor.uniform_(-bound, bound)
            local_tensor.copy_(cpu_tensor)
        elif isinstance(module, nn.Conv1d):
          fan_in = module.in_channels * module.kernel_size[0]
          bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
          for param in module.parameters():
            local_tensor = (
                param.to_local() if hasattr(param, "to_local") else param
            )
            cpu_tensor = torch.empty_like(local_tensor, device="cpu")
            cpu_tensor.uniform_(-bound, bound)
            local_tensor.copy_(cpu_tensor)
        elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
          if module.weight is not None:
            local_tensor = (
                module.weight.to_local()
                if hasattr(module.weight, "to_local")
                else module.weight
            )
            local_tensor.fill_(1.0)
          if module.bias is not None:
            local_tensor = (
                module.bias.to_local()
                if hasattr(module.bias, "to_local")
                else module.bias
            )
            local_tensor.fill_(0.0)
      for buffer in self.buffers():
        buffer.fill_(0)
    logger.info("Conformer weights initialized.")
