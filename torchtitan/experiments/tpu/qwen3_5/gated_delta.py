"""TPU-compatible GatedDeltaNet, attention and MoE modules for Qwen3.5.

Upstream ``torchtitan.models.qwen3_5.gdn`` imports the CUDA/Triton-only
``attn_gym`` package at top level and runs its non-varlen depthwise causal
``Conv1d`` across the folded ``[T] = [B * S]`` axis as a single stream. In
addition, upstream ``Qwen35Attention.forward`` does not pass ``positions`` to
its ``inner_attention``, so neither SDPA nor Splash / Tokamax attention sees
packed-document boundaries.

This module provides:
1. ``_install_attn_gym_stub()``: installs a stub ``attn_gym`` module in
   ``sys.modules`` when ``attn_gym`` is unavailable so
   ``torchtitan.models.qwen3_5`` imports cleanly on CPU and TPU.
2. ``chunked_gated_delta``: pure-PyTorch/XLA chunked gated delta rule matching
   ``attn_gym.linear.chunk_gdn`` / HF ``torch_chunk_gated_delta_rule``.
3. ``TpuGatedDeltaKernel`` and ``TpuInnerGatedDeltaNet``: drop-in ``Module``
   replacements that unflatten ``[T]`` to ``[B, S]`` via
   ``set_unflatten_seq_len`` for both the depthwise causal ``conv1d`` and the
   gated delta rule (``chunked_gated_delta`` or Tokamax's Mosaic kernels), and
   expand grouped Q/K heads to match ``n_value_heads``. Each ``[B, S]`` row is
   one recurrence: documents packed into a row share the convolution and the
   delta-rule state (no varlen ``cu_seqlens``).
4. ``PackedScaledDotProductInnerAttention`` and ``TpuQwen35Attention``: forward
   ``positions`` to the inner attention and apply same-document causal masking
   on packed sequences.
5. ``TpuQwen35TokenDispatcher``: static-shape expert-parallel (EP > 1)
   all-to-all token dispatch and combine that ``TpuGraphTrainer`` can trace.
6. ``TpuQwen35Model``: the Qwen3.5 model with the above, accepting
   ``positions``.
"""

from dataclasses import dataclass, field
import importlib.util
import math
import sys
import types

import spmd_types as spmd
import torch
from torch.autograd.function import once_differentiable
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.fx.traceback import annotate_fn
from torch.nn.attention import sdpa_kernel
import torch.nn.functional as F


def _install_attn_gym_stub() -> None:
  """Registers a stub ``attn_gym`` package if ``attn_gym`` is not installed."""
  if "attn_gym" in sys.modules:
    return
  if importlib.util.find_spec("attn_gym") is not None:
    return

  def _stub_op(name: str):
    def _unavailable(*args, **kwargs):
      del args, kwargs
      raise RuntimeError(
          f"attn_gym.linear.{name} is CUDA/Triton-only; use the TPU "
          "TpuInnerGatedDeltaNet / TpuGatedDeltaKernel modules instead."
      )

    return _unavailable

  def _getattr(name: str):
    if name.startswith("__"):
      raise AttributeError(name)
    return _stub_op(name)

  for mod_name in ("attn_gym", "attn_gym.linear"):
    module = types.ModuleType(mod_name)
    module.__path__ = []  # type: ignore[attr-defined]
    module.__getattr__ = _getattr  # type: ignore[method-assign]
    sys.modules[mod_name] = module


_install_attn_gym_stub()

# pylint: disable=g-import-not-at-top,g-bad-import-order
import torchtitan.config
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.experiments.graph_trainer.simple_fsdp import disable_active_parametrization
from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    scale_route_weights,
    stable_grouped_reduce,
)
from torchtitan.experiments.tpu.unflatten_shim import get_unflatten_seq_len
from torchtitan.models.common.attention import (
    AttentionMasksType,
    ScaledDotProductInnerAttention,
    segment_ids_from_positions,
)
from torchtitan.models.common.token_dispatcher import (
    AllToAllDispatchMetadata,
    LocalDispatchMetadata,
)
from torchtitan.models.qwen3_5.gdn import (
    GatedDeltaKernel,
    InnerGatedDeltaNet,
)
from torchtitan.models.qwen3_5.model import Qwen35Attention, Qwen35Model

# pylint: enable=g-import-not-at-top,g-bad-import-order


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _resolve_batch_and_seq_len(num_tokens: int) -> tuple[int, int]:
  """Resolves ``(batch_size, seq_len)`` from ``get_unflatten_seq_len()``."""
  seq_len = get_unflatten_seq_len()
  if seq_len is None or num_tokens <= seq_len:
    return 1, num_tokens
  if num_tokens % seq_len != 0:
    raise ValueError(
        f"num_tokens ({num_tokens}) must be divisible by seq_len ({seq_len})"
    )
  return num_tokens // seq_len, seq_len


def _causal_conv(x_BSC: torch.Tensor, weight_C1W: torch.Tensor) -> torch.Tensor:
  """Depthwise causal ``conv1d`` on ``[B, S, C]`` via shifted slices.

  ``y[s] = sum_k w[k] * x[s + k - (W - 1)]``, with zeros before the sequence.
  Slices of the front-padded input avoid transposing ``[B, S, C]`` to
  ``[B, C, S]`` and torch_tpu's grouped 1D ``convolution_backward`` shape bug
  under GraphTrainer.

  Args:
    x_BSC: ``[B, S, C]`` input.
    weight_C1W: ``[C, 1, W]`` depthwise ``Conv1d`` weight.

  Returns:
    ``[B, S, C]`` convolution output.
  """
  seq_len = x_BSC.shape[1]
  kernel_size = weight_C1W.shape[-1]
  x_pad = F.pad(x_BSC, (0, 0, kernel_size - 1, 0))
  out_BSC = x_pad[:, :seq_len, :] * weight_C1W[:, 0, 0]
  for k in range(1, kernel_size):
    out_BSC = out_BSC + x_pad[:, k : k + seq_len, :] * weight_C1W[:, 0, k]
  return out_BSC


class _CausalConvSilu(torch.autograd.Function):
  """``silu(_causal_conv(x, w))`` with a backward built from slices too.

  Autograd of ``_causal_conv`` accumulates the ``W`` shifted input gradients
  into the padded buffer with in-place adds at sublane-misaligned sequence
  offsets, which XLA emits as ``dynamic-update-slice`` adds at ~0.19 TB/s on
  TPU7x (12-layer 35B-A3B proxy at 32K tokens/rank: 5.4% of device time). The
  input gradient of a causal convolution is an anti-causal convolution of the
  output gradient, ``dx[t] = sum_k w[k] * dy[t + W - 1 - k]``, so it is computed
  from shifted slices of the end-padded output gradient instead, like the
  forward. The convolution output is recomputed for ``silu``'s derivative.
  """

  @staticmethod
  def forward(ctx, x_BSC: torch.Tensor, weight_C1W: torch.Tensor):
    ctx.save_for_backward(x_BSC, weight_C1W)
    return F.silu(_causal_conv(x_BSC, weight_C1W))

  @staticmethod
  def backward(ctx, grad_BSC: torch.Tensor):
    x_BSC, weight_C1W = ctx.saved_tensors
    seq_len = x_BSC.shape[1]
    kernel_size = weight_C1W.shape[-1]
    grad_y_BSC = torch.ops.aten.silu_backward(
        grad_BSC, _causal_conv(x_BSC, weight_C1W)
    )
    grad_pad = F.pad(grad_y_BSC, (0, 0, 0, kernel_size - 1))
    grad_x_BSC = None
    grad_w = []
    for k in range(kernel_size):
      shift = kernel_size - 1 - k
      term = grad_pad[:, shift : shift + seq_len, :] * weight_C1W[:, 0, k]
      grad_x_BSC = term if grad_x_BSC is None else grad_x_BSC + term
      if shift == 0:
        grad_w.append((x_BSC * grad_y_BSC).sum(dim=(0, 1)))
      elif shift < seq_len:
        grad_w.append(
            (x_BSC[:, : seq_len - shift, :] * grad_y_BSC[:, shift:, :]).sum(
                dim=(0, 1)
            )
        )
      else:
        grad_w.append(grad_y_BSC.new_zeros(weight_C1W.shape[0]))
    grad_w_C1W = torch.stack(grad_w, dim=-1).unsqueeze(1)
    return grad_x_BSC, grad_w_C1W.to(weight_C1W.dtype)


def causal_conv_silu(
    x_BSC: torch.Tensor, weight_C1W: torch.Tensor
) -> torch.Tensor:
  """``silu`` of a depthwise causal ``conv1d`` over ``[B, S, C]``."""
  return _CausalConvSilu.apply(x_BSC, weight_C1W)


def _matmul_bf16x3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Matmul of fp32 operands with near-fp32 accuracy from three bf16 passes.

  torch_tpu runs fp32 matmuls at ``Precision.DEFAULT`` (one bf16 MXU pass) by
  default. Splitting each operand into a bf16-exact high part and a residual
  keeps ``hi @ hi + hi @ lo + lo @ hi`` accurate to about 2^-16 regardless of
  the matmul precision setting.
  """
  a_hi = a.to(torch.bfloat16).float()
  b_hi = b.to(torch.bfloat16).float()
  return a_hi @ b_hi + (a_hi @ (b - b_hi) + (a - a_hi) @ b_hi)


def _unit_lower_triangular_inverse(strict_lower: torch.Tensor) -> torch.Tensor:
  """Returns ``(I + strict_lower)^-1`` for ``[..., C, C]`` with C a power of 2.

  Recursive doubling over diagonal blocks: a unit lower-triangular matrix split
  as ``[[A, 0], [L, D]]`` has inverse ``[[A^-1, 0], [-D^-1 L A^-1, D^-1]]``.
  With ``T_s`` the block-diagonal matrix of the inverses of the ``s x s``
  diagonal blocks and ``L_s`` the lower-left ``s x s`` quadrants of the
  ``2s x 2s`` diagonal blocks of ``strict_lower``, this reads
  ``T_2s = T_s - T_s L_s T_s``. Starting from ``T_2 = I - L_1``,
  ``log2(C) - 1`` levels of two batched (bf16x3) ``C x C`` matmuls build the
  inverse. Unlike a Neumann series, every intermediate is an entry of the
  inverse itself, so there is no growth and cancellation.

  Every intermediate stays ``[..., C, C]`` (masked, mostly-zero blocks are
  multiplied too), so no ``s x s`` block is a trailing dimension that TPU would
  pad to its ``(8, 128)`` tiles. On TPU7x this measured the same as a blockwise
  ``[..., C/s, s, s]`` version (49 vs 50 ms forward + backward for 4 x 2048
  tokens with 32 heads and 128-token chunks): both are bound by HBM traffic of
  the ``[C, C]`` FP32 intermediates, which is what the ``tokamax`` backend of
  ``TpuGatedDeltaKernel`` avoids.

  Args:
    strict_lower: ``[..., C, C]`` strictly lower-triangular matrices (the
      diagonal and upper triangle must be zero).

  Returns:
    ``[..., C, C]`` inverses of ``I + strict_lower``.
  """
  c = strict_lower.shape[-1]
  if c == 1:
    return torch.ones_like(strict_lower)
  idx = torch.arange(c, device=strict_lower.device)
  row, col = idx.unsqueeze(-1), idx
  inv = torch.eye(c, dtype=strict_lower.dtype, device=strict_lower.device)
  s = 1
  while s < c:
    # Lower-left s x s quadrant of each diagonal 2s x 2s block.
    quadrant = (
        (row // (2 * s) == col // (2 * s))
        & (row % (2 * s) >= s)
        & (col % (2 * s) < s)
    )
    lower_left = strict_lower * quadrant
    if s == 1:
      # T_1 = I, so T_2 = I - L_1.
      inv = inv - lower_left
    else:
      inv = inv - _matmul_bf16x3(inv, _matmul_bf16x3(lower_left, inv))
    s *= 2
  return inv


def chunked_gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
  """Chunked gated delta rule in pure PyTorch/XLA (FLA ``chunk_gdn`` math).

  Work within each chunk is expressed as batched matrix multiplications over
  all chunks at once (including the UT transform's unit lower-triangular
  inverse, see ``_unit_lower_triangular_inverse``); only the inter-chunk state
  recurrence is sequential over ``seq_len / chunk_size`` steps, and each step
  is three matmuls. Applies Q/K L2 normalization and ``1 / sqrt(key_head_dim)``
  query scaling in fp32.

  Args:
    q: ``(bs, seq_len, n_heads, key_head_dim)`` un-normalized queries.
    k: ``(bs, seq_len, n_heads, key_head_dim)`` un-normalized keys.
    v: ``(bs, seq_len, n_heads, value_head_dim)`` values.
    g: ``(bs, seq_len, n_heads)`` log-space decay (negative).
    beta: ``(bs, seq_len, n_heads)`` update gate in ``(0, 1)``.
    chunk_size: Tokens per chunk, a power of 2.

  Returns:
    ``(bs, seq_len, n_heads, value_head_dim)`` output in ``q.dtype``.
  """
  out_dtype = q.dtype
  d_k = q.shape[-1]
  q = _l2norm(q.float()) * (d_k**-0.5)
  k = _l2norm(k.float())
  q, k, v = [x.transpose(1, 2).float() for x in (q, k, v)]
  g, beta = [x.transpose(1, 2).float() for x in (g, beta)]

  bs, n_heads, seq_len, _ = k.shape
  d_v = v.shape[-1]
  chunk = chunk_size
  if 0 < seq_len < chunk_size:
    # Short sequences use one chunk, rounded up to a power of 2.
    chunk = 1 << (seq_len - 1).bit_length()
  if chunk & (chunk - 1):
    raise ValueError(f"chunk_size must be a power of 2: {chunk}")
  pad = (chunk - seq_len % chunk) % chunk
  if pad:
    # Padded tokens have beta = 0 (no state update) and g = 0 (no decay).
    q, k, v = [F.pad(x, (0, 0, 0, pad)) for x in (q, k, v)]
    g, beta = [F.pad(x, (0, pad)) for x in (g, beta)]
  num_chunks = (seq_len + pad) // chunk

  q, k, v = [
      x.reshape(bs, n_heads, num_chunks, chunk, x.shape[-1]) for x in (q, k, v)
  ]
  beta = beta.reshape(bs, n_heads, num_chunks, chunk, 1)
  g = g.reshape(bs, n_heads, num_chunks, chunk).cumsum(dim=-1)

  # decay[i, j] = exp(g_i - g_j) for i >= j and 0 otherwise. Masking before
  # exp keeps the positive upper-triangle exponents from overflowing.
  decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()

  # UT transform for all chunks at once:
  #   [W | U] = T [diag(beta) diag(exp(g)) K | diag(beta) V],
  #   T = (I + tril_{-1}(diag(beta) (K K^T * decay)))^-1.
  k_beta = k * beta
  a = (k_beta @ k.transpose(-1, -2) * decay).tril(-1)
  rhs = torch.cat([k_beta * g.exp().unsqueeze(-1), v * beta], dim=-1)
  wu = _unit_lower_triangular_inverse(a) @ rhs
  w, u = wu[..., :d_k], wu[..., d_k:]

  # State-independent per-chunk terms, batched over all chunks.
  qk = q @ k.transpose(-1, -2) * decay  # Causal, including the diagonal.
  g_last = g[..., -1:]
  # Rows [:chunk] give W S (the state's prediction of each value) and rows
  # [chunk:] give exp(g) Q S (the inter-chunk output), in one matmul.
  wq = torch.cat([w, q * g.exp().unsqueeze(-1)], dim=-2)
  k_decayed_t = (k * (g_last - g).exp().unsqueeze(-1)).transpose(-1, -2)
  chunk_decay = g_last.exp().unsqueeze(-1)

  state = torch.zeros(bs, n_heads, d_k, d_v, dtype=q.dtype, device=q.device)
  outs = []
  for i in range(num_chunks):
    wq_state = wq[:, :, i] @ state
    v_new = u[:, :, i] - wq_state[..., :chunk, :]
    outs.append(wq_state[..., chunk:, :] + qk[:, :, i] @ v_new)
    state = state * chunk_decay[:, :, i] + k_decayed_t[:, :, i] @ v_new

  out = torch.stack(outs, dim=2).reshape(bs, n_heads, -1, d_v)[:, :, :seq_len]
  return out.transpose(1, 2).contiguous().to(out_dtype)


class TpuGatedDeltaKernel(GatedDeltaKernel):
  """TPU-compatible ``GatedDeltaKernel`` using ``chunked_gated_delta``."""

  @dataclass(kw_only=True, slots=True)
  class Config(GatedDeltaKernel.Config):
    # "xla": `chunked_gated_delta` (pure PyTorch, any device). "tokamax":
    # Tokamax's Mosaic KDA kernels on TPU (`tokamax_gated_delta`), which keep
    # each chunk in VMEM; `chunk_size` then does not apply.
    backend: str = "xla"
    # "xla" only. On TPU7x (4 x 2048 tokens, 32 heads x 128 dims), compiled
    # forward + backward took 65 ms with 64-token chunks, 49 ms with 128, and
    # 82 ms with 256.
    chunk_size: int = 128
    # "tokamax" only: per-token log decays below this are clamped to it (see
    # `tokamax_gated_delta.DEFAULT_MIN_LOG_DECAY`).
    tokamax_min_log_decay: float = -5.0

  def __init__(self, config: Config):
    super().__init__(config)
    if config.backend not in ("xla", "tokamax"):
      raise ValueError(f"unknown TpuGatedDeltaKernel backend: {config.backend}")
    self.backend = config.backend
    self.chunk_size = config.chunk_size
    self.tokamax_min_log_decay = config.tokamax_min_log_decay

  def forward(
      self,
      xq_THK: torch.Tensor,
      xk_THK: torch.Tensor,
      xv_THV: torch.Tensor,
      g_TH: torch.Tensor,
      beta_TH: torch.Tensor,
      *,
      cu_seqlens: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if cu_seqlens is not None:
      raise NotImplementedError(
          "TpuGatedDeltaKernel does not support varlen cu_seqlens."
      )
    repeats = xv_THV.shape[1] // xq_THK.shape[1]
    if repeats > 1:
      xq_THK = xq_THK.repeat_interleave(repeats, dim=1)
      xk_THK = xk_THK.repeat_interleave(repeats, dim=1)
    num_tokens = xq_THK.shape[0]
    batch_size, seq_len = _resolve_batch_and_seq_len(num_tokens)
    inputs = (
        xq_THK.reshape(batch_size, seq_len, *xq_THK.shape[1:]),
        xk_THK.reshape(batch_size, seq_len, *xk_THK.shape[1:]),
        xv_THV.reshape(batch_size, seq_len, *xv_THV.shape[1:]),
        g_TH.reshape(batch_size, seq_len, *g_TH.shape[1:]),
        beta_TH.reshape(batch_size, seq_len, *beta_TH.shape[1:]),
    )
    if self.backend == "tokamax":
      # pylint: disable-next=g-import-not-at-top
      from torchtitan.experiments.tpu.qwen3_5 import tokamax_gated_delta

      out_BSHV = tokamax_gated_delta.tokamax_gated_delta(
          *inputs, min_log_decay=self.tokamax_min_log_decay
      )
    else:
      out_BSHV = chunked_gated_delta(*inputs, chunk_size=self.chunk_size)
    return out_BSHV.reshape(num_tokens, *out_BSHV.shape[2:])


class TpuInnerGatedDeltaNet(InnerGatedDeltaNet):
  """TPU-compatible ``InnerGatedDeltaNet`` isolating ``[B, S]`` conv streams."""

  @dataclass(kw_only=True, slots=True)
  class Config(InnerGatedDeltaNet.Config):
    kernel: GatedDeltaKernel.Config = field(
        default_factory=TpuGatedDeltaKernel.Config
    )

  def forward(
      self,
      query_TC: torch.Tensor,
      key_TC: torch.Tensor,
      value_TC: torch.Tensor,
      a_TH: torch.Tensor,
      b_TH: torch.Tensor,
      conv_q_weight_C1W: torch.Tensor,
      conv_k_weight_C1W: torch.Tensor,
      conv_v_weight_C1W: torch.Tensor,
      A_log_H: torch.Tensor,
      dt_bias_H: torch.Tensor,
      cu_seqlens: torch.Tensor,
      *,
      key_head_dim: int,
      value_head_dim: int,
  ) -> torch.Tensor:
    del cu_seqlens  # Fixed-shape [B, S] unflattening via set_unflatten_seq_len.
    num_tokens = query_TC.shape[0]
    batch_size, seq_len = _resolve_batch_and_seq_len(num_tokens)

    def causal_conv(
        x_TC: torch.Tensor,
        weight_C1W: torch.Tensor,
    ) -> torch.Tensor:
      channels = weight_C1W.shape[0]
      # Each [B, S] stream is convolved on its own (see `causal_conv_silu`).
      return causal_conv_silu(
          x_TC.reshape(batch_size, seq_len, channels), weight_C1W
      ).reshape(num_tokens, channels)

    xq_THK = causal_conv(query_TC, conv_q_weight_C1W).reshape(
        num_tokens, -1, key_head_dim
    )
    xk_THK = causal_conv(key_TC, conv_k_weight_C1W).reshape(
        num_tokens, -1, key_head_dim
    )
    xv_THV = causal_conv(value_TC, conv_v_weight_C1W).reshape(
        num_tokens, -1, value_head_dim
    )
    g_TH = -torch.exp(A_log_H.float()) * F.softplus(a_TH.float() + dt_bias_H)
    beta_TH = torch.sigmoid(b_TH)
    return self.kernel(
        xq_THK,
        xk_THK,
        xv_THV,
        g_TH,
        beta_TH,
        cu_seqlens=None,
    )


def packed_causal_mask(
    seq_len: int,
    device: torch.device,
    *,
    segment_ids_BL: torch.Tensor | None = None,
) -> torch.Tensor:
  """Causal mask restricted to each packed document within a row."""
  q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
  kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)
  causal_LL = q_idx >= kv_idx
  if segment_ids_BL is None:
    return causal_LL
  same_doc_BLL = segment_ids_BL.unsqueeze(2) == segment_ids_BL.unsqueeze(1)
  return (causal_LL.unsqueeze(0) & same_doc_BLL).unsqueeze(1)


class PackedScaledDotProductInnerAttention(ScaledDotProductInnerAttention):
  """Document-masked causal SDPA compatible with Splash / Tokamax patches."""

  @dataclass(kw_only=True, slots=True)
  class Config(ScaledDotProductInnerAttention.Config):
    pass

  def forward(
      self,
      q_BLHK: torch.Tensor,
      k_BLHK: torch.Tensor,
      v_BLHV: torch.Tensor,
      *,
      attention_masks: AttentionMasksType | None = None,
      scale: float | None = None,
      enable_gqa: bool = False,
      is_causal: bool = True,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
      **kwargs,
  ) -> torch.Tensor:
    if segment_ids is None and positions is not None:
      pos_1d = positions[:, 0] if positions.ndim == 2 else positions
      segment_ids = segment_ids_from_positions(pos_1d)
    if segment_ids is None:
      return super().forward(
          q_BLHK,
          k_BLHK,
          v_BLHV,
          attention_masks=attention_masks,
          scale=scale,
          enable_gqa=enable_gqa,
          is_causal=is_causal,
          **kwargs,
      )
    if attention_masks is not None:
      raise ValueError(
          "PackedScaledDotProductInnerAttention does not support runtime "
          "attention_masks."
      )
    if not is_causal:
      raise ValueError("Packed-document SDPA attention is always causal.")

    was_3d = q_BLHK.ndim == 3
    if was_3d:
      batch_size, seq_len = _resolve_batch_and_seq_len(q_BLHK.shape[0])
      q_BLHK = q_BLHK.reshape(batch_size, seq_len, *q_BLHK.shape[1:])
      k_BLHK = k_BLHK.reshape(batch_size, seq_len, *k_BLHK.shape[1:])
      v_BLHV = v_BLHV.reshape(batch_size, seq_len, *v_BLHV.shape[1:])
    else:
      batch_size, seq_len = q_BLHK.shape[:2]

    q_BHLK, k_BHLK, v_BHLV = (
        q_BLHK.transpose(1, 2),
        k_BLHK.transpose(1, 2),
        v_BLHV.transpose(1, 2),
    )
    mask = packed_causal_mask(
        seq_len,
        q_BHLK.device,
        segment_ids_BL=segment_ids.reshape(batch_size, seq_len),
    )
    with sdpa_kernel(self.sdpa_backends, set_priority=True):
      out_BHLV = F.scaled_dot_product_attention(
          q_BHLK,
          k_BHLK,
          v_BHLV,
          attn_mask=mask,
          scale=scale,
          enable_gqa=enable_gqa,
      )
    out_BLHV = out_BHLV.transpose(1, 2)
    if was_3d:
      return out_BLHV.reshape(-1, *out_BLHV.shape[2:])
    return out_BLHV


class TpuQwen35Attention(Qwen35Attention):
  """Qwen35Attention that forwards ``positions`` to ``inner_attention``."""

  @dataclass(kw_only=True, slots=True)
  class Config(Qwen35Attention.Config):
    pass

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks: AttentionMasksType | None,
      positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    num_tokens = x_TD.shape[0]
    if positions is None:
      batch_size, seq_len = _resolve_batch_and_seq_len(num_tokens)
      if batch_size > 1:
        positions = torch.arange(
            seq_len, dtype=torch.long, device=x_TD.device
        ).repeat(batch_size)

    xq_gate_THC = self.wq(x_TD).view(num_tokens, -1, self.head_dim * 2)
    xq_THK, gate_THV = xq_gate_THC.chunk(2, dim=-1)
    xk_THK = self.wk(x_TD).view(num_tokens, -1, self.head_dim)
    xv_THV = self.wv(x_TD).view(num_tokens, -1, self.head_dim)

    xq_THK = self.q_norm(xq_THK)
    xk_THK = self.k_norm(xk_THK)

    assert self.rotary_dim <= self.head_dim
    xq_THR, xq_THP = (
        xq_THK[..., : self.rotary_dim],
        xq_THK[..., self.rotary_dim :],
    )
    xk_THR, xk_THP = (
        xk_THK[..., : self.rotary_dim],
        xk_THK[..., self.rotary_dim :],
    )
    xq_THR, xk_THR = self.rope(xq_THR, xk_THR, positions)
    xq_THK = torch.cat([xq_THR, xq_THP], dim=-1)
    xk_THK = torch.cat([xk_THR, xk_THP], dim=-1)

    out_THV = self.inner_attention(
        xq_THK,
        xk_THK,
        xv_THV,
        attention_masks=attention_masks,
        scale=self.scaling,
        enable_gqa=self.enable_gqa,
        positions=positions,
    ).contiguous()

    out_THV = out_THV * torch.sigmoid(gate_THV)
    out_TD = out_THV.view(num_tokens, -1)
    return self.wo(out_TD)


class _MaskedRowGatherWithInverse(torch.autograd.Function):
  """Row gather ``x[fwd_idx] * fwd_mask`` whose backward is a gather too.

  ``bwd_idx`` must invert ``fwd_idx`` on the rows where the masks are set, so
  the input gradient ``(dy * fwd_mask)[bwd_idx] * bwd_mask`` needs no
  scatter-add.
  """

  @staticmethod
  def forward(
      ctx,
      x_MD: torch.Tensor,
      fwd_idx_K: torch.Tensor,
      fwd_mask_K1: torch.Tensor,
      bwd_idx_M: torch.Tensor,
      bwd_mask_M1: torch.Tensor,
  ) -> torch.Tensor:
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(fwd_mask_K1, bwd_idx_M, bwd_mask_M1)
    if x_MD.shape[0] == 0 or fwd_idx_K.shape[0] == 0:
      return x_MD.new_zeros((fwd_idx_K.shape[0], x_MD.shape[1]))
    gathered = torch.index_select(x_MD, 0, fwd_idx_K)
    return gathered * fwd_mask_K1.to(gathered.dtype)

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_out_KD: torch.Tensor | None):
    if grad_out_KD is None or not ctx.needs_input_grad[0]:
      return None, None, None, None, None
    fwd_mask_K1, bwd_idx_M, bwd_mask_M1 = ctx.saved_tensors
    if grad_out_KD.shape[0] == 0 or bwd_idx_M.shape[0] == 0:
      return (
          grad_out_KD.new_zeros((bwd_idx_M.shape[0], grad_out_KD.shape[1])),
          None,
          None,
          None,
          None,
      )
    masked_grad_KD = grad_out_KD * fwd_mask_K1.to(grad_out_KD.dtype)
    grad_x_MD = torch.index_select(masked_grad_KD, 0, bwd_idx_M)
    grad_x_MD = grad_x_MD * bwd_mask_M1.to(grad_x_MD.dtype)
    return grad_x_MD, None, None, None, None


def _wait(tensor: torch.Tensor) -> torch.Tensor:
  """Waits on a functional collective's result.

  GraphTrainer's graph splitter requires a collective's ``wait_tensor`` in the
  same executable as the collective.

  Args:
    tensor: A collective's result, possibly an ``AsyncCollectiveTensor``.

  Returns:
    The plain tensor.
  """
  if isinstance(tensor, AsyncCollectiveTensor):
    return tensor.wait()
  return tensor


@dataclass(frozen=True, kw_only=True)
class TpuQwen35DispatchMetadata(LocalDispatchMetadata):
  """Static-shape EP dispatch metadata for ``TpuQwen35TokenDispatcher``."""

  send_flat_idx: torch.Tensor
  send_valid_flat: torch.Tensor
  send_slot_N: torch.Tensor  # pylint: disable=invalid-name
  send_token_valid_N: torch.Tensor  # pylint: disable=invalid-name
  perm_idx_R: torch.Tensor  # pylint: disable=invalid-name
  expert_valid_flat: torch.Tensor
  inv_perm_idx_R: torch.Tensor  # pylint: disable=invalid-name
  recv_valid_flat: torch.Tensor


class TpuQwen35TokenDispatcher(TpuAllToAllTokenDispatcher):
  """Token dispatcher with static-shape EP > 1 dispatch and combine.

  Upstream ``AllToAllTokenDispatcher`` copies the per-rank token splits to the
  host (``.to("cpu").tolist()``) for a variable-size all-to-all, which
  ``TpuGraphTrainer``'s static-shape ``make_fx`` / ``StaticCompiler`` path
  cannot trace.

  With ``ep_mesh is None`` (EP = 1) this is ``TpuAllToAllTokenDispatcher``.
  With EP > 1, each rank packs its ``N = T * K`` expert-sorted routed tokens
  into a static ``(ep_size, cap, D)`` send buffer, ``cap`` tokens per
  destination rank, and exchanges token counts and buffers with uniform
  all-to-alls. The receiver permutes the received tokens into expert-major
  order in a static ``(ep_size * cap, D)`` buffer whose trailing zero rows
  are assigned to the last local expert. The permutations are computed with
  int32 compares and ``argsort`` (no host syncs, no ``searchsorted``), and
  their backward passes are gathers (``_MaskedRowGatherWithInverse``).

  ``capacity_factor=None`` is dropless (``cap = N``, an ``ep_size``-times larger
  buffer than the routed tokens). Otherwise ``cap = ceil(capacity_factor * N /
  ep_size)``: a rank's tokens beyond ``cap`` for one destination rank are
  dropped (zero expert output), unlike the dropless EP = 1 path.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TpuAllToAllTokenDispatcher.Config):
    capacity_factor: float | None = None

  def __init__(self, config: Config):
    super().__init__(config)
    self.capacity_factor = config.capacity_factor

  @annotate_fn({"EP": "dispatch"})
  def dispatch(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      topk_expert_ids_TK: torch.Tensor,
      num_local_tokens_per_expert_E: torch.Tensor,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      AllToAllDispatchMetadata | LocalDispatchMetadata,
  ]:
    if self.ep_mesh is None:
      return super().dispatch(
          x_TD,
          topk_scores_TK,
          topk_expert_ids_TK,
          num_local_tokens_per_expert_E,
      )

    ep_size = self.ep_mesh.size()
    e = self.num_experts // ep_size
    (
        routed_input_ND,
        token_indices_experts_sorted_N,
        topk_scores_experts_sorted_N,
    ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

    if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
      spmd.mutate_type(
          num_local_tokens_per_expert_E,
          src=spmd.P,
          dst={"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
      )

    with maybe_set_sparse_mesh():
      pg = (
          "ep"
          if get_spmd_backend() == "spmd_types"
          else self.ep_mesh.get_group()
      )
      if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
        num_local_tokens_per_expert_E = spmd.reinterpret_mesh(
            num_local_tokens_per_expert_E, spmd.current_mesh()
        )
        routed_input_ND = spmd.reinterpret_mesh(
            routed_input_ND, spmd.current_mesh()
        )

      num_routed = routed_input_ND.shape[0]
      if self.capacity_factor is None:
        cap = max(num_routed, 1)
      else:
        cap = max(
            1,
            min(
                max(num_routed, 1),
                math.ceil(self.capacity_factor * num_routed / ep_size),
            ),
        )
      r_static = ep_size * cap
      device = x_TD.device

      with torch.no_grad():
        local_counts_EP_e = num_local_tokens_per_expert_E.to(
            torch.int32
        ).reshape(ep_size, e)
        if self.capacity_factor is not None:
          local_cum_e = local_counts_EP_e.cumsum(dim=1).clamp(max=cap)
          local_counts_capped_EP_e = torch.cat(
              [local_cum_e[:, :1], local_cum_e[:, 1:] - local_cum_e[:, :-1]],
              dim=1,
          )
        else:
          local_counts_capped_EP_e = local_counts_EP_e

        send_counts_EP = local_counts_EP_e.sum(dim=1)
        send_counts_capped_EP = local_counts_capped_EP_e.sum(dim=1)
        send_ends_EP = send_counts_EP.cumsum(dim=0)
        send_starts_EP = send_ends_EP - send_counts_EP

        slot_cap = torch.arange(cap, device=device, dtype=torch.int32)
        send_valid_EP_cap = slot_cap.unsqueeze(
            0
        ) < send_counts_capped_EP.unsqueeze(1)
        send_src_idx_EP_cap = (
            send_starts_EP.unsqueeze(1) + slot_cap.unsqueeze(0)
        ).clamp(min=0, max=max(num_routed - 1, 0))
        send_flat_idx = send_src_idx_EP_cap.reshape(-1).to(torch.long)
        send_valid_flat = send_valid_EP_cap.reshape(-1, 1)

        idx_N = torch.arange(num_routed, device=device, dtype=torch.int32)
        after_rank_N_EP = idx_N.unsqueeze(1) >= send_ends_EP.unsqueeze(0)
        dst_rank_N = (
            after_rank_N_EP.to(torch.int32)
            .sum(dim=1)
            .clamp(max=max(ep_size - 1, 0))
        )
        send_start_N = (
            after_rank_N_EP.to(torch.int32) * send_counts_EP.unsqueeze(0)
        ).sum(dim=1)
        pos_in_rank_N = idx_N - send_start_N
        send_token_valid_N = (pos_in_rank_N < cap).unsqueeze(1)
        send_slot_N = (
            dst_rank_N * cap + pos_in_rank_N.clamp(min=0, max=max(cap - 1, 0))
        ).to(torch.long)

        recv_counts_EP_e = _wait(
            self._token_count_exchange(
                local_counts_capped_EP_e.reshape(-1),
                pg,
                ep_size,
            )
        )
        t_mat = recv_counts_EP_e.to(torch.int32).reshape(ep_size, e)

        rank_local_ends_EP_e = t_mat.cumsum(dim=1)
        recv_valid_EP_cap = slot_cap.unsqueeze(0) < rank_local_ends_EP_e[:, -1:]
        recv_valid_flat = recv_valid_EP_cap.reshape(-1, 1)

        slot_expert_R = (
            (slot_cap.view(1, cap, 1) >= rank_local_ends_EP_e.unsqueeze(1))
            .to(torch.int32)
            .sum(dim=-1)
            .reshape(-1)
        )
        perm_idx_R = torch.argsort(slot_expert_R, stable=True)
        inv_perm_idx_R = torch.argsort(perm_idx_R)

        local_expert_counts_e = t_mat.sum(dim=0)
        total_valid = local_expert_counts_e.sum()
        q_R = torch.arange(r_static, device=device, dtype=torch.int32)
        expert_valid_flat = (q_R < total_valid).unsqueeze(1)
        pad_rows = r_static - total_valid
        num_global_tokens_per_local_expert_e = torch.cat([
            local_expert_counts_e[:-1],
            local_expert_counts_e[-1:] + pad_rows,
        ])

      send_buf_flat = _MaskedRowGatherWithInverse.apply(
          routed_input_ND,
          send_flat_idx,
          send_valid_flat,
          send_slot_N,
          send_token_valid_N,
      )
      # Uniform splits: every rank sends `cap` rows to every rank.
      recv_buf_flat = _wait(
          self._dispatch_token_exchange(send_buf_flat, pg, None, None)
      )
      routed_input_RD = _MaskedRowGatherWithInverse.apply(
          recv_buf_flat,
          perm_idx_R,
          expert_valid_flat,
          inv_perm_idx_R,
          recv_valid_flat,
      )

    metadata = TpuQwen35DispatchMetadata(
        token_indices_experts_sorted_N=token_indices_experts_sorted_N,
        topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
        send_flat_idx=send_flat_idx,
        send_valid_flat=send_valid_flat,
        send_slot_N=send_slot_N,
        send_token_valid_N=send_token_valid_N,
        perm_idx_R=perm_idx_R,
        expert_valid_flat=expert_valid_flat,
        inv_perm_idx_R=inv_perm_idx_R,
        recv_valid_flat=recv_valid_flat,
    )
    return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

  @annotate_fn({"EP": "combine"})
  def combine(
      self,
      routed_output_RD: torch.Tensor,
      metadata: AllToAllDispatchMetadata | LocalDispatchMetadata,
      x_TD: torch.Tensor,
  ) -> torch.Tensor:
    if self.ep_mesh is None:
      return super().combine(routed_output_RD, metadata, x_TD)
    assert isinstance(metadata, TpuQwen35DispatchMetadata)

    with maybe_set_sparse_mesh():
      pg = (
          "ep"
          if get_spmd_backend() == "spmd_types"
          else self.ep_mesh.get_group()
      )
      recv_back_flat = _MaskedRowGatherWithInverse.apply(
          routed_output_RD,
          metadata.inv_perm_idx_R,
          metadata.recv_valid_flat,
          metadata.perm_idx_R,
          metadata.expert_valid_flat,
      )
      send_back_flat = _wait(
          self._combine_token_exchange(recv_back_flat, pg, None, None)
      )
      routed_output_ND = _MaskedRowGatherWithInverse.apply(
          send_back_flat,
          metadata.send_slot_N,
          metadata.send_token_valid_N,
          metadata.send_flat_idx,
          metadata.send_valid_flat,
      )

    if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
      routed_output_ND = spmd.reinterpret_mesh(
          routed_output_ND, spmd.current_mesh()
      )

    if self.fuse_route_score_backward:
      routed_output_ND = scale_route_weights(
          routed_output_ND,
          metadata.topk_scores_experts_sorted_N,
      )
    else:
      routed_output_ND = (
          routed_output_ND.float()
          * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
      ).to(routed_output_ND.dtype)
    return stable_grouped_reduce(
        routed_output_ND,
        metadata.token_indices_experts_sorted_N,
        self.top_k,
    )


# Per-destination-rank token capacity of `TpuQwen35TokenDispatcher` under EP.
_EP_CAPACITY_FACTOR = 1.25


class TpuQwen35Model(Qwen35Model):
  """Qwen35Model accepting ``positions`` positionally or by keyword."""

  @dataclass(kw_only=True, slots=True)
  class Config(Qwen35Model.Config):

    def update_from_config(self, *, config, **kwargs) -> None:
      # Zero-argument super() does not work in slots dataclasses on Python
      # 3.12 (the class is re-created); call the parent explicitly.
      Qwen35Model.Config.update_from_config(self, config=config, **kwargs)
      if config.parallelism.expert_parallel_degree > 1:
        # `ExpertParallel` shards the expert weights on dim 0, which the
        # paired (w1 + w3) Pallas kernel's fused weight layout does not allow.
        for _, experts, _, _ in self.traverse(TpuGroupedExperts.Config):
          experts.use_pallas_paired_grouped_mm = False
        for _, dispatcher, parent, field_name in list(
            self.traverse(TpuAllToAllTokenDispatcher.Config)
        ):
          if type(dispatcher) is TpuAllToAllTokenDispatcher.Config:  # pylint: disable=unidiomatic-typecheck
            ep_dispatcher = torchtitan.config.derive(
                dispatcher,
                TpuQwen35TokenDispatcher.Config,
                capacity_factor=_EP_CAPACITY_FACTOR,
            )
            if isinstance(parent, list):
              parent[field_name] = ep_dispatcher
            else:
              setattr(parent, field_name, ep_dispatcher)

  def init_states(
      self,
      *,
      buffer_device: torch.device | None = None,
  ) -> None:
    # Under SimpleFSDP, parameter reads return all-gathered copies;
    # initialize the sharded parameters themselves.
    with disable_active_parametrization():
      super().init_states(buffer_device=buffer_device)

  def forward(  # pyrefly: ignore[bad-override]
      self,
      tokens: torch.Tensor,
      positions: torch.Tensor | None = None,
      attention_masks: AttentionMasksType | None = None,
      **kwargs,
  ) -> torch.Tensor:
    return super().forward(
        tokens,
        attention_masks=attention_masks,  # pyrefly: ignore[bad-argument-type]
        positions=positions,
        **kwargs,
    )
