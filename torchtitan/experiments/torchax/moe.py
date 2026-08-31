# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.distributed.tensor import DTensor
import torch.nn.functional as F
import torchax
from torchtitan.experiments.torchax import moe_utils
from torchtitan.protocols.train_spec import MoEArgs
import torchtitan.tools.logging


logger = torchtitan.tools.logging.logger


# can be used as dense FFN layer or shared experts in MoE layers
class FeedForward(nn.Module):
    """
    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


# NOTE: keeping this for-loop implementation for comparison
#       and readability, may remove later
def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
  """
  Runs experts sequentially using masking to avoid dynamic shapes (JIT-friendly).
  Replaces the original torch.split() logic which caused graph breaks on XLA.

  Args:
    w1: Expert weights for the first linear layer (gate).
    w2: Expert weights for the second linear layer (output).
    w3: Expert weights for the third linear layer (value).
    x: Input tensor containing tokens, already sorted by expert index.
    num_tokens_per_expert: Tensor indicating the number of tokens assigned to each expert.

  Returns:
    The output tensor after processing through the MoE experts.
  """
  # Note: x is already sorted by expert index.
  # num_tokens_per_expert contains the size of each expert's segment.

  # Convert counts to cumulative offsets to determine boundaries
  # [c0, c1, c2] -> [0, c0, c0+c1] (start indices)
  # We cast to long for indexing
  counts = num_tokens_per_expert.long()

  # Note: torch.cumsum on XLA is efficient.
  # We construct boundaries: starts at [0, ...], ends at [..., N]
  cum_counts = torch.cumsum(counts, dim=0)

  # Prepend 0 to get start indices
  zero = torch.zeros((1,), device=counts.device, dtype=counts.dtype)
  offsets = torch.cat((zero, cum_counts))

  # Global token index range [0, 1, ..., N-1]
  # This shape is static (batch_size * top_k)
  N = x.shape[0]
  token_indices = torch.arange(N, device=x.device)

  # Initialize output buffer
  out = torch.zeros_like(x)

  # Iterate over each expert
  for i in range(len(counts)):
    # Determine the range [start, end) for expert i
    start = offsets[i]
    end = offsets[i+1]

    # Create a boolean mask for tokens belonging to expert i
    # This operation is JIT-compatible as it uses element-wise comparison on static tensors
    mask = (token_indices >= start) & (token_indices < end)
    mask = mask.unsqueeze(-1).to(x.dtype)  # Expand to (N, 1) for broadcasting

    # Get weights for expert i
    w1_i = w1[i]  # (hidden, dim)
    w2_i = w2[i]  # (dim, hidden)
    w3_i = w3[i]  # (hidden, dim)

    # Run the expert on the FULL masked input.
    # Mathematically: We process (x * mask).
    # This keeps shapes static (N, dim).

    # Layer 1: Gate (w1) and Value (w3)
    h1 = F.silu(torch.matmul(x, w1_i.transpose(-2, -1)))
    h3 = torch.matmul(x, w3_i.transpose(-2, -1))
    h = h1 * h3

    # Layer 2: Output (w2)
    expert_out = torch.matmul(h, w2_i.transpose(-2, -1))

    # Accumulate result: out += expert_out * mask
    # The mask ensures only valid tokens for this expert affect the output.
    out = out + expert_out * mask

  return out


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    # Handle the JAX Global View output which is [Num_Chips, Num_Experts]
    # We want cumulative sum over experts (dim=1) to get offsets.
    if num_tokens_per_expert.dim() == 2:
        offsets = torch.cumsum(num_tokens_per_expert, dim=1, dtype=torch.int32)
    else:
        # Fallback to original behavior for non-JAX Global View output.
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)


    h = F.silu(
        torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
    )
    h = h * torch._grouped_mm(
        x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out


class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.use_grouped_mm = use_grouped_mm

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        if self.use_grouped_mm:
            return _run_experts_grouped_mm(w1, w2, w3, x, num_tokens_per_expert)
        else:
            return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class TokenChoiceTopKRouter(nn.Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        score_func (Literal["softmax", "sigmoid"]): Whether to use sigmoid or softmax for router scores.
        route_norm (bool): Whether to normalize the routing scores when using sigmoid.
        route_scale (float): Scaling factor applied to the routing scores.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"],
        route_norm: bool,
        route_scale: float,
        _debug_force_load_balance: bool = False,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self._debug_force_load_balance = _debug_force_load_balance

    def _debug_force_load_balance_routing(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns (selected_experts_indices [N, K] LongTensor, top_scores [N, K] FloatTensor).
        """
        n_tokens = scores.size(0)
        # Round-robin indices with exact balance
        selected_experts_indices = (
            torch.arange(
                n_tokens * self.top_k, device=scores.device, dtype=torch.int64
            ).reshape(n_tokens, self.top_k)
            % self.num_experts
        )
        top_scores = scores.gather(dim=1, index=selected_experts_indices)  # [N,K]
        return selected_experts_indices, top_scores

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        if expert_bias is not None:
            _, selected_experts_indices = torch.topk(
                scores + expert_bias, k=self.top_k, dim=1
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=1
            )

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_scores, selected_experts_indices, num_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


# NOTE: the reason we make this a stateless module is to support
#       expert_tensor_parallel_degree=1 with consistent TP/EP APIs.
class TokenReorderer(nn.Module):
    """
    This module reorders token indices to match the order of experts, enabling
    efficient parallel processing of tokens by experts.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of experts each token will be routed to.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reorders token indices to match the order of experts for MoE routing.

        Args:
            top_scores (torch.Tensor): Routing scores for selected experts,
                shape (batch_size * seq_len, top_k)
            selected_experts_indices (torch.Tensor): Expert indices selected for each token,
                shape (batch_size*seq_len, top_k)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores_experts_sorted: Scores reordered to match expert ordering
                - token_indices_experts_sorted: Token indices reordered to match expert ordering
                - num_tokens_per_expert: Number of tokens assigned to each expert
        """
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )


class MoE(nn.Module):

  def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
    super().__init__()

    num_experts = moe_args.num_experts
    self.experts = GroupedExperts(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        use_grouped_mm=moe_args.use_grouped_mm,
    )
    self.router = TokenChoiceTopKRouter(
        dim=dim,
        num_experts=num_experts,
        top_k=moe_args.top_k,
        score_func=moe_args.score_func,
        route_norm=moe_args.route_norm,
        route_scale=moe_args.route_scale,
        _debug_force_load_balance=moe_args._debug_force_load_balance,
    )
    self.shared_experts = (
        FeedForward(
            dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts
        )
        if moe_args.num_shared_experts > 0
        else None
    )
    self.score_before_experts = moe_args.score_before_experts
    self.top_k = moe_args.top_k
    self.num_experts = moe_args.num_experts

    # Internal JAX state
    self.mesh = None
    self._jax_count = None
    self._jax_permute = None
    self._jax_unpermute = None

    # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
    # NOTE: tokens_per_expert is accumulated in the model forward pass.
    #       expert_bias is updated outside the model in an optimizer step pre hook
    #       to work with gradient accumulation.
    self.load_balance_coeff = moe_args.load_balance_coeff
    if self.load_balance_coeff is not None:
      assert self.load_balance_coeff > 0.0
      self.register_buffer(
          "expert_bias",
          torch.zeros(num_experts, dtype=torch.float32),
          persistent=True,
      )
    else:
      self.expert_bias = None
    # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
    self.register_buffer(
        "tokens_per_expert",
        torch.zeros(num_experts, dtype=torch.float32),
        persistent=False,
    )

  def _init_jax_runners(self):
    """Attempts to retrieve the global mesh and initialize JAX runners."""
    if self.mesh is None:
      try:
        env = torchax.default_env()
        if hasattr(env, "_mesh"):
          self.mesh = env._mesh
          self._jax_count = moe_utils.make_count_runner(
              self.mesh, self.num_experts
          )
          self._jax_permute = moe_utils.make_permute_runner(
              self.mesh, self.top_k
          )
          self._jax_unpermute = moe_utils.make_unpermute_runner(
              self.mesh, self.top_k
          )
          logger.info(
              "[MoE] Successfully initialized JAX local sharding runners."
          )
      except Exception as e:
        pass

  def _count_tokens(self, selected_experts: torch.Tensor) -> torch.Tensor:
    """Compute histogram of tokens per expert.

    If mesh is present, uses JAX local counting to avoid Global Reduce.
    """
    if self.mesh is not None and isinstance(
        selected_experts, torchax.tensor.Tensor
    ):
      j_experts = torchax.interop.jax_view(selected_experts)
      j_counts = self._jax_count(j_experts)  # pyrefly: ignore[not-callable]
      return torchax.interop.torch_view(j_counts)

    # Fallback to one-hot implementation
    logger.warning(
        "[MoE] Falling back to one-hot implementation for counting tokens."
    )
    expert_ids_flat = selected_experts.view(-1)
    expert_one_hot = F.one_hot(
        expert_ids_flat, num_classes=self.num_experts
    ).to(torch.float32)
    return expert_one_hot.sum(dim=0)

  def _permute(
      self,
      x_flat: torch.Tensor,
      selected_experts: torch.Tensor,
      top_scores: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort input tokens by expert ID."""

    if self.mesh is not None and isinstance(x_flat, torchax.tensor.Tensor):
      j_x, j_experts, j_scores = torchax.interop.jax_view(
          (x_flat, selected_experts, top_scores)
      )

      j_routed, j_indices, j_scores_sorted = self._jax_permute(  # pyrefly: ignore[not-callable]
          j_x, j_experts, j_scores
      )

      return torchax.interop.torch_view(
          (j_routed, j_indices, j_scores_sorted)
      )

    # Fallback to PyTorch implementation
    expert_ids_flat = selected_experts.view(-1)
    sort_indices = torch.argsort(expert_ids_flat, stable=True)

    if self.top_k > 1:
      x_expanded = (
          x_flat.unsqueeze(1)
          .expand(-1, self.top_k, -1)
          .reshape(-1, x_flat.size(1))
      )
      routed_input = x_expanded[sort_indices]
      top_scores_sorted = top_scores.view(-1)[sort_indices]
    else:
      routed_input = x_flat[sort_indices]
      top_scores_sorted = top_scores.view(-1)[sort_indices]

    return routed_input, sort_indices, top_scores_sorted

  def _unpermute(
      self,
      routed_output: torch.Tensor,
      sort_indices: torch.Tensor,
      output_shape: torch.Size,
  ) -> torch.Tensor:
    """Scatter outputs back to original token positions."""

    if self.mesh is not None and isinstance(
        routed_output, torchax.tensor.Tensor
    ):
      j_routed, j_indices = torchax.interop.jax_view(
          (routed_output, sort_indices)
      )
      j_out = self._jax_unpermute(j_routed, j_indices, j_routed)  # pyrefly: ignore[not-callable]
      return torchax.interop.torch_view(j_out)

    # Fallback to PyTorch implementation
    out_flat = torch.zeros(
        output_shape, device=routed_output.device, dtype=routed_output.dtype
    )
    original_indices = sort_indices // self.top_k
    out_flat.index_add_(0, original_indices, routed_output)
    return out_flat

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.mesh is None:
      self._init_jax_runners()

    bs, slen, dim = x.shape
    x_flat = x.view(-1, dim)

    # 1. Routing
    top_scores, selected_experts, _ = self.router(x_flat, self.expert_bias)

    # 2. Count tokens (Local if mesh active)
    num_tokens_per_expert = self._count_tokens(selected_experts)

    # 3. Permutation (Local Sort if mesh active)
    routed_input, sort_indices, top_scores_sorted = self._permute(
        x_flat, selected_experts, top_scores
    )

    # 4. Score Before Experts
    if self.score_before_experts:
      routed_input = routed_input * top_scores_sorted.unsqueeze(-1).to(
          routed_input.dtype
      )

    # 5. Expert Computation (GMM)
    routed_output = self.experts(routed_input, num_tokens_per_expert)

    # Shared expert
    if self.shared_experts is not None:
      out = self.shared_experts(x)
    else:
      out = torch.zeros_like(x)

    # 6. Score After Experts
    if not self.score_before_experts:
      routed_output = routed_output * top_scores_sorted.unsqueeze(-1).to(
          routed_output.dtype
      )

    # 7. Un-Permute / Combine
    out_flat = self._unpermute(routed_output, sort_indices, x_flat.shape)

    out = out.view(bs, slen, dim) + out_flat.view(bs, slen, dim)
    return out

  def init_weights(
      self,
      init_std: float,
      buffer_device: torch.device,
  ):
    self.experts.init_weights(init_std)
    self.router.init_weights(init_std)
    if self.shared_experts is not None:
      self.shared_experts.init_weights(init_std)

    with torch.device(buffer_device):
      self.tokens_per_expert = torch.zeros(
          self.experts.num_experts, dtype=torch.float32
      )
      if self.load_balance_coeff is not None:
        self.expert_bias = torch.zeros(
            self.experts.num_experts, dtype=torch.float32
        )
