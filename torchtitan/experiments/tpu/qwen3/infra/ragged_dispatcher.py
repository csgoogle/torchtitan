"""Bounded ragged token dispatcher for one-expert-per-rank TPU MoE."""

from dataclasses import dataclass, fields
import math

import torch

from torchtitan.experiments.tpu.qwen3.infra.ragged_all_to_all import (
    RouteMetadata,
    build_route_metadata,
    combine_route,
    dispatch_route,
)
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    BaseEPTokenDispatcher,
    LocalDispatchMetadata,
)
from torchtitan.ops.scatter_add import deterministic_scatter_add


@dataclass(frozen=True, kw_only=True)
class RaggedDispatchMetadata(LocalDispatchMetadata):
  """Local token order plus paired ragged payload metadata for combine."""

  route: RouteMetadata
  group_name: str


class BoundedRaggedTokenDispatcher(BaseEPTokenDispatcher):
  """Static-shape ragged all-to-all with deterministic overflow truncation."""

  @dataclass(kw_only=True, slots=True)
  class Config(BaseEPTokenDispatcher.Config):
    capacity_factor: float = 1.25

  def __init__(self, config: Config):
    super().__init__(config)
    if config.capacity_factor <= 0:
      raise ValueError("capacity_factor must be positive")
    self.capacity_factor = config.capacity_factor

  def _route_shape(self, token_count: int) -> tuple[int, int]:
    if self.ep_mesh is None:
      raise RuntimeError("ragged dispatcher requires expert parallelism")
    if self.sp_size != 1:
      raise NotImplementedError(
          "ragged dispatcher does not support sequence parallelism"
      )

    ep_size = self.ep_mesh.size()
    if self.num_experts != ep_size:
      raise ValueError(
          "ragged dispatcher requires one local expert "
          f"(num_experts={self.num_experts}, ep_size={ep_size})"
      )
    max_send = token_count * self.top_k
    capacity = math.ceil(
        self.capacity_factor * ep_size * max_send / self.num_experts
    )
    return max_send, capacity

  def dispatch(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      topk_expert_ids_TK: torch.Tensor,
      num_local_tokens_per_expert_E: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, RaggedDispatchMetadata]:
    max_send, capacity = self._route_shape(x_TD.shape[0])
    assert self.ep_mesh is not None

    (
        routed_input_ND,
        token_indices_experts_sorted_N,
        topk_scores_experts_sorted_N,
    ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)
    if routed_input_ND.shape[0] != max_send:
      raise ValueError(
          f"expected {max_send} local routes, got {routed_input_ND.shape[0]}"
      )

    send_sizes = num_local_tokens_per_expert_E.to(torch.int32).contiguous()
    route_experts = torch.repeat_interleave(
        torch.arange(self.num_experts, device=x_TD.device, dtype=torch.int64),
        send_sizes.to(torch.int64),
        output_size=max_send,
    )
    group = self.ep_mesh.get_group()
    route = build_route_metadata(send_sizes, route_experts, capacity, group)
    routed_input_RD = dispatch_route(routed_input_ND, route, group.group_name)
    num_global_tokens_per_local_expert_e = route.dispatch.recv_sizes.sum(
        dtype=torch.int32
    ).reshape(1)
    metadata = RaggedDispatchMetadata(
        token_indices_experts_sorted_N=token_indices_experts_sorted_N,
        topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
        route=route,
        group_name=group.group_name,
    )
    return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

  def combine(
      self,
      routed_output_RD: torch.Tensor,
      metadata: RaggedDispatchMetadata,
      x_TD: torch.Tensor,
      *,
      num_local_tokens_after_padding: int,
      local_seq_len_after_padding: int,
  ) -> torch.Tensor:
    del local_seq_len_after_padding
    if self.sp_size != 1:
      raise NotImplementedError(
          "ragged dispatcher does not support sequence parallelism"
      )

    routed_output_ND = combine_route(
        routed_output_RD,
        metadata.route,
        metadata.group_name,
    )
    routed_output_ND = (
        routed_output_ND.to(torch.float32)
        * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
    ).to(routed_output_ND.dtype)
    out_TD = torch.zeros(
        num_local_tokens_after_padding,
        x_TD.shape[-1],
        device=x_TD.device,
        dtype=x_TD.dtype,
    )
    token_indices_N = metadata.token_indices_experts_sorted_N
    return deterministic_scatter_add(
        out_TD,
        token_indices_N.reshape(-1, 1).expand(-1, out_TD.shape[-1]),
        routed_output_ND,
    )


def use_tpu_ragged_dispatcher(model_config, *, capacity_factor: float) -> None:
  """Replace standard all-to-all dispatcher configs in a model config tree."""
  for _, config, _, _ in model_config.traverse(
      BoundedRaggedTokenDispatcher.Config
  ):
    config.capacity_factor = capacity_factor

  matches = list(model_config.traverse(AllToAllTokenDispatcher.Config))
  for _, config, parent, field_name in matches:
    replacement = BoundedRaggedTokenDispatcher.Config(
        **{field.name: getattr(config, field.name) for field in fields(config)},
        capacity_factor=capacity_factor,
    )
    if isinstance(parent, list):
      assert isinstance(field_name, int)
      parent[field_name] = replacement
    else:
      assert parent is not None and isinstance(field_name, str)
      setattr(parent, field_name, replacement)


__all__ = [
    "BoundedRaggedTokenDispatcher",
    "RaggedDispatchMetadata",
    "use_tpu_ragged_dispatcher",
]
