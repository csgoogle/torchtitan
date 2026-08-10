from types import SimpleNamespace
from unittest import mock

import torch

from torchtitan.experiments.tpu.qwen3.infra.ragged_all_to_all import (
    _finalize_metadata,
)
from torchtitan.experiments.tpu.qwen3.infra.ragged_dispatcher import (
    BoundedRaggedTokenDispatcher,
    use_tpu_ragged_dispatcher,
)
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher


def _single_rank_route(rows: int, capacity: int):
  counts = torch.tensor([rows], dtype=torch.int32)
  offsets = torch.tensor([0], dtype=torch.int32)
  response = torch.tensor([[min(rows, capacity), 0]], dtype=torch.int32)
  experts = torch.zeros(rows, dtype=torch.int64)
  return _finalize_metadata(
      offsets,
      counts,
      response,
      offsets,
      torch.tensor([min(rows, capacity)], dtype=torch.int32),
      experts,
      capacity,
  )


def _dispatcher(capacity_factor=1.0):
  dispatcher = BoundedRaggedTokenDispatcher.Config(
      num_experts=1,
      top_k=2,
      capacity_factor=capacity_factor,
  ).build()
  group = SimpleNamespace(group_name="test")
  dispatcher.ep_mesh = SimpleNamespace(size=lambda: 1, get_group=lambda: group)
  return dispatcher


def test_dispatch_and_combine_match_local_routing_without_drops():
  dispatcher = _dispatcher()
  x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
  scores = torch.tensor([[0.75, 0.25], [0.6, 0.4], [0.9, 0.1]])
  experts = torch.zeros(3, 2, dtype=torch.int64)
  counts = torch.tensor([6], dtype=torch.int32)
  route = _single_rank_route(rows=6, capacity=6)

  with (
      mock.patch(
          "torchtitan.experiments.tpu.qwen3.infra.ragged_dispatcher.build_route_metadata",
          return_value=route,
      ),
      mock.patch(
          "torchtitan.experiments.tpu.qwen3.infra.ragged_dispatcher.dispatch_route",
          side_effect=lambda value, *args: value,
      ),
      mock.patch(
          "torchtitan.experiments.tpu.qwen3.infra.ragged_dispatcher.combine_route",
          side_effect=lambda value, *args: value,
      ),
  ):
    routed, local_counts, metadata = dispatcher.dispatch(x, scores, experts, counts)
    actual = dispatcher.combine(
        routed,
        metadata,
        x,
        num_local_tokens_after_padding=3,
        local_seq_len_after_padding=3,
    )

  torch.testing.assert_close(local_counts, torch.tensor([6], dtype=torch.int32))
  torch.testing.assert_close(actual, x)


def test_dispatch_rejects_multiple_local_experts_and_sequence_parallelism():
  dispatcher = _dispatcher()
  dispatcher.num_experts = 2
  try:
    dispatcher._route_shape(3)
  except ValueError as error:
    assert "one local expert" in str(error)
  else:
    raise AssertionError("expected one-local-expert validation")

  dispatcher.num_experts = 1
  dispatcher.sp_size = 2
  try:
    dispatcher._route_shape(3)
  except NotImplementedError as error:
    assert "sequence parallelism" in str(error)
  else:
    raise AssertionError("expected sequence-parallelism validation")


def test_config_replacement_preserves_dispatcher_fields():
  class RootConfig:
    def __init__(self):
      self.dispatcher = AllToAllTokenDispatcher.Config(num_experts=8, top_k=2)

    def traverse(self, config_type):
      if isinstance(self.dispatcher, config_type):
        yield "dispatcher", self.dispatcher, self, "dispatcher"

  root = RootConfig()
  use_tpu_ragged_dispatcher(root, capacity_factor=4.0)

  assert isinstance(root.dispatcher, BoundedRaggedTokenDispatcher.Config)
  assert root.dispatcher.num_experts == 8
  assert root.dispatcher.top_k == 2
  assert root.dispatcher.capacity_factor == 4.0
