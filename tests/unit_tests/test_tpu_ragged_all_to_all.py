import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from torchtitan.experiments.tpu.qwen3.infra.ragged_all_to_all import (
    A2AMetadata,
    _build_request,
    _build_response,
    _finalize_metadata,
    _ragged_a2a,
    _truncate_received,
    build_route_metadata,
    combine_route,
    dispatch_route,
)


def _exclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
  return torch.cumsum(x, dim=dim, dtype=torch.int32) - x


def _simulate_metadata(matrix: torch.Tensor, capacity: int):
  world = matrix.shape[0]
  requests = []
  input_offsets = []
  route_experts = []
  for source in range(world):
    offsets, request = _build_request(matrix[source])
    input_offsets.append(offsets)
    requests.append(request)
    route_experts.append(
        torch.repeat_interleave(
            torch.arange(world, dtype=torch.int64), matrix[source].to(torch.int64)
        )
    )
  requests = torch.stack(requests)

  received_input_offsets = []
  accepted_recv_sizes = []
  responses = []
  for destination in range(world):
    received = requests[:, destination, :]
    original_offsets, accepted, response = _build_response(received, capacity)
    received_input_offsets.append(original_offsets)
    accepted_recv_sizes.append(accepted)
    responses.append(response)
  responses = torch.stack(responses)

  return [
      _finalize_metadata(
          input_offsets[rank],
          matrix[rank],
          responses[:, rank, :],
          received_input_offsets[rank],
          accepted_recv_sizes[rank],
          route_experts[rank],
          capacity,
      )
      for rank in range(world)
  ]


def test_builder_uses_two_packed_int32_exchanges():
  send_sizes = torch.tensor([2, 0, 3], dtype=torch.int32)
  route_experts = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int64)
  replies = iter(
      (
          torch.tensor([[1, 0], [4, 2], [0, 5]], dtype=torch.int32),
          torch.tensor([[2, 0], [0, 1], [1, 4]], dtype=torch.int32),
      )
  )
  observed = []

  def exchange(value, group):
    assert group == "group"
    observed.append(value.clone())
    return next(replies)

  route = build_route_metadata(
      send_sizes, route_experts, capacity=5, group="group", exchange=exchange
  )

  assert len(observed) == 2
  torch.testing.assert_close(
      observed[0], torch.tensor([[2, 0], [0, 2], [3, 2]], dtype=torch.int32)
  )
  torch.testing.assert_close(
      route.dispatch.send_sizes,
      torch.tensor([2, 0, 1], dtype=torch.int32),
  )


def test_packed_protocol_matches_sender_prefix_oracle():
  matrix = torch.tensor([[4, 1, 0], [3, 4, 2], [2, 3, 5]], dtype=torch.int32)
  capacity = 6
  metadata = _simulate_metadata(matrix, capacity)
  accepted = torch.tensor([[4, 1, 0], [2, 4, 2], [0, 1, 4]], dtype=torch.int32)
  source_offsets = _exclusive_cumsum(matrix, 1)
  destination_offsets = _exclusive_cumsum(accepted, 0)

  for rank, route in enumerate(metadata):
    torch.testing.assert_close(route.dispatch.input_offsets, source_offsets[rank])
    torch.testing.assert_close(route.dispatch.send_sizes, accepted[rank])
    torch.testing.assert_close(
        route.dispatch.output_offsets, destination_offsets[rank]
    )
    torch.testing.assert_close(route.dispatch.recv_sizes, accepted[:, rank])
    torch.testing.assert_close(
        route.combine.input_offsets, destination_offsets[:, rank]
    )
    torch.testing.assert_close(route.combine.send_sizes, accepted[:, rank])
    torch.testing.assert_close(route.combine.output_offsets, source_offsets[:, rank])
    torch.testing.assert_close(route.combine.recv_sizes, accepted[rank])
    assert route.dropped.item() == (matrix[rank] - accepted[rank]).sum().item()
    assert route.dropped.dtype == torch.int32
    for leg in (route.dispatch, route.combine):
      for tensor in (
          leg.input_offsets,
          leg.send_sizes,
          leg.output_offsets,
          leg.recv_sizes,
      ):
        assert tensor.dtype == torch.int32

  torch.testing.assert_close(
      metadata[1].combine.output_mask,
      torch.tensor([True, True, False, True, True, True, True, True, True]),
  )
  for rank, route in enumerate(metadata):
    torch.testing.assert_close(
        route.dispatch.output_mask,
        torch.arange(capacity) < int(accepted[:, rank].sum()),
    )


def test_truncation_is_deterministic_in_sender_order():
  counts = torch.tensor([4, 3, 2], dtype=torch.int32)
  torch.testing.assert_close(
      _truncate_received(counts, 6),
      torch.tensor([4, 2, 0], dtype=torch.int32),
  )


def test_ragged_op_rejects_non_int32_metadata():
  operand = torch.zeros(4, 2)
  vector = torch.zeros(1, dtype=torch.int32)
  metadata = A2AMetadata(
      input_offsets=vector.to(torch.int64),
      send_sizes=vector,
      output_offsets=vector,
      recv_sizes=vector,
      output_rows=4,
      output_mask=torch.ones(4, dtype=torch.bool),
  )
  with unittest.TestCase().assertRaisesRegex(TypeError, "int32"):
    _ragged_a2a(operand, metadata, "test")


def test_forward_and_vjp_drop_the_overflow_suffix():
  route = _simulate_metadata(torch.tensor([[5]], dtype=torch.int32), capacity=3)[0]
  x = torch.arange(10, dtype=torch.float32).reshape(5, 2).requires_grad_()

  def fake_ragged(
      operand,
      metadata,
      group_name,
  ):
    del group_name
    kept = int(metadata.send_sizes[0])
    result = F.pad(operand[:kept], (0, 0, 0, metadata.output_rows - kept))
    return torch.where(
        metadata.output_mask[:, None], result, torch.zeros_like(result)
    )

  with mock.patch(
      "torchtitan.experiments.tpu.qwen3.infra.ragged_all_to_all._ragged_a2a",
      side_effect=fake_ragged,
  ):
    received = dispatch_route(x, route, "test")
    returned = combine_route(received, route, "test")
    returned.sum().backward()

  torch.testing.assert_close(returned[:3], x.detach()[:3])
  torch.testing.assert_close(returned[3:], torch.zeros_like(returned[3:]))
  torch.testing.assert_close(x.grad[:3], torch.ones_like(x.grad[:3]))
  torch.testing.assert_close(x.grad[3:], torch.zeros_like(x.grad[3:]))
