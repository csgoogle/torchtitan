"""One-round, static-capacity ragged all-to-all routing.

The payload exchange needs four device-side int32 vectors per rank: input
offsets and send sizes for each destination, plus output offsets and receive
sizes for each source. A sender initially knows only the first pair. This
module derives the complete dispatch and inverse-combine metadata with two
small, packed metadata exchanges:

1. Each sender exchanges ``(requested size, input offset)`` with destinations.
2. Each destination accepts a deterministic sender-rank prefix up to its fixed
   capacity and returns ``(accepted size, output offset)`` to the senders.
3. The accepted segments define both the dispatch payload all-to-all and its
   exact inverse for combine and autograd.

The protocol has one segment per source/destination pair, fixed and identical
buffer bounds on every rank, one payload round, and deterministic overflow
truncation. It supports arbitrary payload trailing dimensions. It does not
represent multiple expert subsegments within a destination, sequence-parallel
token indexing, multi-round overflow, or dynamic output shapes; those are
dispatcher or protocol extensions rather than implicit behavior here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.distributed._functional_collectives import all_to_all_single


def _i32_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
  """Return an int32 cumulative sum without inheriting an int64 input dtype."""
  return torch.cumsum(x, dim=dim, dtype=torch.int32)


def _exchange_metadata(x: torch.Tensor, group) -> torch.Tensor:
  """Exchange one fixed-size metadata row with every rank and wait for it."""
  result = all_to_all_single(x, None, None, group)
  return torch.ops._c10d_functional.wait_tensor(result)


def _truncate_received(counts: torch.Tensor, capacity: int) -> torch.Tensor:
  """Accept requested rows in deterministic ascending sender-rank order."""
  counts = counts.to(torch.int32)
  prior = _i32_cumsum(counts, 0) - counts
  remaining = (capacity - prior).clamp(min=0)
  return torch.minimum(counts, remaining).to(torch.int32)


@dataclass(frozen=True)
class A2AMetadata:
  """Device metadata and static output shape for one ragged all-to-all leg."""

  input_offsets: torch.Tensor
  send_sizes: torch.Tensor
  output_offsets: torch.Tensor
  recv_sizes: torch.Tensor
  output_rows: int
  output_mask: torch.Tensor


@dataclass(frozen=True)
class RouteMetadata:
  """Paired dispatch and inverse-combine legs for one routing round."""

  dispatch: A2AMetadata
  combine: A2AMetadata
  dropped: torch.Tensor


def _build_request(send_sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Pack each destination's requested size and source input offset."""
  send_sizes = send_sizes.to(torch.int32)
  input_offsets = _i32_cumsum(send_sizes, 0) - send_sizes
  request = torch.stack((send_sizes, input_offsets), dim=1)
  return input_offsets, request.to(torch.int32).contiguous()


def _build_response(
    request: torch.Tensor,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Truncate received requests and pack accepted sizes and output offsets."""
  received_sizes = request[:, 0].to(torch.int32).contiguous()
  received_input_offsets = request[:, 1].to(torch.int32).contiguous()
  accepted_sizes = _truncate_received(received_sizes, capacity)
  output_offsets = _i32_cumsum(accepted_sizes, 0) - accepted_sizes
  response = torch.stack((accepted_sizes, output_offsets), dim=1)
  return (
      received_input_offsets,
      accepted_sizes,
      response.to(torch.int32).contiguous(),
  )


def _finalize_metadata(
    input_offsets: torch.Tensor,
    original_send_sizes: torch.Tensor,
    returned_response: torch.Tensor,
    received_input_offsets: torch.Tensor,
    accepted_recv_sizes: torch.Tensor,
    route_experts: torch.Tensor,
    capacity: int,
) -> RouteMetadata:
  """Assemble forward/inverse legs and masks from both metadata exchanges."""
  send_sizes = returned_response[:, 0].to(torch.int32).contiguous()
  output_offsets = returned_response[:, 1].to(torch.int32).contiguous()
  recv_offsets = _i32_cumsum(accepted_recv_sizes, 0) - accepted_recv_sizes

  # Normalize custom-call metadata to contiguous int32 vectors at the protocol
  # boundary. ``contiguous()`` only copies strided column views here; inverse
  # fields reuse the same read-only tensors rather than forcing eight clones.
  input_offsets = input_offsets.to(torch.int32).contiguous()
  accepted_recv_sizes = accepted_recv_sizes.to(torch.int32).contiguous()
  recv_offsets = recv_offsets.to(torch.int32).contiguous()
  received_input_offsets = received_input_offsets.to(torch.int32).contiguous()

  dispatch_rows = torch.arange(
      capacity, device=send_sizes.device, dtype=torch.int32
  )
  received = accepted_recv_sizes.sum(dtype=torch.int32)
  dispatch_mask = dispatch_rows < received

  route_rows = torch.arange(
      route_experts.shape[0], device=send_sizes.device, dtype=torch.int32
  )
  expert_index = route_experts.to(torch.int64)
  starts = torch.index_select(input_offsets, 0, expert_index)
  accepted = torch.index_select(send_sizes, 0, expert_index)
  combine_mask = (route_rows >= starts) & (route_rows < starts + accepted)
  dropped = original_send_sizes.sum(dtype=torch.int32)
  dropped = dropped - send_sizes.sum(dtype=torch.int32)

  dispatch = A2AMetadata(
      input_offsets=input_offsets,
      send_sizes=send_sizes,
      output_offsets=output_offsets,
      recv_sizes=accepted_recv_sizes,
      output_rows=capacity,
      output_mask=dispatch_mask.contiguous(),
  )
  combine = A2AMetadata(
      input_offsets=recv_offsets,
      send_sizes=accepted_recv_sizes,
      output_offsets=received_input_offsets,
      recv_sizes=send_sizes,
      output_rows=route_experts.shape[0],
      output_mask=combine_mask.contiguous(),
  )
  return RouteMetadata(dispatch=dispatch, combine=combine, dropped=dropped)


def build_route_metadata(
    send_sizes: torch.Tensor,
    route_experts: torch.Tensor,
    capacity: int,
    group,
    *,
    exchange: Callable[[torch.Tensor, Any], torch.Tensor] | None = None,
) -> RouteMetadata:
  """Build one bounded dispatch leg and its inverse with two metadata A2As.

  ``send_sizes`` has one entry per destination. ``route_experts`` contains the
  destination index for every row in the already destination-sorted payload.
  Tests may inject ``exchange`` to simulate all ranks without distributed
  initialization; production uses the fixed-size collective helper above.
  """
  if send_sizes.dtype != torch.int32:
    raise TypeError(f"send_sizes must be int32, got {send_sizes.dtype}")
  if send_sizes.dim() != 1 or route_experts.dim() != 1:
    raise ValueError("send_sizes and route_experts must be one-dimensional")
  if send_sizes.numel() == 0:
    raise ValueError("send_sizes must contain one entry per destination")

  exchange_metadata = exchange or _exchange_metadata
  with torch.no_grad():
    input_offsets, request = _build_request(send_sizes)
    received_request = exchange_metadata(request, group)
    received_input_offsets, accepted_recv_sizes, response = _build_response(
        received_request, capacity
    )
    returned_response = exchange_metadata(response, group)
    return _finalize_metadata(
        input_offsets,
        send_sizes,
        returned_response,
        received_input_offsets,
        accepted_recv_sizes,
        route_experts,
        capacity,
    )


def _ragged_a2a(
    operand: torch.Tensor,
    metadata: A2AMetadata,
    group_name: str,
) -> torch.Tensor:
  """Execute one masked, static-output ragged payload all-to-all."""
  vectors = (
      metadata.input_offsets,
      metadata.send_sizes,
      metadata.output_offsets,
      metadata.recv_sizes,
  )
  if any(x.dtype != torch.int32 for x in vectors):
    raise TypeError(f"ragged metadata must be int32: {[x.dtype for x in vectors]}")
  if any(x.shape != vectors[0].shape for x in vectors[1:]):
    raise ValueError(f"ragged metadata shapes must match: {[x.shape for x in vectors]}")
  if any(not x.is_contiguous() for x in vectors):
    raise ValueError("ragged metadata must be contiguous")
  if metadata.output_mask.shape != (metadata.output_rows,):
    raise ValueError(
        "output mask must have shape "
        f"{(metadata.output_rows,)}, got {metadata.output_mask.shape}"
    )

  output = operand.new_zeros(
      (metadata.output_rows,) + tuple(operand.shape[1:])
  )
  result = torch.ops.tpu.ragged_all_to_all(
      operand.contiguous(),
      output,
      metadata.input_offsets,
      metadata.send_sizes,
      metadata.output_offsets,
      metadata.recv_sizes,
      group_name,
  )
  keep = metadata.output_mask.reshape(
      (metadata.output_rows,) + (1,) * (result.dim() - 1)
  )
  return torch.where(keep, result, torch.zeros_like(result))


class _RaggedRoute(torch.autograd.Function):
  """Autograd bridge whose backward is the inverse ragged payload leg."""

  @staticmethod
  def forward(
      ctx,
      x,
      group_name,
      f_input_offsets,
      f_send_sizes,
      f_output_offsets,
      f_recv_sizes,
      f_output_rows,
      f_output_mask,
      b_input_offsets,
      b_send_sizes,
      b_output_offsets,
      b_recv_sizes,
      b_output_rows,
      b_output_mask,
  ):
    """Run the requested leg and retain only inverse metadata for backward."""
    forward_metadata = A2AMetadata(
        f_input_offsets,
        f_send_sizes,
        f_output_offsets,
        f_recv_sizes,
        f_output_rows,
        f_output_mask,
    )
    result = _ragged_a2a(x, forward_metadata, group_name)
    ctx.group_name = group_name
    ctx.output_rows = b_output_rows
    ctx.save_for_backward(
        b_input_offsets,
        b_send_sizes,
        b_output_offsets,
        b_recv_sizes,
        b_output_mask,
    )
    return result

  @staticmethod
  def backward(ctx, grad_output):
    """Route gradients through the inverse leg saved by ``forward``."""
    input_offsets, send_sizes, output_offsets, recv_sizes, output_mask = (
        ctx.saved_tensors
    )
    inverse_metadata = A2AMetadata(
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        ctx.output_rows,
        output_mask,
    )
    grad_input = _ragged_a2a(
        grad_output, inverse_metadata, ctx.group_name
    )
    return (grad_input,) + (None,) * 13


def _apply_route(
    x: torch.Tensor,
    forward: A2AMetadata,
    backward: A2AMetadata,
    group_name: str,
) -> torch.Tensor:
  """Flatten a metadata pair into the explicit custom-autograd call."""
  return _RaggedRoute.apply(
      x,
      group_name,
      forward.input_offsets,
      forward.send_sizes,
      forward.output_offsets,
      forward.recv_sizes,
      forward.output_rows,
      forward.output_mask,
      backward.input_offsets,
      backward.send_sizes,
      backward.output_offsets,
      backward.recv_sizes,
      backward.output_rows,
      backward.output_mask,
  )


def dispatch_route(
    x: torch.Tensor,
    metadata: RouteMetadata,
    group_name: str,
) -> torch.Tensor:
  """Route sender-packed rows into the fixed-capacity receive buffer."""
  return _apply_route(x, metadata.dispatch, metadata.combine, group_name)


def combine_route(
    x: torch.Tensor,
    metadata: RouteMetadata,
    group_name: str,
) -> torch.Tensor:
  """Return expert outputs to their original sender-packed row positions."""
  return _apply_route(x, metadata.combine, metadata.dispatch, group_name)


__all__ = [
    "A2AMetadata",
    "RouteMetadata",
    "build_route_metadata",
    "combine_route",
    "dispatch_route",
]
