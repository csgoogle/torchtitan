"""Topology-aware DeviceMesh construction for TorchTPU.

Cluster spawners like torchrun assign global rank ids by host list and spawn
order, with no topology input — so the rank -> chip mapping is arbitrary
(ranks 1 and 2 need not be physical neighbors). PyTorch expresses process groups
and DeviceMesh purely in these ranks, so the standard `init_device_mesh`
(sequential rank layout) places logically adjacent mesh coordinates on
physically distant devices. Collectives then run over multi-hop, contended paths
instead of neighbor ICI links, which degrades performance most on the innermost
(highest-traffic) mesh dimension. JAX does not have this problem because its
meshes are built from device objects that carry their physical coordinates.
PyTorch arranges labels that don't, so the mapping must be reconstructed.

This module serves two purposes:
1. Topology recovery: each rank's physical device coordinates are collected via
   an all_gather, making the resulting mesh correct and independent of the
   spawner's rank assignment.
2. Assignment quality: on top of the recovered physical layout, JAX's device
   mesh utility computes a good logical-to-physical assignment using
   per-generation placement handlers (e.g. v2/v3 tray rings, v5e rotated pincer,
   v5p 2x2x2, TPU7x rings, ...) and the N-D torus axis-assignment logic in
   `jax.experimental.mesh_utils`.

How it works
`mesh_utils` never needs real `jax.Device` objects — it only accesses the
following attributes: .device_kind, .coords, .core_on_chip, .platform, .id
(and .process_index / .slice_index for create_hybrid_device_mesh)

Each torch rank reads those fields from its local JAX device and all-gathers
them as a single int64 tensor. Every rank then reconstructs the same global list
of these device stubs, hands it to `mesh_utils`, and translates the arranged
device array back to torch ranks via an id -> rank map. Because all ranks derive
the layout from identical gathered data, the mesh tensor is deterministic and
identical across ranks, as DeviceMesh requires.

Note: `mesh_shape` follows the JAX convention — axes ordered by increasing
network intensity, i.e. the highest-traffic axis last. This matches the usual
PyTorch convention of data-parallel outermost.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh as _torch_init_device_mesh


@dataclasses.dataclass(frozen=True)
class _DeviceStub:
  """Duck-types the subset of jax.Device that mesh_utils touches."""

  id: int
  coords: Tuple[int, int, int]
  core_on_chip: int
  device_kind: str
  platform: str
  process_index: int
  slice_index: int


def _local_jax_device():
  """Returns the JAX device this torch process drives.

  If exactly one local device is visible (device pinned per process via the
  runtime/launcher), the binding is unambiguous and no environment variable is
  used. Otherwise LOCAL_RANK is used to identify the device.
  """
  import jax

  local_devices = jax.local_devices()
  if len(local_devices) == 1:
    return local_devices[0]

  local_rank = os.environ.get("LOCAL_RANK")
  if local_rank is None:
    raise RuntimeError(
        f"{len(local_devices)} local TPU devices are visible but LOCAL_RANK is "
        "not set, so this process cannot determine which device to use."
    )
  idx = int(local_rank)
  if not 0 <= idx < len(local_devices):
    raise RuntimeError(
        f"LOCAL_RANK={idx} is out of range for {len(local_devices)} visible "
        "local TPU devices."
    )
  return local_devices[idx]


def _gather_global_device_info(device_type: str) -> list[_DeviceStub]:
  """Each torch rank reports its JAX device; returns list indexed by rank."""
  d = _local_jax_device()

  coords = tuple(getattr(d, "coords", (0, 0, 0)))
  payload = torch.tensor(
      [
          d.id,
          *coords,
          getattr(d, "core_on_chip", 0),
          d.process_index,
          getattr(d, "slice_index", 0),
      ],
      dtype=torch.int64,
      device=device_type,
  )

  world_size = dist.get_world_size()
  gathered = [torch.zeros_like(payload) for _ in range(world_size)]
  dist.all_gather(gathered, payload)

  # device_kind / platform are uniform across a slice, so read them from the
  # local device instead of shipping strings through the collective (keeps this
  # compatible with backends that only gather tensors).
  infos = []
  for t in gathered:
    dev_id, x, y, z, core, proc, slc = t.tolist()
    infos.append(
        _DeviceStub(
            id=dev_id,
            coords=(x, y, z),
            core_on_chip=core,
            device_kind=d.device_kind,
            platform=d.platform,
            process_index=proc,
            slice_index=slc,
        )
    )
  return infos


__all__ = ["init_device_mesh", "topology_rank_mesh"]


def topology_rank_mesh(
    device_type: str,
    mesh_shape: Tuple[int, ...],
    *,
    topology: str = "jax",
    dcn_mesh_shape: Optional[Tuple[int, ...]] = None,
    process_is_granule: bool = False,
    allow_split_physical_axes: bool = False,
    contiguous_submeshes: bool = False,
) -> torch.Tensor:
  """Computes a topology-aware rank layout tensor, without building a mesh.

  Returns an int64 tensor of global torch ranks with shape `mesh_shape` (for
  topology="hybrid": elementwise `dcn_mesh_shape * mesh_shape`), suitable for
  passing directly to `DeviceMesh(device_type, mesh=...)` or any other consumer
  of rank layouts (custom mesh construction, layout benchmarking).

  Deterministic across ranks: every rank derives the layout from the same
  all-gathered device payloads, satisfying DeviceMesh's requirement that the
  mesh tensor be identical on all ranks. Requires an initialized process group

  topology="default" returns the upstream sequential layout
  (`arange(world_size).reshape(mesh_shape)`), useful for compatibility.

  See PyTorch's `init_device_mesh` for parameter semantics.
  """
  if not dist.is_initialized():
    dist.init_process_group(device_type)
  world_size = dist.get_world_size()

  if topology == "default":
    full_shape = (
        tuple(d * m for d, m in zip(dcn_mesh_shape, mesh_shape))
        if dcn_mesh_shape
        else tuple(mesh_shape)
    )
    return torch.arange(world_size, dtype=torch.int64).reshape(full_shape)

  infos = _gather_global_device_info(device_type)
  if len({i.id for i in infos}) != world_size:
    raise RuntimeError(
        "Duplicate JAX device ids across torch ranks — expected exactly one "
        "torch process per TPU device (chip in megacore mode). This usually "
        "means multiple processes resolved the same local device."
    )

  # Print before.
  if dist.get_rank() == 0:
    for rank, info in enumerate(infos):
      print(
          "Device %d at coordinates %s is driven by rank %d",
          info.id,
          info.coords,
          rank,
      )

  # infos is indexed by torch rank; invert to map JAX id -> torch rank.
  id_to_rank = {info.id: rank for rank, info in enumerate(infos)}

  # jax.devices() enumerates by device id. Some device-kind handlers in
  # mesh_utils (e.g. the v2/v3 tray reshape) are sensitive to input order, so
  # replicate that canonical ordering.
  jax_ordered = sorted(infos, key=lambda i: i.id)

  from jax.experimental import mesh_utils

  if topology == "hybrid":
    if dcn_mesh_shape is None:
      raise ValueError("topology='hybrid' requires dcn_mesh_shape")
    arranged = mesh_utils.create_hybrid_device_mesh(
        mesh_shape,
        dcn_mesh_shape,
        devices=jax_ordered,
        process_is_granule=process_is_granule,
        allow_split_physical_axes=allow_split_physical_axes,
    )
  elif topology == "jax":
    arranged = mesh_utils.create_device_mesh(
        mesh_shape,
        devices=jax_ordered,
        contiguous_submeshes=contiguous_submeshes,
        allow_split_physical_axes=allow_split_physical_axes,
    )
  else:
    raise ValueError(f"Unknown topology: {topology!r}")

  # arranged is an ndarray of _DeviceStub in logical-mesh shape; translate each
  # device back to the torch rank driving it.
  rank_mesh = np.vectorize(lambda dev: id_to_rank[dev.id], otypes=[np.int64])(
      arranged
  )

  return torch.from_numpy(rank_mesh)


def init_device_mesh(
    device_type: str,
    mesh_shape: Tuple[int, ...],
    *,
    mesh_dim_names: Optional[Tuple[str, ...]] = None,
    topology: str = "jax",
    dcn_mesh_shape: Optional[Tuple[int, ...]] = None,
    process_is_granule: bool = False,
    allow_split_physical_axes: bool = False,
    contiguous_submeshes: bool = False,
) -> DeviceMesh:
  """Drop-in replacement for `torch.distributed.init_device_mesh` on TPU.

  Maps mesh coordinates to torch ranks using JAX's topology-aware device
  arrangement, so that logically adjacent mesh positions are physically
  adjacent on the ICI fabric. The rank layout itself is computed by
  `topology_rank_mesh`, which can also be used standalone for custom
  mesh construction.

  Note: `mesh_shape` follows the JAX convention — axes ordered by increasing
  network intensity, i.e. the highest-traffic axis last. This matches the
  PyTorch convention of data-parallel outermost.

  Args:
    device_type: Device type for the mesh. Non-TPU device types are delegated to
      `torch.distributed.init_device_mesh` unchanged.
    mesh_shape: Logical mesh shape, ordered by increasing network intensity.
    mesh_dim_names: Optional names for the mesh dimensions.
    topology: "jax" (default) arranges a single slice over ICI via
      mesh_utils.create_device_mesh; "hybrid" uses create_hybrid_device_mesh for
      meshes spanning a slower outer network — pass dcn_mesh_shape, and set
      process_is_granule=True to make *hosts* the outer granule; "default"
      delegates to torch's init_device_mesh.
    dcn_mesh_shape: Outer (slower network) mesh shape for "hybrid". The returned
      mesh has shape `dcn_mesh_shape * mesh_shape` elementwise.
    process_is_granule: Treat processes (hosts) rather than slices as the
      outer-network granule in "hybrid" mode.
    allow_split_physical_axes: Forwarded to mesh_utils; permits splitting a
      physical axis across logical axes when required to realize mesh_shape.
    contiguous_submeshes: Forwarded to mesh_utils.create_device_mesh.

  Returns:
    A DeviceMesh whose rank layout follows JAX's device arrangement.
  """
  if topology == "default" or device_type not in ("tpu", "xla"):
    return _torch_init_device_mesh(
        device_type, mesh_shape, mesh_dim_names=mesh_dim_names
    )

  rank_mesh = topology_rank_mesh(
      device_type,
      mesh_shape,
      topology=topology,
      dcn_mesh_shape=dcn_mesh_shape,
      process_is_granule=process_is_granule,
      allow_split_physical_axes=allow_split_physical_axes,
      contiguous_submeshes=contiguous_submeshes,
  )

  if dist.get_rank() == 0:
    print(
        f"[topo mesh] rank layout ({tuple(rank_mesh.shape)}):\n"
        f"{rank_mesh.numpy()}"
    )

  return DeviceMesh(
      device_type,
      rank_mesh,
      mesh_dim_names=mesh_dim_names,
  )
