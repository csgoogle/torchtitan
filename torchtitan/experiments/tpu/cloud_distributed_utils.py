# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for configuring torch_tpu on Cloud."""

import functools
import os

import frozendict
from torchtitan.experiments.tpu import local_tpu_device
from torchtitan.tools.logging import logger


_TPU_CHIP_TO_HOST_BOUNDS = frozendict.frozendict({
    (local_tpu_device.TpuChip.V7X, 8): "2,2,1,2",
    (local_tpu_device.TpuChip.V7X, 4): "2,2,1,1",
    (local_tpu_device.TpuChip.V7X, 1): "1,1,1,1",
    (local_tpu_device.TpuChip.V6E, 8): "2,4,1",
    (local_tpu_device.TpuChip.V6E, 4): "2,2,1",
    (local_tpu_device.TpuChip.V6E, 1): "1,1,1",
    (local_tpu_device.TpuChip.V5P, 8): "2,4,1",
    (local_tpu_device.TpuChip.V5P, 4): "2,2,1",
    (local_tpu_device.TpuChip.V5P, 1): "1,1,1",
    (local_tpu_device.TpuChip.V5E, 8): "2,4,1",
    (local_tpu_device.TpuChip.V5E, 4): "2,2,1",
    (local_tpu_device.TpuChip.V5E, 1): "1,1,1",
})


def get_num_slices(environ: dict[str, str] | None = None) -> int:
  """Returns the number of TPU slices this job spans.

  Multi-slice jobs are identified by ``MEGASCALE_NUM_SLICES``, which the
  launcher sets (and which GKE propagates to every pod of a multi-slice
  JobSet). A single-slice job either omits it or sets it to 1.

  Args:
    environ: Environment to read. Defaults to ``os.environ``.

  Returns:
    The slice count, at least 1.

  Raises:
    ValueError: If ``MEGASCALE_NUM_SLICES`` is set but is not a positive
      integer.
  """
  if environ is None:
    environ = os.environ
  raw = environ.get("MEGASCALE_NUM_SLICES", "1")
  try:
    num_slices = int(raw)
  except ValueError as e:
    raise ValueError(
        f"MEGASCALE_NUM_SLICES must be a positive integer, got {raw!r}."
    ) from e
  if num_slices < 1:
    raise ValueError(
        f"MEGASCALE_NUM_SLICES must be a positive integer, got {raw!r}."
    )
  return num_slices


def get_slice_id(environ: dict[str, str] | None = None) -> int:
  """Returns this process's slice index, 0 on a single-slice job."""
  if environ is None:
    environ = os.environ
  return int(environ.get("MEGASCALE_SLICE_ID", "0"))


def is_multi_slice(environ: dict[str, str] | None = None) -> bool:
  """Whether this job spans more than one TPU slice."""
  return get_num_slices(environ) > 1


def _maybe_init_distributed_on_gke(
    slicebuilder_first_worker_port: int = 10000,
    environ: dict[str, str] | None = None,
) -> bool:
  """Initializes environment variables for distributed TPU training on GKE.

  This function configures environment variables required by SliceBuilder
  when running torch_tpu on GKE, based on the provided environment
  variables set by the Kubernetes setup. It adjusts bounds and addresses
  to match the expected configuration for distributed TPU workers.

  Args:
    slicebuilder_first_worker_port: The starting port number for SliceBuilder
      worker communication.
    environ: A dictionary of environment variables. Defaults to `os.environ`.

  Returns:
    True if the environment variables were successfully initialized for
    distributed TPU training on GKE, False otherwise.
  """
  if environ is None:
    environ = os.environ

  if (
      "TPU_WORKER_HOSTNAMES" not in environ
      or "TPU_CHIPS_PER_HOST_BOUNDS" not in environ
      or "TPU_HOST_BOUNDS" not in environ
      or "WORLD_SIZE" not in environ
      or "LOCAL_RANK" not in environ
      or "RANK" not in environ
  ):
    return False

  logger.info("Trying to initialize distributed TPU training on GKE.")
  world_size = int(environ["WORLD_SIZE"])
  local_rank = int(environ["LOCAL_RANK"])
  rank = int(environ["RANK"])

  # On a multi-slice job WORLD_SIZE spans every slice, while the TPU_*_BOUNDS
  # that GKE injects describe ONE slice. SliceBuilder is also per-slice: each
  # slice builds its own ICI mesh and MegaScale joins them over DCN. So every
  # comparison against the bounds below, and the task id handed to the slice's
  # PJRT client, must use the PER-SLICE world size and the WITHIN-SLICE rank.
  # Using the global values makes the bounds check fail and this function bail
  # out, leaving TORCH_TPU_SLICEBUILDER_ADDRESSES unset.
  num_slices = get_num_slices(environ)
  if world_size % num_slices:
    raise ValueError(
        f"WORLD_SIZE={world_size} is not divisible by "
        f"MEGASCALE_NUM_SLICES={num_slices}; every slice must contribute the "
        "same number of processes."
    )
  slice_world_size = world_size // num_slices
  slice_rank = rank % slice_world_size
  is_v7 = ("TPU_ACCELERATOR_TYPE" in environ and
           environ["TPU_ACCELERATOR_TYPE"].startswith("tpu7x"))

  tpu_chips_per_host_bounds = list(
      map(int, environ["TPU_CHIPS_PER_HOST_BOUNDS"].split(","))
  )
  tpu_chips_per_host_bounds_product = functools.reduce(
      lambda x, y: x * y, tpu_chips_per_host_bounds
  )

  if tpu_chips_per_host_bounds_product == 1:
    return False  # either single host or environment is set manually.

  tpu_host_bounds = list(map(int, environ["TPU_HOST_BOUNDS"].split(",")))
  tpu_host_bounds_product = functools.reduce(
      lambda x, y: x * y, tpu_host_bounds
  )

  tpu_worker_hostnames = environ["TPU_WORKER_HOSTNAMES"].split(",")

  if tpu_host_bounds_product != len(tpu_worker_hostnames):
    return False  # environment is likely to set incorrectly.

  if is_v7:
    tpu_chips_per_host_bounds_product *= 2

  if (
      tpu_chips_per_host_bounds_product * tpu_host_bounds_product
      != slice_world_size
  ):
    return False  # environment is likely to set incorrectly.

  environ["TPU_CHIPS_PER_HOST_BOUNDS"] = ",".join(
      ["1"] * (4 if is_v7 else 3)
  )
  for i in range(len(tpu_host_bounds)):
    tpu_host_bounds[i] = tpu_host_bounds[i] * tpu_chips_per_host_bounds[i]
  environ["TPU_HOST_BOUNDS"] = ",".join(map(str, tpu_host_bounds))
  if is_v7:
    environ["TPU_HOST_BOUNDS"] += ",2"

  environ["TPU_VISIBLE_CHIPS"] = str(local_rank)
  # Within-slice: the slice's PJRT client numbers its own tasks from zero.
  environ["CLOUD_TPU_TASK_ID"] = str(slice_rank)
  ports = list(
      range(
          slicebuilder_first_worker_port,
          slicebuilder_first_worker_port + tpu_chips_per_host_bounds_product,
      )
  )
  environ["TPU_PROCESS_PORT"] = str(ports[local_rank])
  tpu_worker_addresses = []
  for tpu_worker_hostname in tpu_worker_hostnames:
    for port in ports:
      tpu_worker_addresses.append(tpu_worker_hostname + ":" + str(port))
  tpu_worker_addresses_str = ",".join(tpu_worker_addresses)
  environ["TPU_PROCESS_ADDRESSES"] = tpu_worker_addresses_str
  environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = tpu_worker_addresses_str
  environ["TORCH_TPU_TOPOLOGY"] = environ["TPU_HOST_BOUNDS"]

  if num_slices > 1:
    # The MegaScale PJRT client is what gives devices a slice index and a
    # globally unique id; without it every slice numbers its devices from zero
    # and torch_tpu's topology_aware_mesh rejects the job with "duplicate pjrt
    # device ids across torch ranks".
    missing = [
        v
        for v in ("MEGASCALE_COORDINATOR_ADDRESS", "MEGASCALE_SLICE_ID")
        if v not in environ
    ]
    if missing:
      raise ValueError(
          f"MEGASCALE_NUM_SLICES={num_slices} requires {missing} to be set so "
          "libtpu creates the MegaScale PJRT client. Without it each slice "
          "numbers its devices from zero and the mesh cannot be built."
      )
    # The MegaScale PJRT client exposes no HostMemoryAllocator, so any pinned
    # host allocation aborts the process from inside the PJRT C API. Opt the
    # data path out of page-locked batches before the first one is built.
    environ.setdefault("TORCHTITAN_PIN_HOST_MEMORY", "0")
    if environ["TORCHTITAN_PIN_HOST_MEMORY"] != "0":
      logger.warning(
          "TORCHTITAN_PIN_HOST_MEMORY=%s was set explicitly on a multi-slice"
          " job; pinned host allocations abort under the MegaScale PJRT"
          " client, which implements no host memory allocator.",
          environ["TORCHTITAN_PIN_HOST_MEMORY"],
      )
    logger.info(
        "Multi-slice TPU: %d slices, this is slice %d. Per-slice world size"
        " %d (global %d), within-slice task id %d, per-slice topology %s.",
        num_slices,
        get_slice_id(environ),
        slice_world_size,
        world_size,
        slice_rank,
        environ["TORCH_TPU_TOPOLOGY"],
    )
  return True


def _maybe_init_distributed_on_tpu_vm() -> bool:
  """Initializes environment variables for distributed TPU training on a TPU VM.

  This function configures environment variables required for distributed
  training when running on a single TPU VM. It sets up bounds, visible chips,
  and worker addresses based on the local TPU configuration and world size.

  Returns:
    True if the environment variables were successfully initialized for
    distributed TPU training on a TPU VM, False otherwise.

  Raises:
    ValueError: If the number of local chips does not match the WORLD_SIZE.
  """
  world_size = int(os.getenv("WORLD_SIZE", "1"))
  if world_size == 1:
    return False

  logger.info("Trying to initialize distributed TPU training on TPU VM.")
  local_chips = local_tpu_device.get_local_chips()
  if local_chips is None:
    return False
  tpu_chip, num_chips = local_chips
  if tpu_chip is None:
    return False
  if num_chips is None:
    return False

  if num_chips != world_size:
    raise ValueError(
        f"Assuming single-chip setup, got {num_chips} chips"
        f" and world size {world_size}."
    )

  local_rank = int(os.environ["LOCAL_RANK"])
  os.environ["CLOUD_TPU_TASK_ID"] = str(local_rank)
  os.environ["TPU_VISIBLE_CHIPS"] = str(local_rank)
  if tpu_chip == local_tpu_device.TpuChip.V7X:
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1,1"
  else:
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
  os.environ["TPU_HOST_BOUNDS"] = _TPU_CHIP_TO_HOST_BOUNDS[tpu_chip, num_chips]

  # only works for singlehost !!!
  worker0 = os.getenv("WORKERS_0_HOSTNAME", "localhost")
  ports = list(range(10000, 10000 + world_size))
  tpu_worker_addresses = []
  for port in ports:
    tpu_worker_addresses.append(worker0 + ":" + str(port))
  tpu_worker_addresses_str = ",".join(tpu_worker_addresses)
  os.environ["TPU_PROCESS_ADDRESSES"] = tpu_worker_addresses_str
  os.environ["TPU_PROCESS_PORT"] = str(10000 + local_rank)
  os.environ["TPU_ACCELERATOR_TYPE"] = f"{tpu_chip.name}-{num_chips}"
  os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = tpu_worker_addresses_str
  os.environ["TORCH_TPU_TOPOLOGY"] = os.environ["TPU_HOST_BOUNDS"]
  return True


def apply_multislice_parallelism_defaults(parallelism_config) -> None:
  """Points the HSDP replicate axis at the slice boundary, in place.

  On a multi-slice job the whole point of HSDP is that gradients are
  reduce-scattered inside a slice over ICI and only all-reduced between slices
  over the much slower DCN. That holds when the replicate axis is exactly the
  slice count, so this defaults ``data_parallel_replicate_degree`` to the
  number of slices when the recipe left it at 1. An explicit value is
  respected, but it must be a multiple of the slice count -- otherwise a shard
  group straddles DCN, which is the layout multi-slice exists to avoid.

  No-op on a single-slice job.

  Args:
    parallelism_config: The run's ``ParallelismConfig``, mutated in place.

  Raises:
    ValueError: If an explicit replicate degree is not a multiple of the slice
      count.
  """
  num_slices = get_num_slices()
  if num_slices == 1:
    return

  replicate = parallelism_config.data_parallel_replicate_degree
  if replicate == 1:
    parallelism_config.data_parallel_replicate_degree = num_slices
    logger.info(
        "Multi-slice: defaulting data_parallel_replicate_degree to the slice"
        " count (%d) so every FSDP shard group stays inside one slice.",
        num_slices,
    )
  elif replicate % num_slices:
    raise ValueError(
        f"data_parallel_replicate_degree={replicate} is not a multiple of the"
        f" slice count {num_slices}. The replicate axis is what gets split"
        " across DCN, so an indivisible value would put FSDP shard groups on"
        " both sides of the inter-slice link."
    )
  else:
    logger.info(
        "Multi-slice: keeping the recipe's data_parallel_replicate_degree=%d"
        " across %d slices.",
        replicate,
        num_slices,
    )


def maybe_init_distributed() -> bool:
  """Initializes environment variables for distributed TPU training."""
  return _maybe_init_distributed_on_gke() or _maybe_init_distributed_on_tpu_vm()
