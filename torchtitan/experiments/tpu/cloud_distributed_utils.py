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


def _maybe_init_distributed_on_gke(
    slicebuilder_first_worker_port: int = 100000,
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
  ):
    return False

  logger.info("Trying to initialize distributed TPU training on GKE.")
  world_size = int(environ["WORLD_SIZE"])
  local_rank = int(environ["LOCAL_RANK"])
  rank = int(environ["RANK"])
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

  if tpu_chips_per_host_bounds_product * tpu_host_bounds_product != world_size:
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
  environ["CLOUD_TPU_TASK_ID"] = str(rank)
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


def maybe_init_distributed() -> bool:
  """Initializes environment variables for distributed TPU training."""
  return _maybe_init_distributed_on_gke() or _maybe_init_distributed_on_tpu_vm()
