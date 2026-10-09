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

"""Multi-slice environment setup in cloud_distributed_utils.

These run on CPU: the functions under test only rewrite an environment dict.
"""

import unittest
from unittest import mock

from torchtitan.config.configs import ParallelismConfig
from torchtitan.experiments.tpu import cloud_distributed_utils as cdu


def _gke_env(
    *, num_slices: int = 2, slice_id: int = 0, vms: int = 2, world: int | None = None
) -> dict[str, str]:
  """A v7x multi-slice pod environment as GKE injects it.

  The TPU_*_BOUNDS describe ONE slice while WORLD_SIZE spans every slice; that
  asymmetry is what the code under test has to cope with.
  """
  procs_per_vm = 8  # v7x: 4 chips x 2 TensorCores
  return {
      "TPU_WORKER_HOSTNAMES": ",".join(f"h{slice_id}-{v}" for v in range(vms)),
      "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1,1",
      "TPU_HOST_BOUNDS": "1,1,2",
      "TPU_ACCELERATOR_TYPE": "tpu7x-8",
      "WORLD_SIZE": str(
          world if world is not None else num_slices * vms * procs_per_vm
      ),
      "RANK": str(slice_id * vms * procs_per_vm + 3),
      "LOCAL_RANK": "3",
      "MEGASCALE_NUM_SLICES": str(num_slices),
      "MEGASCALE_SLICE_ID": str(slice_id),
      "MEGASCALE_COORDINATOR_ADDRESS": "h0-0:8080",
  }


class SliceCountTest(unittest.TestCase):

  def test_defaults_to_one_slice(self):
    self.assertEqual(cdu.get_num_slices({}), 1)
    self.assertFalse(cdu.is_multi_slice({}))

  def test_reads_megascale_num_slices(self):
    self.assertEqual(cdu.get_num_slices({"MEGASCALE_NUM_SLICES": "4"}), 4)
    self.assertTrue(cdu.is_multi_slice({"MEGASCALE_NUM_SLICES": "2"}))

  def test_slice_id_defaults_to_zero(self):
    self.assertEqual(cdu.get_slice_id({}), 0)

  def test_slice_id_is_derived_from_the_rank(self):
    """Ranks are handed out in slice order, so MEGASCALE_SLICE_ID is optional."""
    base = {"MEGASCALE_NUM_SLICES": "2", "WORLD_SIZE": "128"}
    for rank, expected in ((0, 0), (63, 0), (64, 1), (127, 1)):
      with self.subTest(rank=rank):
        self.assertEqual(cdu.get_slice_id({**base, "RANK": str(rank)}), expected)

  def test_slice_id_derivation_handles_four_slices(self):
    self.assertEqual(
        cdu.get_slice_id(
            {"MEGASCALE_NUM_SLICES": "4", "WORLD_SIZE": "128", "RANK": "90"}
        ),
        2,
    )

  def test_slice_id_from_the_pod_hostname(self):
    """gcluster renders <workload>-main-job-<sliceIndex>-<workerIndex>."""
    for host, slices, expected in (
        ("muse-ms-main-job-1-3", "2", 1),
        ("muse-ms-main-job-0-0", "2", 0),
        ("muse-ms-main-job-3-7", "4", 3),
        ("muse-ms-main-job-9-0", "2", None),  # out of range for the slice count
        ("no-index-here", "2", None),
    ):
      with self.subTest(host=host):
        self.assertEqual(
            cdu._slice_id_from_hostname(
                {"HOSTNAME": host, "MEGASCALE_NUM_SLICES": slices}
            ),
            expected,
        )

  def test_hostname_is_used_when_no_rank_is_available_yet(self):
    """The launcher needs the slice index before torchrun assigns ranks."""
    self.assertEqual(
        cdu.get_slice_id(
            {"HOSTNAME": "w-main-job-1-5", "MEGASCALE_NUM_SLICES": "2"}
        ),
        1,
    )

  def test_rank_takes_precedence_over_the_hostname(self):
    self.assertEqual(
        cdu.get_slice_id({
            "HOSTNAME": "w-main-job-1-5",
            "MEGASCALE_NUM_SLICES": "2",
            "WORLD_SIZE": "128",
            "RANK": "3",
        }),
        0,
    )

  def test_explicit_slice_id_is_cross_checked_against_the_rank(self):
    """A launcher computing slice ids and node ranks differently would hang."""
    env = {"MEGASCALE_NUM_SLICES": "2", "WORLD_SIZE": "128", "RANK": "64"}
    self.assertEqual(cdu.get_slice_id({**env, "MEGASCALE_SLICE_ID": "1"}), 1)
    with self.assertRaisesRegex(ValueError, "disagrees with the slice implied"):
      cdu.get_slice_id({**env, "MEGASCALE_SLICE_ID": "0"})

  def test_rejects_non_positive_and_garbage(self):
    for bad in ("0", "-1", "two", ""):
      with self.subTest(value=bad):
        with self.assertRaises(ValueError):
          cdu.get_num_slices({"MEGASCALE_NUM_SLICES": bad})


class GkeMultiSliceInitTest(unittest.TestCase):

  def test_initializes_on_multi_slice(self):
    """The per-slice bounds vs global WORLD_SIZE must not bail the setup out."""
    env = _gke_env(num_slices=2, slice_id=1, vms=2)
    self.assertTrue(cdu._maybe_init_distributed_on_gke(environ=env))
    self.assertIn("TORCH_TPU_SLICEBUILDER_ADDRESSES", env)

  def test_task_id_is_within_slice(self):
    """The slice's PJRT client numbers its own tasks from zero."""
    env = _gke_env(num_slices=2, slice_id=1, vms=2)
    rank, world = int(env["RANK"]), int(env["WORLD_SIZE"])
    cdu._maybe_init_distributed_on_gke(environ=env)
    self.assertEqual(int(env["CLOUD_TPU_TASK_ID"]), rank % (world // 2))

  def test_slicebuilder_addresses_cover_only_the_local_slice(self):
    """SliceBuilder is per-slice; MegaScale joins slices over DCN."""
    env = _gke_env(num_slices=2, slice_id=0, vms=2)
    cdu._maybe_init_distributed_on_gke(environ=env)
    addrs = env["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")
    self.assertEqual(len(addrs), 2 * 8)
    hosts = {a.rsplit(":", 1)[0] for a in addrs}
    self.assertEqual(hosts, {"h0-0", "h0-1"})

  def test_single_slice_task_id_is_the_global_rank(self):
    env = _gke_env(num_slices=1, slice_id=0, vms=2)
    for k in (
        "MEGASCALE_NUM_SLICES",
        "MEGASCALE_SLICE_ID",
        "MEGASCALE_COORDINATOR_ADDRESS",
    ):
      del env[k]
    rank = int(env["RANK"])
    self.assertTrue(cdu._maybe_init_distributed_on_gke(environ=env))
    self.assertEqual(int(env["CLOUD_TPU_TASK_ID"]), rank)

  def test_requires_the_megascale_client_on_multi_slice(self):
    """Without it each slice numbers devices from zero and the mesh is rejected."""
    env = _gke_env(num_slices=2)
    del env["MEGASCALE_COORDINATOR_ADDRESS"]
    with self.assertRaisesRegex(ValueError, "MEGASCALE_COORDINATOR_ADDRESS"):
      cdu._maybe_init_distributed_on_gke(environ=env)

  def test_slice_id_is_not_required_in_the_environment(self):
    """It is derived from the rank; only the coordinator address is mandatory."""
    env = _gke_env(num_slices=2, slice_id=1, vms=2)
    del env["MEGASCALE_SLICE_ID"]
    self.assertTrue(cdu._maybe_init_distributed_on_gke(environ=env))

  def test_rejects_world_size_not_divisible_by_slices(self):
    with self.assertRaisesRegex(ValueError, "not divisible"):
      cdu._maybe_init_distributed_on_gke(environ=_gke_env(num_slices=2, world=31))

  def test_bails_out_on_inconsistent_bounds(self):
    """Divisible but the per-slice size disagrees with the bounds -> no setup."""
    self.assertFalse(
        cdu._maybe_init_distributed_on_gke(environ=_gke_env(num_slices=2, world=30))
    )


class ReplicateAxisDefaultTest(unittest.TestCase):

  def test_defaults_replicate_to_slice_count(self):
    with mock.patch.dict("os.environ", {"MEGASCALE_NUM_SLICES": "2"}):
      config = ParallelismConfig()
      cdu.apply_multislice_parallelism_defaults(config)
      self.assertEqual(config.data_parallel_replicate_degree, 2)

  def test_keeps_an_explicit_multiple(self):
    with mock.patch.dict("os.environ", {"MEGASCALE_NUM_SLICES": "2"}):
      config = ParallelismConfig(data_parallel_replicate_degree=4)
      cdu.apply_multislice_parallelism_defaults(config)
      self.assertEqual(config.data_parallel_replicate_degree, 4)

  def test_rejects_a_replicate_degree_that_straddles_dcn(self):
    with mock.patch.dict("os.environ", {"MEGASCALE_NUM_SLICES": "2"}):
      config = ParallelismConfig(data_parallel_replicate_degree=3)
      with self.assertRaisesRegex(ValueError, "not a multiple"):
        cdu.apply_multislice_parallelism_defaults(config)

  def test_noop_on_single_slice(self):
    with mock.patch.dict("os.environ", {"MEGASCALE_NUM_SLICES": "1"}):
      config = ParallelismConfig()
      cdu.apply_multislice_parallelism_defaults(config)
      self.assertEqual(config.data_parallel_replicate_degree, 1)


class PinnedHostMemoryTest(unittest.TestCase):
  """Multi-slice must opt out of page-locked batches.

  The MegaScale PJRT client implements no HostMemoryAllocator, and the pinned
  allocation aborts the process from inside the PJRT C API rather than raising,
  so the decision has to be made before the first batch is built. ``collators``
  is left untouched: ``TextCollator`` reads ``HAS_PIN_MEMORY`` as a module
  global on every call, so rebinding it is enough.
  """

  def test_disables_pinning_and_is_idempotent(self):
    from torchtitan.components.data import collators
    from torchtitan.experiments.tpu import workarounds

    original = collators.HAS_PIN_MEMORY
    try:
      collators.HAS_PIN_MEMORY = True
      workarounds.disable_pinned_host_memory()
      self.assertFalse(collators.HAS_PIN_MEMORY)
      workarounds.disable_pinned_host_memory()  # no-op the second time
      self.assertFalse(collators.HAS_PIN_MEMORY)
    finally:
      collators.HAS_PIN_MEMORY = original


if __name__ == "__main__":
  unittest.main()
