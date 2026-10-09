"""Unit tests for single-slice and multi-slice GKE distributed initialization."""

from absl.testing import absltest
from torchtitan.experiments.tpu import cloud_distributed_utils


class CloudDistributedUtilsTest(absltest.TestCase):

  def test_single_slice_gke_v6e(self):
    env = {
        "TPU_WORKER_HOSTNAMES": "pod-0.jobset",
        "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
        "TPU_HOST_BOUNDS": "1,1,1",
        "TPU_ACCELERATOR_TYPE": "tpu-v6e-slice",
        "WORLD_SIZE": "4",
        "LOCAL_RANK": "3",
        "RANK": "3",
    }
    self.assertTrue(
        cloud_distributed_utils._maybe_init_distributed_on_gke(environ=env)
    )
    self.assertEqual(env["CLOUD_TPU_TASK_ID"], "3")
    self.assertEqual(env["TPU_VISIBLE_CHIPS"], "3")
    self.assertEqual(env["TPU_HOST_BOUNDS"], "2,2,1")

  def test_multislice_gke_v5p(self):
    env = {
        "TPU_WORKER_HOSTNAMES": "pod-1.jobset",
        "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
        "TPU_HOST_BOUNDS": "1,1,1",
        "TPU_ACCELERATOR_TYPE": "tpu-v5p-slice",
        "MEGASCALE_NUM_SLICES": "2",
        "MEGASCALE_COORDINATOR_ADDRESS": "pod-0.jobset:8080",
        "WORLD_SIZE": "8",
        "LOCAL_RANK": "2",
        "RANK": "6",
    }
    self.assertTrue(
        cloud_distributed_utils._maybe_init_distributed_on_gke(environ=env)
    )
    self.assertEqual(env["CLOUD_TPU_TASK_ID"], "2")
    self.assertEqual(env["MEGASCALE_COORDINATOR_ADDRESS"], "pod-0.jobset:12345")
    self.assertEqual(env["MEGASCALE_PORT"], "12347")
    self.assertEqual(env["MEGASCALE_DEBUG_PORT"], "8083")

  def test_multislice_gke_v7x(self):
    env = {
        "TPU_WORKER_HOSTNAMES": "pod-1.jobset",
        "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
        "TPU_HOST_BOUNDS": "1,1,1",
        "TPU_ACCELERATOR_TYPE": "tpu7x",
        "MEGASCALE_NUM_SLICES": "2",
        "MEGASCALE_COORDINATOR_ADDRESS": "pod-0.jobset:8080",
        "WORLD_SIZE": "16",
        "LOCAL_RANK": "7",
        "RANK": "15",
    }
    self.assertTrue(
        cloud_distributed_utils._maybe_init_distributed_on_gke(environ=env)
    )
    self.assertEqual(env["CLOUD_TPU_TASK_ID"], "7")
    self.assertEqual(env["TPU_HOST_BOUNDS"], "2,2,1,2")
    self.assertEqual(env["MEGASCALE_PORT"], "12352")
    self.assertEqual(env["MEGASCALE_DEBUG_PORT"], "8088")

  def test_multislice_world_size_mismatch_returns_false(self):
    env = {
        "TPU_WORKER_HOSTNAMES": "pod-0.jobset",
        "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
        "TPU_HOST_BOUNDS": "1,1,1",
        "MEGASCALE_NUM_SLICES": "2",
        "WORLD_SIZE": "4",
        "LOCAL_RANK": "0",
        "RANK": "0",
    }
    self.assertFalse(
        cloud_distributed_utils._maybe_init_distributed_on_gke(environ=env)
    )


if __name__ == "__main__":
  absltest.main()
