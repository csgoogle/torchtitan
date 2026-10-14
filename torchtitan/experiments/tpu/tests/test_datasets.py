"""Tests for TPU dataset selection in `experiments/tpu/datasets.py`."""

import os
import tempfile

from absl.testing import absltest

from torchtitan.experiments.tpu import datasets
from torchtitan.hf_datasets.text_datasets import DATASETS


class TextDatasetTest(absltest.TestCase):

  def test_c4_test_default_uses_repo_asset(self) -> None:
    ds = datasets._text_dataset("c4_test", None)
    self.assertIs(ds, DATASETS["c4_test"])
    data_files = ds.source.load_dataset_kwargs["data_files"]
    self.assertEqual(
        data_files, "tests/assets/c4_test/data.json"
    )

  def test_c4_keeps_upstream_source(self) -> None:
    ds = datasets._text_dataset("c4", None)
    self.assertEqual(ds, DATASETS["c4"])
    self.assertNotIn("data_files", ds.source.load_dataset_kwargs)

  def test_c4_dataset_path_replaces_source_path(self) -> None:
    ds = datasets._text_dataset("c4", "/data/c4_mirror")
    upstream = DATASETS["c4"]
    self.assertEqual(ds.source.path, "/data/c4_mirror")
    self.assertEqual(ds.source.name, upstream.source.name)
    self.assertEqual(
        ds.source.load_dataset_kwargs, upstream.source.load_dataset_kwargs
    )

  def test_c4_test_dataset_path_merges_kwargs(self) -> None:
    with tempfile.TemporaryDirectory() as d:
      ds = datasets._text_dataset("c4_test", d)
      self.assertEqual(
          ds.source.load_dataset_kwargs["data_files"],
          os.path.join(d, "data.json"),
      )
    ds = datasets._text_dataset("c4_test", "/tmp/custom.json")
    self.assertEqual(
        ds.source.load_dataset_kwargs["data_files"], "/tmp/custom.json"
    )
    self.assertEqual(ds.source.path, DATASETS["c4_test"].source.path)


if __name__ == "__main__":
  absltest.main()
