# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Grain dataset/dataloader configs shared by the TPU recipes.

Upstream d771abb0 migrated the data stack to Grain and deleted
`HuggingFaceTextDataLoader`, which every TPU `config_registry.py` used as
`HuggingFaceTextDataLoader.Config(dataset="c4_test")`. Upstream recipes now
spell the same thing as::

    GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        shuffle=False,
    )

That does not work verbatim here: `DATASETS["c4_test"]` hardcodes the
repo-relative asset path `./tests/assets/c4_test/data.json`, which does not
resolve under internal runfiles, and the copybara rule that rewrites
`./tests/assets/` to `tests/assets/` is scoped to
`experiments/**`, so it never reaches `hf_datasets/`. Centralizing the dataset
here -- under `experiments/` -- puts the literal in that rule's scope, so the
path stays correct in the internal tree and is rewritten back on export.

Everything except the asset path is inherited from the upstream entry via
`dataclasses.replace`, so upstream changes to the processor or the filters flow
through without another edit here.
"""

import dataclasses
import os
from typing import Any

# Imported from the defining submodules rather than the
# `torchtitan.components.data` package: [internal link] cannot tell a
# package attribute from an unlisted submodule, so the package-level form
# is reported as an ambiguous import.
from torchtitan.components.data.dataset import DatasetConfig, SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.hf_datasets.text_datasets import DATASETS

__all__ = [
    "C4_TEST",
    "TPUGrainDataLoader",
    "c4_test_dataloader",
]

# Runfiles-relative location of the checked-in c4 sample used by every TPU
# test recipe. Kept as one literal so the copybara path rewrite has a single
# anchor.
_C4_TEST_DATA_FILES = "tests/assets/c4_test/data.json"


def _internal_c4_test(data_files: str = _C4_TEST_DATA_FILES) -> SingleDatasetConfig:
  """Repoints upstream's `c4_test` entry at the internal asset path."""
  upstream = DATASETS["c4_test"]
  return dataclasses.replace(
      upstream,
      source=dataclasses.replace(
          upstream.source,
          load_dataset_kwargs={"data_files": data_files},
      ),
  )


C4_TEST = _internal_c4_test()


class TPUGrainDataLoader(GrainDataLoader):
  """GrainDataLoader wrapper preserving TPU CLI flags (`dataset`, `dataset_path`)."""

  @dataclasses.dataclass(kw_only=True, slots=True)
  class Config(GrainDataLoader.Config):
    dataset: str = "c4_test"  # pytype: disable=signature-mismatch
    dataset_path: str | None = None

  def __init__(self, config: Config, **kwargs: Any) -> None:
    if isinstance(config.dataset, str):
      if config.dataset in DATASETS:
        data_files = _C4_TEST_DATA_FILES
        if config.dataset_path:
          if os.path.isdir(
              config.dataset_path
          ) or not config.dataset_path.endswith(".json"):
            data_files = os.path.join(config.dataset_path, "data.json")
          else:
            data_files = config.dataset_path
        base_ds = DATASETS[config.dataset]
        ds_cfg = ConcatThenSplitPackingConfig(
            dataset=dataclasses.replace(
                base_ds,
                source=dataclasses.replace(
                    base_ds.source,
                    load_dataset_kwargs={"data_files": data_files},
                ),
            )
        )
      else:
        from torchtitan.models.flux.flux_datasets import (  # pylint: disable=g-import-not-at-top
            DATASETS as FLUX_DATASETS,
            FluxSampleProcessor,
        )

        base_ds = FLUX_DATASETS[config.dataset]
        src = base_ds.source
        if config.dataset_path:
          src = dataclasses.replace(src, path=config.dataset_path)
        proc = base_ds.processor
        if isinstance(proc, FluxSampleProcessor.Config) and proc.img_size is None:
          proc = dataclasses.replace(proc, img_size=256)
        ds_cfg = dataclasses.replace(base_ds, source=src, processor=proc)
      config = dataclasses.replace(config, dataset=ds_cfg)  # pytype: disable=wrong-arg-types
    super().__init__(config, **kwargs)


def c4_test_dataloader(**kwargs: Any) -> TPUGrainDataLoader.Config:
  """Builds the packed `c4_test` dataloader used by the TPU test recipes.

  Args:
    **kwargs: Forwarded to `TPUGrainDataLoader.Config`; use this to override
      defaults such as `shuffle` or `seed`.

  Returns:
    A dataloader config that concatenates then splits `c4_test` into
    fixed-length sequences, while keeping `--dataloader.dataset` and
    `--dataloader.dataset_path` CLI-configurable for the TPU test harnesses.
  """
  kwargs.setdefault("shuffle", False)
  return TPUGrainDataLoader.Config(**kwargs)

