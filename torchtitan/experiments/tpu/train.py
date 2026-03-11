# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer compatible with TorchTpu."""

import argparse
import json
import tempfile
from typing import List, Optional, Sequence, Tuple

from absl import flags
from absl.flags import argparse_flags
import os as os
import torch
from torchtitan.config import ConfigManager
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
import torchtitan.train

from torchtitan.experiments.tpu import utils as tpu_utils
import torchtitan.experiments.tpu.deepseek_v3   # trigger model registration
import torchtitan.experiments.tpu.flux  # trigger model registration
import torchtitan.experiments.tpu.llama3  # trigger model registration
import torchtitan.experiments.tpu.qwen3   # trigger model registration
import torchtitan.experiments.tpu.tpu_job_config as tpu_job_config_module

from absl import app


def start_trainer(config: tpu_job_config_module.TPUJobConfig):
  """Starts the training process."""
  rank = int(os.getenv("RANK", "0"))
  if rank == 0:
    init_logger()

  trainer: Optional[torchtitan.train.Trainer] = None
  if config.model.name == "flux":
    from torchtitan.models.flux.train import FluxTrainer
    trainer_cls = FluxTrainer
  else:
    trainer_cls = torchtitan.train.Trainer

  try:
    trainer = trainer_cls(config)

    if config.checkpoint.create_seed_checkpoint:
      assert int(os.environ["WORLD_SIZE"]) == 1, (
          "Must create seed checkpoint using a single device, to disable"
          " sharding."
      )
      assert (
          config.checkpoint.enable
      ), "Must enable checkpointing when creating a seed checkpoint."
      trainer.checkpointer.save(curr_step=0, last_step=True)
      logger.info("Created seed checkpoint")
    else:
      if rank == 0:
        logger.info(
            "Training with config: %s",
            json.dumps(config.to_dict(), indent=2, sort_keys=True)
        )
      trainer.train()
  except Exception:
    if trainer:
      trainer.close()
    raise
  else:
    trainer.close()
    torch.distributed.destroy_process_group()
    logger.info("Process group destroyed")


def main(parsed_args: Tuple[argparse.Namespace, List[str]]):
  args, remaining_args = parsed_args
  config_manager = ConfigManager(tpu_job_config_module.TPUJobConfig)
  config = config_manager.parse_args(remaining_args)

  device_type = utils.get_device_type()
  print("Selected device type: ", device_type)

  device = tpu_utils.get_device()  # Initialize TPU device if needed.

  if device.type == "tpu" and config.optimizer.implementation == "fused":
    # TODO: b/45811390 - Remove this once fused optimizer is supported on TPU.
    logger.warning(
        "Fused optimizer is not supported on TPU, changing to foreach."
    )
    config.optimizer.implementation = "foreach"

  start_trainer(config)


if __name__ == "__main__":
  app.run(main, flags_parser=(
      lambda args: (None, flags.FLAGS(args, known_only=True)[1:])))
