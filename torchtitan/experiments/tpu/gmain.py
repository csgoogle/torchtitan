"""Entry point for Torchtitan TPU experiments."""

import argparse
import os
import typing
from typing import Callable, List, Sequence, Tuple

from absl import flags
from absl.flags import argparse_flags
import torch.multiprocessing as mp
import torchtitan.config
from torchtitan.experiments.tpu import distributed_utils as tpu_distributed_utils
from torchtitan.experiments.tpu import utils as tpu_utils
import torchtitan.experiments.tpu.tpu_job_config as tpu_job_config_module

from absl import app


global_start_trainer_func = None


def _start_trainer(config: tpu_job_config_module.TPUTrainerConfig):
  """Starts the trainer with the specified eager mode if enabled."""
  global global_start_trainer_func  # pylint: disable=global-variable-not-assigned
  if os.environ.get("TORCHTITAN_DEVICE_TYPE", ""):
    tpu_utils.set_device_type(os.environ["TORCHTITAN_DEVICE_TYPE"])
  if tpu_utils.get_device_type() == "tpu" and config.tpu_config.eager_mode:
    from torch_tpu._internal import execution_mode  # pylint: disable=g-import-not-at-top

    eager_mode_enum = getattr(
        execution_mode.EagerMode, config.tpu_config.eager_mode
    )
    with execution_mode.set_eager_mode(eager_mode_enum):
      global_start_trainer_func(config)
  else:
    global_start_trainer_func(config)


def _parse_flags(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
  """Parses command-line flags."""
  parser = argparse_flags.ArgumentParser(
      description="Minimal trainer for TorchTitan models.",
      inherited_absl_flags=flags.FLAGS,
  )
  parser.add_argument(
      "--device",
      help="device to run the training on",
      choices=["cpu", "cuda", "tpu"],
      default="",
      required=True,
  )
  parser.add_argument(
      "--nproc_per_node",
      help="Number of devices to run the training on",
      type=int,
      default=1,
  )
  args, remaining_args = parser.parse_known_args(argv[1:])
  return args, remaining_args


def main(parsed_args: Tuple[argparse.Namespace, List[str]]):
  args, remaining_args = parsed_args
  config_manager = torchtitan.config.ConfigManager()
  config = typing.cast(
      tpu_job_config_module.TPUTrainerConfig,
      config_manager.parse_args(remaining_args),
  )

  _start_trainer(config)


def handle_main(
    start_trainer_func: Callable[[tpu_job_config_module.TPUTrainerConfig], None],
):
  global global_start_trainer_func
  global_start_trainer_func = start_trainer_func
  app.run(main, flags_parser=(
      lambda args: (None, flags.FLAGS(args, known_only=True)[1:])))
