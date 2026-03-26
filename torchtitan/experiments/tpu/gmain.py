"""Entry point for Torchtitan TPU experiments."""

import argparse
import os
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
  config_manager = torchtitan.config.ConfigManager(
      tpu_job_config_module.TPUJobConfig)
  config = config_manager.parse_args(remaining_args)

  global_start_trainer_func(config)


def handle_main(
    start_trainer_func: Callable[[tpu_job_config_module.TPUJobConfig], None]):
  global global_start_trainer_func
  global_start_trainer_func = start_trainer_func
  app.run(main, flags_parser=(
      lambda args: (None, flags.FLAGS(args, known_only=True)[1:])))
