import torch
import torch.nn as nn

import torchtitan.components.loss
import torchtitan.protocols.train_spec
import torchtitan.config
import torchtitan.distributed
import torchtitan.trainer

from torchtitan.protocols.train_spec import BaseModelArgs, ParallelizeFunction
from torchtitan.tools.logging import logger


def pipeline_llama(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: torchtitan.trainer.Trainer.Config,
    device: torch.device,
    model_args: torchtitan.protocols.train_spec.BaseModelArgs,
    parallelize_fn: torchtitan.protocols.train_spec.ParallelizeFunction,
    loss_fn: torchtitan.components.loss.LossFunction,
):
  pass
