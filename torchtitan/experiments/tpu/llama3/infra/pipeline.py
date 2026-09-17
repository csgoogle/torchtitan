import torch
import torch.nn as nn

import torchtitan.components.loss
import torchtitan.config
import torchtitan.distributed
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ParallelizeFunction
import torchtitan.trainer

from torchtitan.tools.logging import logger


def pipeline_llama(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: torchtitan.trainer.Trainer.Config,
    device: torch.device,
    model_args: BaseModel.Config,
    parallelize_fn: ParallelizeFunction,
    loss_fn: torchtitan.components.loss.LossFunction,
):
  pass
