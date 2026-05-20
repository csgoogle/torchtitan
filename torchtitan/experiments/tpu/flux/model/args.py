# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from torch import nn
from torchtitan.config import Configurable
from torchtitan.models.flux.model.autoencoder import AutoEncoderParams
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class FluxModelArgs(BaseModelArgs):
  in_channels: int = 64
  out_channels: int = 64
  vec_in_dim: int = 768
  context_in_dim: int = 4096
  hidden_size: int = 3072
  mlp_ratio: float = 4.0
  num_heads: int = 24
  depth: int = 19
  depth_single_blocks: int = 38
  axes_dim: tuple[int, int, int] = (16, 56, 56)
  theta: int = 10_000
  qkv_bias: bool = True
  autoencoder_params: AutoEncoderParams | None = None

  def update_from_config(
      self, job_config: Configurable.Config, **kwargs
  ) -> None:
    # Lazy import to avoid circular dependencies.
    from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
    # Check if we are running a TPU Job
    if isinstance(job_config, TPUTrainerConfig):
      logger.info("TPUTrainerConfig detected for Flux.")
      pass

  def get_nparams_and_flops(
      self, model: nn.Module, seq_len: int
  ) -> tuple[int, int]:
    # TODO: add actual calculation after figuring out where to call/use this
    # func now that args is deprecated.
    pass
