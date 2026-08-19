from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.config import derive
from torchtitan.experiments.tpu.kernels.expert_segment_matmul import (
    segment_matmul_pallas_2d,
)
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.model import ModelConfigConverter


class TpuGroupedExperts(GroupedExperts):
  """TPU expert implementation with an opt-in dynamic MegaBlox path."""

  @dataclass(kw_only=True, slots=True)
  class Config(GroupedExperts.Config):
    use_pallas_grouped_mm: bool = False
    pallas_grouped_mm_tile_m: int = 1024

  def __init__(self, config: Config):
    super().__init__(config)
    self.use_pallas_grouped_mm = config.use_pallas_grouped_mm
    self.pallas_grouped_mm_tile_m = config.pallas_grouped_mm_tile_m

  def _experts_forward(
      self,
      x_RD: torch.Tensor,
      num_tokens_per_expert_E: torch.Tensor,
  ) -> torch.Tensor:
    # The Qwen TPU recipe uses Pallas because native torch._grouped_mm mode-2
    # dWeight is incorrect at M=4096. See
    # https://github.com/google-pytorch/torch_tpu/issues/2668.
    if not self.use_pallas_grouped_mm:
      return super()._experts_forward(x_RD, num_tokens_per_expert_E)

    if isinstance(self.w1_EFD, DTensor):
      w1_EFD = self.w1_EFD.to_local()
      assert isinstance(self.w2_EDF, DTensor)
      w2_EDF = self.w2_EDF.to_local()
      assert isinstance(self.w3_EFD, DTensor)
      w3_EFD = self.w3_EFD.to_local()
    else:
      w1_EFD = self.w1_EFD
      w2_EDF = self.w2_EDF
      w3_EFD = self.w3_EFD

    x_bf16 = x_RD.bfloat16()
    tile_m = self.pallas_grouped_mm_tile_m
    hidden_RF = F.silu(
        segment_matmul_pallas_2d(
            x_bf16,
            w1_EFD.bfloat16().transpose(-2, -1),
            num_tokens_per_expert_E,
            tile_m=tile_m,
        )
    )
    hidden_RF = hidden_RF * segment_matmul_pallas_2d(
        x_bf16,
        w3_EFD.bfloat16().transpose(-2, -1),
        num_tokens_per_expert_E,
        tile_m=tile_m,
    )
    return segment_matmul_pallas_2d(
        hidden_RF,
        w2_EDF.bfloat16().transpose(-2, -1),
        num_tokens_per_expert_E,
        tile_m=tile_m,
    ).type_as(x_RD)


class TpuGroupedExpertsConverter(ModelConfigConverter):
  """Select the TPU grouped-expert implementation before construction."""

  @dataclass(kw_only=True, slots=True)
  class Config(ModelConfigConverter.Config):
    pass

  def __init__(self, config: Config):
    pass

  def convert(self, model_config):
    matches = list(model_config.traverse(GroupedExperts.Config))
    for _, config, parent, field_name in matches:
      replacement = derive(config, TpuGroupedExperts.Config)
      if parent is None:
        model_config = replacement
      elif isinstance(parent, list):
        assert isinstance(field_name, int)
        parent[field_name] = replacement
      else:
        assert isinstance(field_name, str)
        setattr(parent, field_name, replacement)
    return model_config


__all__ = ["TpuGroupedExperts", "TpuGroupedExpertsConverter"]
