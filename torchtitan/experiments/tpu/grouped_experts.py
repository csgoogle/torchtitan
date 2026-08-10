from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.config import derive
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.model import ModelConfigConverter


class TpuGroupedExperts(GroupedExperts):
  """Grouped experts with a dense fast path for one local expert."""

  @dataclass(kw_only=True, slots=True)
  class Config(GroupedExperts.Config):
    pass

  def _experts_forward(
      self,
      x_RD: torch.Tensor,
      num_tokens_per_expert_E: torch.Tensor,
  ) -> torch.Tensor:
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

    if w1_EFD.shape[0] != 1:
      return super()._experts_forward(x_RD, num_tokens_per_expert_E)

    x_bf16 = x_RD.bfloat16()
    w1_FD = w1_EFD[0].bfloat16()
    w2_DF = w2_EDF[0].bfloat16()
    w3_FD = w3_EFD[0].bfloat16()
    hidden_RF = F.silu(F.linear(x_bf16, w1_FD)) * F.linear(x_bf16, w3_FD)
    return F.linear(hidden_RF, w2_DF).type_as(x_RD)


class TpuGroupedExpertsConverter(ModelConfigConverter):
  """Select the TPU grouped-expert implementation before model construction."""

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
