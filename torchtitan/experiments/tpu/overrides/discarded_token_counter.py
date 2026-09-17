"""Sends `MoE`'s `tokens_per_expert_E` accumulation to a throwaway tensor.

The counter only ever feeds the expert-bias update, so with no
`load_balance_coeff` nothing reads it -- but `MoE.forward` writes it anyway,
and a scan body cannot write to a tensor it closed over.
"""

from dataclasses import dataclass

import torch

from torchtitan.config import derive, override
from torchtitan.models.common.moe import MoE


class TpuDiscardedTokenCounterMoE(MoE):
  """`MoE` that hands out a throwaway in place of its token counter.

  Only the `tokens_per_expert_E` attribute differs, and only when
  `load_balance_coeff` is unset. Every parameter, buffer, and checkpoint key is
  inherited unchanged, and the output is bit-identical to the base class.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(MoE.Config):
    pass

  def __getattr__(self, name: str):
    value = super().__getattr__(name)
    if name != "tokens_per_expert_E" or self.load_balance_coeff is not None:
      return value
    # The base class's `add_` has to land on a tensor allocated *inside* the
    # scan body. Swapping `_buffers` around the call instead would not do: that
    # is a mutation of state the body closed over, which is the very thing a
    # higher-order op rejects. A plain property cannot do this either --
    # `register_buffer` probes `hasattr` and would refuse the name.
    return torch.zeros_like(value)


@override(
    "tpu_discarded_token_counter",
    target=MoE.Config,
    exact=True,
    description=(
        "Send MoE's dead token-counter accumulation to a throwaway so a"
        " decoder layer can be traced inside scan."
    ),
)
def tpu_discarded_token_counter(
    config: MoE.Config,
) -> TpuDiscardedTokenCounterMoE.Config:
  return derive(config, TpuDiscardedTokenCounterMoE.Config)


__all__ = ["TpuDiscardedTokenCounterMoE", "tpu_discarded_token_counter"]
