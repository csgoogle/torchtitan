"""Tests for the discarded MoE token-counter override."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn

from torchtitan.experiments.tpu.overrides.discarded_token_counter import (
    TpuDiscardedTokenCounterMoE,
    tpu_discarded_token_counter,
)
from torchtitan.models.common.config_utils import make_moe_config
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import (
    GroupedExperts,
    MoE,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher


_NUM_EXPERTS, _TOP_K, _DIM = 4, 2, 8


def _moe_config(*, load_balance_coeff: float | None) -> MoE.Config:
  aux = load_balance_coeff is None
  return make_moe_config(
      num_experts=_NUM_EXPERTS,
      load_balance_coeff=load_balance_coeff,
      aux_loss_type="sequence_wise" if aux else None,
      aux_loss_coeff=1e-3 if aux else None,
      router=TokenChoiceTopKRouter.Config(
          num_experts=_NUM_EXPERTS,
          top_k=_TOP_K,
          gate=Linear.Config(
              in_features=_DIM, out_features=_NUM_EXPERTS, bias=False
          ),
      ),
      experts=GroupedExperts.Config(
          dim=_DIM,
          hidden_dim=_DIM,
          num_experts=_NUM_EXPERTS,
          token_dispatcher=LocalTokenDispatcher.Config(
              num_experts=_NUM_EXPERTS, top_k=_TOP_K
          ),
      ),
  )


class TpuDiscardedTokenCounterMoETest(parameterized.TestCase):

  @parameterized.parameters(1e-3, None)
  def test_counter_accumulates_only_for_the_bias_scheme(self, coeff):
    """The counter must stay untouched under the aux loss, so scan can trace it."""
    torch.manual_seed(0)
    moe = tpu_discarded_token_counter(_moe_config(load_balance_coeff=coeff)).build()
    for p in moe.parameters():
      nn.init.normal_(p, std=0.02)

    moe(torch.randn(2, 6, _DIM))

    expected = 2 * 6 * _TOP_K if coeff is not None else 0
    self.assertEqual(moe.tokens_per_expert_E.sum().item(), expected)

  def test_output_matches_the_base_class(self):
    torch.manual_seed(0)
    base = _moe_config(load_balance_coeff=None).build()
    for p in base.parameters():
      nn.init.normal_(p, std=0.02)
    overridden = tpu_discarded_token_counter(_moe_config(load_balance_coeff=None)).build()
    overridden.load_state_dict(base.state_dict())

    x = torch.randn(2, 6, _DIM)
    torch.testing.assert_close(overridden(x), base(x), rtol=0, atol=0)


if __name__ == "__main__":
  absltest.main()
