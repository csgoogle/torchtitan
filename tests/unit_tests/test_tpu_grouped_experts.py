import copy

import torch
import torch.nn.functional as F

from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher


def _reference(x, w1, w2, w3):
    x_bf16 = x.bfloat16()
    hidden = F.silu(F.linear(x_bf16, w1[0].bfloat16()))
    hidden = hidden * F.linear(x_bf16, w3[0].bfloat16())
    return F.linear(hidden, w2[0].bfloat16()).type_as(x)


def _build_experts(num_experts=1):
    config = TpuGroupedExperts.Config(
        dim=16,
        hidden_dim=24,
        num_experts=num_experts,
        token_dispatcher=LocalTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=1,
        ),
    )
    experts = config.build()
    for param in experts.parameters():
        torch.nn.init.normal_(param)
    return experts


def test_dense_single_expert_forward_and_backward():
    torch.manual_seed(0)
    experts = _build_experts()
    reference = copy.deepcopy(experts)
    x = torch.randn(13, 16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    counts = torch.tensor([13], dtype=torch.int32)

    actual = experts._experts_forward(x, counts)
    expected = _reference(
        x_ref,
        reference.w1_EFD,
        reference.w2_EDF,
        reference.w3_EFD,
    )
    torch.testing.assert_close(actual, expected)

    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    for actual_param, expected_param in zip(
        experts.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_param.grad, expected_param.grad)


def test_multiple_experts_fall_back_to_grouped_mm():
    torch.manual_seed(0)
    experts = _build_experts(num_experts=2)
    reference = copy.deepcopy(experts)
    x = torch.randn(7, 16)
    counts = torch.tensor([3, 4], dtype=torch.int32)

    actual = experts._experts_forward(x, counts)
    expected = super(TpuGroupedExperts, reference)._experts_forward(x, counts)
    torch.testing.assert_close(actual, expected)


def test_model_converter_preserves_config_fields():
    class RootConfig:
        def __init__(self):
            self.experts = GroupedExperts.Config(
                dim=16,
                hidden_dim=24,
                num_experts=8,
                token_dispatcher=LocalTokenDispatcher.Config(
                    num_experts=8,
                    top_k=2,
                ),
            )

        def traverse(self, config_type):
            if isinstance(self.experts, config_type):
                yield "experts", self.experts, self, "experts"

    root = RootConfig()
    original = root.experts
    converted = TpuGroupedExpertsConverter.Config().build().convert(root)

    assert converted is root
    assert isinstance(root.experts, TpuGroupedExperts.Config)
    assert root.experts.dim == original.dim
    assert root.experts.hidden_dim == original.hidden_dim
    assert root.experts.num_experts == original.num_experts
    assert root.experts.token_dispatcher == original.token_dispatcher
