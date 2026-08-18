from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch

from torchtitan.experiments.tpu.grouped_experts import (
    _native_weight_sharding,
    _transpose_last_two_dims,
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.qwen3.config_registry import (
    qwen3_moe_30b_fsdp_only_full_ac_profile,
)
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.moe_sharding import expert_param_placement_dense
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
from torchtitan.protocols.sharding import ShardingConfig, SpmdLayout


def _build_experts(
    num_experts=1,
    *,
    use_pallas_grouped_mm=False,
    use_kernel_native_weight_layout=False,
    param_init=None,
):
    config = TpuGroupedExperts.Config(
        dim=16,
        hidden_dim=24,
        num_experts=num_experts,
        use_pallas_grouped_mm=use_pallas_grouped_mm,
        use_kernel_native_weight_layout=use_kernel_native_weight_layout,
        param_init=param_init,
        token_dispatcher=LocalTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=1,
        ),
    )
    experts = config.build()
    if param_init is None:
        for param in experts.parameters():
            torch.nn.init.normal_(param)
    return experts


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
    assert not root.experts.pallas_grouped_mm_use_full_tgmm_tiling


def test_fsdp_recipe_preserves_global_aux_contract_and_selects_tpu_experts():
    config = qwen3_moe_30b_fsdp_only_full_ac_profile()

    assert config.model_spec is not None
    assert config.model_spec.flavor == "30B-A3B-global-aux"
    assert config.metrics.enable_moe_routing
    assert config.optimizer.param_groups[0].optimizer_kwargs["lr"] == 3e-4
    experts = list(config.model_spec.model.traverse(TpuGroupedExperts.Config))
    assert len(experts) == 48
    assert all(expert.use_pallas_grouped_mm for _, expert, _, _ in experts)
    assert all(expert.pallas_grouped_mm_tile_m == 256 for _, expert, _, _ in experts)
    assert all(
        expert.pallas_grouped_mm_use_full_tgmm_tiling
        for _, expert, _, _ in experts
    )
    assert all(expert.use_kernel_native_weight_layout for _, expert, _, _ in experts)


def test_kernel_native_layout_requires_pallas():
    with pytest.raises(ValueError, match="Kernel-native expert weights require Pallas"):
        _build_experts(use_kernel_native_weight_layout=True)


def test_kernel_native_layout_preserves_initialization_and_state_dict():
    param_init = {
        "w1_EFD": lambda x: torch.nn.init.normal_(x, std=0.02),
        "w2_EDF": lambda x: torch.nn.init.normal_(x, std=0.01),
        "w3_EFD": lambda x: torch.nn.init.normal_(x, std=0.03),
    }
    torch.manual_seed(0)
    stock = _build_experts(num_experts=3, param_init=param_init)
    stock.init_states()
    torch.manual_seed(0)
    native = _build_experts(
        num_experts=3,
        use_pallas_grouped_mm=True,
        use_kernel_native_weight_layout=True,
        param_init=param_init,
    )
    native.init_states()

    assert native.w1_EDF.is_contiguous()
    assert native.w2_EFD.is_contiguous()
    assert native.w3_EDF.is_contiguous()
    stock_state = stock.state_dict()
    native_state = native.state_dict()
    assert set(native_state) == set(stock_state)
    for name in stock_state:
        torch.testing.assert_close(
            native_state[name], stock_state[name], rtol=0, atol=0
        )

    restored = _build_experts(
        num_experts=3,
        use_pallas_grouped_mm=True,
        use_kernel_native_weight_layout=True,
    )
    restored.load_state_dict(stock_state)
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, stock_state[name], rtol=0, atol=0)


def test_kernel_native_layout_checkpoint_hook_order():
    native = _build_experts(
        num_experts=2,
        use_pallas_grouped_mm=True,
        use_kernel_native_weight_layout=True,
    )
    save_keys = set()

    def observe_save(module, state_dict, prefix, local_metadata):
        save_keys.update(key.removeprefix(prefix) for key in state_dict)

    native.register_state_dict_post_hook(observe_save)
    state = native.state_dict()
    assert {"w1_EFD", "w2_EDF", "w3_EFD"} <= save_keys
    assert not {"w1_EDF", "w2_EFD", "w3_EDF"} & save_keys

    load_keys = set()

    def observe_load(module, state_dict, prefix, *args):
        load_keys.update(key.removeprefix(prefix) for key in state_dict)

    native.register_load_state_dict_pre_hook(observe_load)
    native.load_state_dict(state)
    assert {"w1_EDF", "w2_EFD", "w3_EDF"} <= load_keys
    assert not {"w1_EFD", "w2_EDF", "w3_EFD"} & load_keys


def test_kernel_native_layout_pallas_forward_and_backward():
    param_init = {
        name: lambda x: torch.nn.init.normal_(x, std=0.02)
        for name in ("w1_EFD", "w2_EDF", "w3_EFD")
    }
    torch.manual_seed(0)
    stock = _build_experts(num_experts=2, param_init=param_init)
    stock.init_states()
    native = _build_experts(
        num_experts=2,
        use_pallas_grouped_mm=True,
        use_kernel_native_weight_layout=True,
        param_init=param_init,
    )
    native.load_state_dict(stock.state_dict())
    x = torch.randn(7, 16, requires_grad=True)
    x_native = x.detach().clone().requires_grad_(True)
    counts = torch.tensor([3, 4], dtype=torch.int32)

    def grouped_mm(
        x, weight, counts, *, tile_m, use_full_tgmm_tiling=False
    ):
        assert weight.is_contiguous()
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
        return torch._grouped_mm(x, weight, offs=offsets)

    expected = stock._experts_forward(x, counts)
    with patch(
        "torchtitan.experiments.tpu.grouped_experts.segment_matmul_pallas_2d",
        side_effect=grouped_mm,
    ):
        actual = native._experts_forward(x_native, counts)
    torch.testing.assert_close(actual, expected)

    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(x_native.grad, x.grad, rtol=0.01, atol=1e-6)
    for name, grad in (
        ("w1_EFD", native.w1_EDF.grad.transpose(-2, -1)),
        ("w2_EDF", native.w2_EFD.grad.transpose(-2, -1)),
        ("w3_EFD", native.w3_EDF.grad.transpose(-2, -1)),
    ):
        expected_param = getattr(stock, name)
        torch.testing.assert_close(grad, expected_param.grad, rtol=0.01, atol=1e-6)


def test_kernel_native_layout_transposes_tp_sharding_axes():
    config = ShardingConfig(
        state_shardings={
            "w1_EFD": expert_param_placement_dense(tp_placement=spmd.S(1)),
            "w2_EDF": expert_param_placement_dense(tp_placement=spmd.S(2)),
            "w3_EFD": expert_param_placement_dense(tp_placement=spmd.S(1)),
        }
    )
    native = _native_weight_sharding(config)
    assert set(native.state_shardings) == {"w1_EDF", "w2_EFD", "w3_EDF"}
    assert native.state_shardings["w1_EDF"].axis_types[MeshAxisName.TP] == spmd.S(2)
    assert native.state_shardings["w2_EFD"].axis_types[MeshAxisName.TP] == spmd.S(1)
    assert native.state_shardings["w3_EDF"].axis_types[MeshAxisName.TP] == spmd.S(2)


def test_kernel_native_layout_transposes_rank_aware_partition_spec():
    layout = SpmdLayout(
        {
            MeshAxisName.DP_SHARD: spmd.V,
            MeshAxisName.TP: spmd.V,
        },
        partition_spec=(MeshAxisName.DP_SHARD, None, MeshAxisName.TP),
    )

    native = _transpose_last_two_dims(layout)

    assert native.partition_spec == (
        MeshAxisName.DP_SHARD,
        MeshAxisName.TP,
        None,
    )
