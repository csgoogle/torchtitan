from unittest.mock import patch

from torchtitan.experiments.tpu.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher


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
    with patch("torchtitan.experiments.tpu.grouped_experts.logger.info") as log_info:
        converted = TpuGroupedExpertsConverter.Config().build().convert(root)

    assert converted is root
    assert isinstance(root.experts, TpuGroupedExperts.Config)
    assert root.experts.dim == original.dim
    assert root.experts.hidden_dim == original.hidden_dim
    assert root.experts.num_experts == original.num_experts
    assert root.experts.token_dispatcher == original.token_dispatcher
    assert root.experts.pallas_grouped_mm_tile_k == 0
    assert root.experts.pallas_grouped_mm_tile_n == 0
    assert root.experts.pallas_grouped_mm_tgmm_tiling == (128, 128, 128)
    log_info.assert_called_once_with(
        "Replaced %d GroupedExperts configuration(s) for TPU", 1
    )
