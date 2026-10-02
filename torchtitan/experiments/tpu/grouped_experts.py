from dataclasses import dataclass, replace

import torch
import spmd_types as spmd
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.config import derive
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.experiments.tpu.kernels.expert_segment_matmul import (
    segment_matmul_pallas_2d,
)
from torchtitan.experiments.tpu.kernels.paired_segment_matmul import (
    paired_segment_matmul_pallas_2d,
)
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.sharding import ShardingConfig, SpmdLayout
from torchtitan.tools.logging import logger


_CANONICAL_TO_NATIVE_WEIGHT = {
    "w1_EFD": "w1_EDF",
    "w2_EDF": "w2_EFD",
    "w3_EFD": "w3_EDF",
}


def _transpose_last_two_dims(layout: SpmdLayout, *, ndim: int = 3) -> SpmdLayout:
    """Transpose tensor-dimension references in an SPMD layout."""

    if ndim < 2:
        raise ValueError(f"Expected rank of at least 2, got {ndim}")

    permutation = list(range(ndim))
    permutation[-2], permutation[-1] = permutation[-1], permutation[-2]

    axis_types = {}
    for axis, axis_type in layout.axis_types.items():
        if isinstance(axis_type, spmd.Shard):
            if not 0 <= axis_type.dim < ndim:
                raise ValueError(
                    f"Tensor dimension {axis_type.dim} is invalid for rank {ndim}"
                )
            axis_type = spmd.S(permutation[axis_type.dim])
        axis_types[axis] = axis_type

    partition_spec = layout.partition_spec
    if partition_spec is not None:
        if len(partition_spec) != ndim:
            raise ValueError(
                f"Expected rank-{ndim} partition spec, got {len(partition_spec)} dims"
            )
        partition_spec = tuple(partition_spec[dim] for dim in permutation)
    return replace(
        layout,
        axis_types=axis_types,
        partition_spec=partition_spec,
    )


def _native_weight_sharding(config: ShardingConfig) -> ShardingConfig:
    """Map canonical expert state names/layouts to kernel-native storage."""

    state_shardings = dict(config.state_shardings)
    for canonical_name, native_name in _CANONICAL_TO_NATIVE_WEIGHT.items():
        state_shardings[native_name] = _transpose_last_two_dims(
            state_shardings.pop(canonical_name)
        )
    return replace(config, state_shardings=state_shardings)


class TpuGroupedExperts(GroupedExperts):
    """TPU expert kernels with opt-in Pallas grouped paths.

    The optional kernel-native layout stores each expert weight in the exact
    ``(E, K, N)`` order consumed by Pallas, where ``K`` is the contracted input
    dimension and ``N`` is the output dimension. Thus ``K=D, N=F`` for w1/w3,
    while ``K=F, N=D`` for w2. State-dict hooks retain the canonical Qwen
    ``w1_EFD`` / ``w2_EDF`` / ``w3_EFD`` checkpoint contract.

    Translation hooks register during construction. Later save hooks on this
    module see canonical keys; later load pre-hooks see native keys.

    The paired path changes only W1/W3 execution: their parameters, FSDP
    collectives, checkpoint keys, and the W2 projection remain independent.
    The unpaired dWeight kernel retains FP32 accumulation and stores in the
    final FSDP reduction dtype selected by the parallelizer.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        activation: str = "silu"
        use_pallas_grouped_mm: bool = False
        # Share routed-token loads between the equal-shaped W1/W3 projections.
        use_pallas_paired_grouped_mm: bool = False
        pallas_grouped_mm_tile_m: int = 1024
        pallas_grouped_mm_tile_k: int = 0
        pallas_grouped_mm_tile_n: int = 0
        pallas_grouped_mm_tgmm_tiling: tuple[int, int, int] = (128, 128, 128)
        use_kernel_native_weight_layout: bool = False

    def __init__(self, config: Config):
        if config.activation not in ("silu", "gelu_tanh"):
            raise ValueError(
                f"Unsupported expert activation: {config.activation!r}"
            )
        if config.use_kernel_native_weight_layout:
            if not config.use_pallas_grouped_mm:
                raise ValueError(
                    "Kernel-native expert weights require Pallas grouped matmul"
                )
            Module.__init__(self)
            self.num_experts = config.num_experts
            self.w1_EDF = torch.nn.Parameter(
                torch.empty(config.num_experts, config.dim, config.hidden_dim)
            )
            self.w2_EFD = torch.nn.Parameter(
                torch.empty(config.num_experts, config.hidden_dim, config.dim)
            )
            self.w3_EDF = torch.nn.Parameter(
                torch.empty(config.num_experts, config.dim, config.hidden_dim)
            )
            self.token_dispatcher = config.token_dispatcher.build()
        else:
            super().__init__(config)

        self.activation = config.activation
        self.use_pallas_grouped_mm = config.use_pallas_grouped_mm
        self.use_pallas_paired_grouped_mm = config.use_pallas_paired_grouped_mm
        self.pallas_grouped_mm_tile_m = config.pallas_grouped_mm_tile_m
        self.pallas_grouped_mm_tile_k = config.pallas_grouped_mm_tile_k
        self.pallas_grouped_mm_tile_n = config.pallas_grouped_mm_tile_n
        self.pallas_grouped_mm_tgmm_tiling = config.pallas_grouped_mm_tgmm_tiling
        # Standalone modules preserve the historical FP32 result. Qwen's
        # parallelizer replaces this with the final, post-CLI FSDP reduction
        # dtype before tracing or sharding.
        self._unpaired_dweight_output_dtype = torch.float32
        self.use_kernel_native_weight_layout = config.use_kernel_native_weight_layout
        self._native_sharding_applied = False

        if self.use_pallas_paired_grouped_mm:
            if not self.use_pallas_grouped_mm:
                raise ValueError("Paired grouped matmul requires Pallas")
            if self.pallas_grouped_mm_tgmm_tiling[2] != 0:
                raise ValueError(
                    "Paired grouped matmul requires a full N TGMM tile"
                )

        if not self.use_kernel_native_weight_layout:
            return
        self.register_state_dict_post_hook(self._canonical_weights_on_save)
        self.register_load_state_dict_pre_hook(self._native_weights_on_load)

    def configure_dweight_output_dtype(self, reduce_dtype: torch.dtype) -> None:
        """Match unpaired expert dWeight stores to the FSDP reduction dtype."""
        if reduce_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                "Expert dWeight output supports only "
                f"float32 or bfloat16 reductions, got {reduce_dtype}"
            )
        self._unpaired_dweight_output_dtype = reduce_dtype

    def _init_param(self, name: str, param: torch.nn.Parameter) -> None:
        """Initialize native storage through a canonical contiguous tensor."""

        if not self.use_kernel_native_weight_layout:
            return super()._init_param(name, param)

        native_to_canonical = {
            native_name: canonical_name
            for canonical_name, native_name in _CANONICAL_TO_NATIVE_WEIGHT.items()
        }
        canonical_name = native_to_canonical[name]
        canonical = torch.empty_like(param).transpose(-2, -1).contiguous()
        Module._init_param(self, canonical_name, canonical)
        with torch.no_grad():
            param.copy_(canonical.transpose(-2, -1))

    def _distribute_states(self, parallel_dims) -> None:
        if (
            self.use_kernel_native_weight_layout
            and not self._native_sharding_applied
            and self._sharding_config is not None
        ):
            self._sharding_config = _native_weight_sharding(self._sharding_config)
            self._native_sharding_applied = True
        super()._distribute_states(parallel_dims)

    def parallelize(self, parallel_dims) -> None:
        # The paired GEMM can consume expert-major rows after EP dispatch, but
        # EP produces a ragged row count that is not generally tile-aligned.
        # Supporting it requires masked or padded tail rows, removing any
        # padding before combine, and validating the nested EP/FSDP sharding.
        if (
            self.use_pallas_paired_grouped_mm
            and parallel_dims.get_optional_mesh("ep") is not None
        ):
            raise ValueError("Paired grouped matmul requires EP=1 exact-row dispatch")
        super().parallelize(parallel_dims)

    @staticmethod
    def _canonical_weights_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Expose kernel-native parameters in canonical checkpoint layout."""

        for canonical_name, native_name in _CANONICAL_TO_NATIVE_WEIGHT.items():
            native = state_dict.pop(f"{prefix}{native_name}")
            state_dict[f"{prefix}{canonical_name}"] = native.transpose(
                -2, -1
            ).contiguous()

    @staticmethod
    def _native_weights_on_load(module, state_dict, prefix, *args) -> None:
        """Load canonical checkpoint tensors into kernel-native storage."""

        for canonical_name, native_name in _CANONICAL_TO_NATIVE_WEIGHT.items():
            canonical_key = f"{prefix}{canonical_name}"
            if canonical_key in state_dict:
                state_dict[f"{prefix}{native_name}"] = state_dict.pop(
                    canonical_key
                ).transpose(-2, -1).contiguous()

    def _native_experts_forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """``GroupedExperts._experts_forward`` with a configurable activation.

        Mirrors ``GroupedExperts._experts_forward`` in
        ``torchtitan/models/common/moe.py`` (keep in sync); upstream hard-codes
        silu.
        """
        if isinstance(self.w1_EFD, DTensor):
            w1_EFD = self.w1_EFD.to_local()
            w2_EDF = self.w2_EDF.to_local()
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD, w2_EDF, w3_EFD = self.w1_EFD, self.w2_EDF, self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if (
            get_spmd_backend() == "spmd_types"
            and spmd.is_type_checking()
            and spmd_mesh_size("ep") == 1
        ):
            for axis in ("dp", "cp"):
                spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

        gate_RF = torch._grouped_mm(
            x_RD.bfloat16(), w1_EFD.bfloat16().transpose(-2, -1), offs=offsets_E
        )
        up_RF = torch._grouped_mm(
            x_RD.bfloat16(), w3_EFD.bfloat16().transpose(-2, -1), offs=offsets_E
        )
        return torch._grouped_mm(
            self._activation_fn(gate_RF) * up_RF,
            w2_EDF.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        ).type_as(x_RD)

    def _activation_fn(self, gate_RF: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu_tanh":
            return F.gelu(gate_RF, approximate="tanh")
        return F.silu(gate_RF)

    def _experts_forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        # The Qwen TPU recipe uses Pallas because native torch._grouped_mm mode-2
        # dWeight is incorrect at M=4096. See
        # https://github.com/google-pytorch/torch_tpu/issues/2668.
        if not self.use_pallas_grouped_mm:
            return self._native_experts_forward(x_RD, num_tokens_per_expert_E)
        def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to_local() if isinstance(tensor, DTensor) else tensor

        if self.use_kernel_native_weight_layout:
            w1_EKN = local_tensor(self.w1_EDF)
            w2_EKN = local_tensor(self.w2_EFD)
            w3_EKN = local_tensor(self.w3_EDF)
        else:
            w1_EKN = local_tensor(self.w1_EFD).transpose(-2, -1)
            w2_EKN = local_tensor(self.w2_EDF).transpose(-2, -1)
            w3_EKN = local_tensor(self.w3_EFD).transpose(-2, -1)

        # Match GroupedExperts: expert compute is BF16 even when the surrounding
        # model is FP32, and the final result is restored to the caller's dtype.
        x_bf16 = x_RD.bfloat16()
        tile_m = self.pallas_grouped_mm_tile_m
        tile_k = self.pallas_grouped_mm_tile_k
        tile_n = self.pallas_grouped_mm_tile_n
        tgmm_tiling = self.pallas_grouped_mm_tgmm_tiling
        if self.use_pallas_paired_grouped_mm:
            # Paired W1/W3 always return BF16 dWeights: two full FP32 output
            # tiles exceed TPU scoped VMEM. FP32 FSDP reductions widen those
            # gradients after the kernel; BF16 reductions consume them as-is.
            gate_RF, up_RF = paired_segment_matmul_pallas_2d(
                x_bf16,
                w1_EKN.bfloat16(),
                w3_EKN.bfloat16(),
                num_tokens_per_expert_E,
                tile_m=tile_m,
                tgmm_tile_m=tgmm_tiling[0],
                tgmm_tile_k=tgmm_tiling[1],
            )
        else:
            gate_RF = segment_matmul_pallas_2d(
                x_bf16,
                w1_EKN.bfloat16(),
                num_tokens_per_expert_E,
                tile_m=tile_m,
                tile_k=tile_k,
                tile_n=tile_n,
                tgmm_tiling=tgmm_tiling,
                dweight_output_dtype=self._unpaired_dweight_output_dtype,
            )
            up_RF = segment_matmul_pallas_2d(
                x_bf16,
                w3_EKN.bfloat16(),
                num_tokens_per_expert_E,
                tile_m=tile_m,
                tile_k=tile_k,
                tile_n=tile_n,
                tgmm_tiling=tgmm_tiling,
                dweight_output_dtype=self._unpaired_dweight_output_dtype,
            )
        hidden_RF = self._activation_fn(gate_RF) * up_RF
        return segment_matmul_pallas_2d(
            hidden_RF,
            w2_EKN.bfloat16(),
            num_tokens_per_expert_E,
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
            tgmm_tiling=tgmm_tiling,
            dweight_output_dtype=self._unpaired_dweight_output_dtype,
        ).type_as(x_RD)


def configure_expert_dweight_output_dtype(
    model: torch.nn.Module, reduce_dtype: torch.dtype
) -> None:
    """Resolve unpaired dWeight stores before FSDP and graph tracing."""
    for module in model.modules():
        if isinstance(module, TpuGroupedExperts):
            module.configure_dweight_output_dtype(reduce_dtype)


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
        logger.info("Replaced %d GroupedExperts configuration(s) for TPU", len(matches))
        return model_config


__all__ = ["TpuGroupedExperts", "TpuGroupedExpertsConverter"]
