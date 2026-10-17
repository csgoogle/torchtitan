from dataclasses import dataclass, replace

import torch
import spmd_types as spmd
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.config import derive
from torchtitan.experiments.tpu.kernels.expert_segment_matmul import (
    segment_matmul_pallas_2d,
)
from torchtitan.experiments.tpu.kernels.paired_segment_matmul import (
    paired_segment_matmul_pallas_2d,
)
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.tools.logging import logger


_CANONICAL_TO_NATIVE_WEIGHT = {
    "w1_EFD": "w1_EDF",
    "w2_EDF": "w2_EFD",
    "w3_EFD": "w3_EDF",
}


def _transpose_last_two_dims(
    layout: spmd.SpmdType, *, ndim: int = 3
) -> spmd.SpmdType:
    """Transpose tensor-dimension references in an SPMD layout.

    Upstream d771abb0 deleted `torchtitan.protocols.sharding.SpmdLayout` in
    favour of `spmd_types.SpmdType`, renaming the per-axis field
    `axis_types` -> `local_type`.

    This reads the RAW fields rather than upstream's `_per_axis_types()`
    helper. That helper flattens `partition_spec` entries into `S(dim)`
    per-axis types, which is fine for inspection but cannot be fed back into
    `local_type`: `SpmdType.__post_init__` rejects a value that carries both
    `S(i)` entries and an explicit `PartitionSpec`. Permuting the raw fields
    independently keeps the two representations mutually exclusive, exactly as
    they arrived.
    """

    if ndim < 2:
        raise ValueError(f"Expected rank of at least 2, got {ndim}")

    permutation = list(range(ndim))
    permutation[-2], permutation[-1] = permutation[-1], permutation[-2]

    local_type = {}
    for axis, axis_type in layout.local_type.items():
        if isinstance(axis_type, spmd.Shard):
            if not 0 <= axis_type.dim < ndim:
                raise ValueError(
                    f"Tensor dimension {axis_type.dim} is invalid for rank {ndim}"
                )
            axis_type = spmd.S(permutation[axis_type.dim])
        local_type[axis] = axis_type

    partition_spec = layout.partition_spec
    if partition_spec is not None:
        if len(partition_spec) != ndim:
            raise ValueError(
                f"Expected rank-{ndim} partition spec, got {len(partition_spec)} dims"
            )
        partition_spec = tuple(partition_spec[dim] for dim in permutation)
    return replace(
        layout,
        local_type=local_type,
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
    W2's dWeight may tile differently from W1/W3's: a dWeight is tiled along
    ``K``, the weight's input dimension, which is ``F`` for W2 and ``D`` for
    W1/W3, so one K tile cannot suit both.
    The dWeight kernels accumulate in FP32 and store in the BF16 dtype of the
    weights they are given, whatever the param and reduce dtypes are.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        use_pallas_grouped_mm: bool = False
        # Share routed-token loads between the equal-shaped W1/W3 projections.
        use_pallas_paired_grouped_mm: bool = False
        pallas_grouped_mm_tile_m: int = 1024
        pallas_grouped_mm_tile_k: int = 0
        pallas_grouped_mm_tile_n: int = 0
        pallas_grouped_mm_tgmm_tiling: tuple[int, int, int] = (128, 128, 128)
        # TGMM tiling for the W2 (down-projection) dWeight. ``None`` follows
        # ``pallas_grouped_mm_tgmm_tiling``. Set it when W1/W3 want a K tile
        # that W2's shorter K axis cannot hold.
        pallas_grouped_mm_w2_tgmm_tiling: tuple[int, int, int] | None = None
        use_kernel_native_weight_layout: bool = False
        # Let `flatten_for_fsdp` store each `[E, K, N]` bank as `[E*K, N]` when
        # the FSDP degree exceeds E, so FSDP shards it contiguously on axis 0
        # instead of on an inner axis (which costs a relayout copy around every
        # gather and gradient reduce-scatter).
        flatten_expert_axis_for_fsdp: bool = False

    def __init__(self, config: Config):
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
            # `token_dispatcher` moved to the enclosing `RoutedExperts` in
            # upstream d771abb0; `GroupedExperts` is now dispatch-free.
        else:
            super().__init__(config)

        self.use_pallas_grouped_mm = config.use_pallas_grouped_mm
        self.use_pallas_paired_grouped_mm = config.use_pallas_paired_grouped_mm
        self.pallas_grouped_mm_tile_m = config.pallas_grouped_mm_tile_m
        self.pallas_grouped_mm_tile_k = config.pallas_grouped_mm_tile_k
        self.pallas_grouped_mm_tile_n = config.pallas_grouped_mm_tile_n
        self.pallas_grouped_mm_tgmm_tiling = config.pallas_grouped_mm_tgmm_tiling
        self.pallas_grouped_mm_w2_tgmm_tiling = config.pallas_grouped_mm_w2_tgmm_tiling
        self.use_kernel_native_weight_layout = config.use_kernel_native_weight_layout
        self._native_sharding_applied = False
        self.flatten_expert_axis_for_fsdp = config.flatten_expert_axis_for_fsdp
        # Parameter name -> its `[E, K, N]` shape, for banks stored flattened.
        self._flat_bank_shapes: dict[str, torch.Size] = {}
        if self.flatten_expert_axis_for_fsdp and not self.use_pallas_grouped_mm:
            raise ValueError("Flattened expert banks require Pallas grouped matmul")

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

    def flatten_for_fsdp(self, degree: int) -> list[str]:
        """Stores each expert bank ``[E, K, N]`` as ``[E*K, N]`` for ``degree``.

        Only when enabled, when ``degree`` is a multiple of E that E does not
        divide, and when each rank's slice of the merged axis stays inside one
        expert, so the gathered bank views back as ``[E, K, N]`` for free. Call
        before FSDP wraps the parameters. Returns the flattened names.
        """
        if not self.flatten_expert_axis_for_fsdp or self._flat_bank_shapes:
            return []
        num_experts = self.num_experts
        if num_experts % degree == 0:
            return []
        ranks_per_expert, rem = divmod(degree, num_experts)
        if rem:
            return []
        flattened = []
        for name, param in list(self._parameters.items()):
            if param is None or param.ndim != 3 or param.shape[0] != num_experts:
                continue
            if param.shape[1] % ranks_per_expert:
                continue
            self._flat_bank_shapes[name] = param.shape
            setattr(
                self,
                name,
                torch.nn.Parameter(
                    param.detach().reshape(-1, param.shape[-1]),
                    requires_grad=param.requires_grad,
                ),
            )
            flattened.append(name)
        return flattened

    def _bank(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Views a flattened ``[E*K, N]`` bank back as ``[E, K, N]``."""
        shape = self._flat_bank_shapes.get(name)
        return tensor if shape is None else tensor.view(shape)

    def _init_param(self, name: str, param: torch.nn.Parameter) -> None:
        """Initialize native storage through a canonical contiguous tensor."""

        if name in self._flat_bank_shapes:
            # The initializers are elementwise iid, so a flattened bank takes
            # the canonical weight's initializer directly. Values differ from
            # the unflattened draw (different RNG order), not the distribution.
            canonical_name = {
                native: canonical
                for canonical, native in _CANONICAL_TO_NATIVE_WEIGHT.items()
            }.get(name, name)
            return Module._init_param(self, canonical_name, param)

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
            native = module._bank(native_name, state_dict.pop(f"{prefix}{native_name}"))
            state_dict[f"{prefix}{canonical_name}"] = native.transpose(
                -2, -1
            ).contiguous()

    @staticmethod
    def _native_weights_on_load(module, state_dict, prefix, *args) -> None:
        """Load canonical checkpoint tensors into kernel-native storage."""

        for canonical_name, native_name in _CANONICAL_TO_NATIVE_WEIGHT.items():
            canonical_key = f"{prefix}{canonical_name}"
            if canonical_key in state_dict:
                native = state_dict.pop(canonical_key).transpose(-2, -1).contiguous()
                if native_name in module._flat_bank_shapes:
                    native = native.reshape(-1, native.shape[-1])
                state_dict[f"{prefix}{native_name}"] = native

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        # The Qwen TPU recipe uses Pallas because native torch._grouped_mm mode-2
        # dWeight is incorrect at M=4096. See
        # https://github.com/google-pytorch/torch_tpu/issues/2668.
        if not self.use_pallas_grouped_mm:
            return super().forward(x_RD, num_tokens_per_expert_E)
        def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to_local() if isinstance(tensor, DTensor) else tensor

        if self.use_kernel_native_weight_layout:
            w1_EKN = self._bank("w1_EDF", local_tensor(self.w1_EDF))
            w2_EKN = self._bank("w2_EFD", local_tensor(self.w2_EFD))
            w3_EKN = self._bank("w3_EDF", local_tensor(self.w3_EDF))
        else:
            w1_EKN = self._bank("w1_EFD", local_tensor(self.w1_EFD)).transpose(-2, -1)
            w2_EKN = self._bank("w2_EDF", local_tensor(self.w2_EDF)).transpose(-2, -1)
            w3_EKN = self._bank("w3_EFD", local_tensor(self.w3_EFD)).transpose(-2, -1)

        # Match GroupedExperts: expert compute is BF16 even when the surrounding
        # model is FP32, and the final result is restored to the caller's dtype.
        x_bf16 = x_RD.bfloat16()
        tile_m = self.pallas_grouped_mm_tile_m
        # The kernels need a row count divisible by tile_m. Append zero rows to
        # the last expert's group: they add nothing to dWeight and are sliced
        # off the output. A no-op for tile-aligned batches.
        num_rows = x_bf16.shape[0]
        pad_rows = -num_rows % tile_m
        if pad_rows:
            x_bf16 = F.pad(x_bf16, (0, 0, 0, pad_rows))
            num_tokens_per_expert_E = torch.cat(
                [num_tokens_per_expert_E[:-1], num_tokens_per_expert_E[-1:] + pad_rows]
            )
        tile_k = self.pallas_grouped_mm_tile_k
        tile_n = self.pallas_grouped_mm_tile_n
        tgmm_tiling = self.pallas_grouped_mm_tgmm_tiling
        w2_tgmm_tiling = self.pallas_grouped_mm_w2_tgmm_tiling or tgmm_tiling
        if self.use_pallas_paired_grouped_mm:
            # Every expert kernel gets BF16 weights and returns BF16 dWeights.
            # FP32 FSDP reductions widen those gradients after the kernel; BF16
            # reductions consume them as-is.
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
            )
            up_RF = segment_matmul_pallas_2d(
                x_bf16,
                w3_EKN.bfloat16(),
                num_tokens_per_expert_E,
                tile_m=tile_m,
                tile_k=tile_k,
                tile_n=tile_n,
                tgmm_tiling=tgmm_tiling,
            )
        hidden_RF = F.silu(gate_RF) * up_RF
        return segment_matmul_pallas_2d(
            hidden_RF,
            w2_EKN.bfloat16(),
            num_tokens_per_expert_E,
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
            tgmm_tiling=w2_tgmm_tiling,
        )[:num_rows].type_as(x_RD)


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
