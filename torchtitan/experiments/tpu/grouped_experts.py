from dataclasses import dataclass, replace

import spmd_types as spmd
import torch
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
from torchtitan.protocols.sharding import ShardingConfig, SpmdLayout


_CANONICAL_TO_NATIVE_WEIGHT = {
    "w1_EFD": "w1_EDF",
    "w2_EDF": "w2_EFD",
    "w3_EFD": "w3_EDF",
}


def _transpose_last_two_dims(layout: SpmdLayout, *, ndim: int = 3) -> SpmdLayout:
  """Transpose tensor-dimension references in an SPMD layout."""

  if ndim < 2:
    raise ValueError(f"Expected rank of at least 2, got {ndim}")

  def transpose_dim(dim: int) -> int:
    if not 0 <= dim < ndim:
      raise ValueError(f"Tensor dimension {dim} is invalid for rank {ndim}")
    if dim == ndim - 2:
      return ndim - 1
    if dim == ndim - 1:
      return ndim - 2
    return dim

  def transpose_axis_type(axis_type):
    if not isinstance(axis_type, spmd.Shard):
      return axis_type
    return spmd.S(transpose_dim(axis_type.dim))

  partition_spec = layout.partition_spec
  if partition_spec is not None:
    if len(partition_spec) != ndim:
      raise ValueError(
          f"Expected rank-{ndim} partition spec, got {len(partition_spec)} dims"
      )
    partition_spec = list(partition_spec)
    partition_spec[-2], partition_spec[-1] = (
        partition_spec[-1],
        partition_spec[-2],
    )
    partition_spec = tuple(partition_spec)
  return SpmdLayout(
      {
          axis: transpose_axis_type(axis_type)
          for axis, axis_type in layout.axis_types.items()
      },
      partition_spec=partition_spec,
  )


def _packed_gate_up_layout(layout: SpmdLayout) -> SpmdLayout:
  """Map ``(E, F, D)`` placements to packed ``(E, 2, D, F)`` storage."""

  def remap_axis_type(axis_type):
    if not isinstance(axis_type, spmd.Shard):
      return axis_type
    return spmd.S({0: 0, 1: 3, 2: 2}[axis_type.dim])

  partition_spec = layout.partition_spec
  if partition_spec is not None:
    if len(partition_spec) != 3:
      raise ValueError("Canonical gate/up partition spec must have three dims")
    partition_spec = (
        partition_spec[0],
        None,
        partition_spec[2],
        partition_spec[1],
    )
  return SpmdLayout(
      {
          axis: remap_axis_type(axis_type)
          for axis, axis_type in layout.axis_types.items()
      },
      partition_spec=partition_spec,
  )


def _native_weight_sharding(
    config: ShardingConfig, *, pack_gate_up: bool = False
) -> ShardingConfig:
  """Map canonical expert state names/layouts to kernel-native storage."""

  state_shardings = dict(config.state_shardings)
  if pack_gate_up:
    w1_layout = state_shardings.pop("w1_EFD")
    w3_layout = state_shardings.pop("w3_EFD")
    if w1_layout != w3_layout:
      raise ValueError("Packed gate/up weights require identical sharding")
    state_shardings["w13_E2DF"] = _packed_gate_up_layout(w1_layout)
    state_shardings["w2_EFD"] = _transpose_last_two_dims(
        state_shardings.pop("w2_EDF")
    )
    return replace(config, state_shardings=state_shardings)

  for canonical_name, native_name in _CANONICAL_TO_NATIVE_WEIGHT.items():
    state_shardings[native_name] = _transpose_last_two_dims(
        state_shardings.pop(canonical_name)
    )
  return replace(config, state_shardings=state_shardings)


class TpuGroupedExperts(GroupedExperts):
  """TPU expert kernels with optional native and paired Pallas paths.

  The optional kernel-native layout stores W2 in the ``(E, K, N)`` order
  consumed by Pallas. The paired path packs W1 and W3 into one ``(E, 2, K, N)``
  parameter so FSDP issues one collective for both projections. State-dict
  hooks retain the canonical Qwen ``w1_EFD`` / ``w2_EDF`` / ``w3_EFD``
  checkpoint contract.

  Translation hooks register during construction. Later save hooks on this
  module see canonical keys; later load pre-hooks see native keys.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(GroupedExperts.Config):
    use_pallas_grouped_mm: bool = False
    use_pallas_paired_grouped_mm: bool = False
    emit_bf16_rounded_fp32_paired_dweight: bool = False
    pallas_grouped_mm_tile_m: int = 1024
    # Full K-by-N TGMM tiles use proportionally larger FP32 VMEM scratch, so
    # recipes opt in only after validating their expert dimensions.
    pallas_grouped_mm_use_full_tgmm_tiling: bool = False
    use_kernel_native_weight_layout: bool = False

  def __init__(self, config: Config):
    if config.use_pallas_paired_grouped_mm and not config.use_pallas_grouped_mm:
      raise ValueError("Paired grouped matmul requires Pallas grouped matmul")
    if (
        config.emit_bf16_rounded_fp32_paired_dweight
        and not config.use_pallas_paired_grouped_mm
    ):
      raise ValueError(
          "BF16-rounded FP32 dWeight requires paired grouped matmul"
      )
    if (
        config.use_pallas_paired_grouped_mm
        and not config.use_kernel_native_weight_layout
    ):
      raise ValueError("Paired grouped matmul requires kernel-native weights")
    if config.use_kernel_native_weight_layout:
      if not config.use_pallas_grouped_mm:
        raise ValueError(
            "Kernel-native expert weights require Pallas grouped matmul"
      )
      Module.__init__(self)
      self.num_experts = config.num_experts
      if config.use_pallas_paired_grouped_mm:
        self.w13_E2DF = torch.nn.Parameter(
            torch.empty(config.num_experts, 2, config.dim, config.hidden_dim)
        )
      else:
        self.w1_EDF = torch.nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
      self.w2_EFD = torch.nn.Parameter(
          torch.empty(config.num_experts, config.hidden_dim, config.dim)
      )
      if not config.use_pallas_paired_grouped_mm:
        self.w3_EDF = torch.nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
      self.token_dispatcher = config.token_dispatcher.build()
    else:
      super().__init__(config)

    self.use_pallas_grouped_mm = config.use_pallas_grouped_mm
    self.use_pallas_paired_grouped_mm = config.use_pallas_paired_grouped_mm
    self.emit_bf16_rounded_fp32_paired_dweight = (
        config.emit_bf16_rounded_fp32_paired_dweight
    )
    self.pallas_grouped_mm_tile_m = config.pallas_grouped_mm_tile_m
    self.pallas_grouped_mm_use_full_tgmm_tiling = (
        config.pallas_grouped_mm_use_full_tgmm_tiling
    )
    self.use_kernel_native_weight_layout = config.use_kernel_native_weight_layout
    self._native_sharding_applied = False

    if not self.use_kernel_native_weight_layout:
      return
    self.register_state_dict_post_hook(self._canonical_weights_on_save)
    self.register_load_state_dict_pre_hook(self._native_weights_on_load)

  def _init_param(self, name: str, param: torch.nn.Parameter) -> None:
    """Initialize native storage through a canonical contiguous tensor."""

    if not self.use_kernel_native_weight_layout:
      return super()._init_param(name, param)

    if self.use_pallas_paired_grouped_mm and name == "w13_E2DF":
      canonical = torch.empty_like(param[:, 0]).transpose(-2, -1).contiguous()
      Module._init_param(self, "w1_EFD", canonical)
      with torch.no_grad():
        param[:, 0].copy_(canonical.transpose(-2, -1))
      return

    native_to_canonical = {
        native_name: canonical_name
        for canonical_name, native_name in _CANONICAL_TO_NATIVE_WEIGHT.items()
    }
    canonical_name = native_to_canonical[name]
    canonical = torch.empty_like(param).transpose(-2, -1).contiguous()
    Module._init_param(self, canonical_name, canonical)
    with torch.no_grad():
      param.copy_(canonical.transpose(-2, -1))
      if self.use_pallas_paired_grouped_mm and name == "w2_EFD":
        # Keep the canonical W1, W2, W3 initializer RNG order.
        w3_canonical = (
            torch.empty_like(self.w13_E2DF[:, 1])
            .transpose(-2, -1)
            .contiguous()
        )
        Module._init_param(self, "w3_EFD", w3_canonical)
        self.w13_E2DF[:, 1].copy_(w3_canonical.transpose(-2, -1))

  def _distribute_states(self, parallel_dims) -> None:
    if (
        self.use_kernel_native_weight_layout
        and not self._native_sharding_applied
        and self._sharding_config is not None
    ):
      self._sharding_config = _native_weight_sharding(
          self._sharding_config,
          pack_gate_up=self.use_pallas_paired_grouped_mm,
      )
      self._native_sharding_applied = True
    super()._distribute_states(parallel_dims)

  @staticmethod
  def _canonical_weights_on_save(module, state_dict, prefix, local_metadata) -> None:
    """Expose kernel-native parameters in canonical checkpoint layout."""

    if module.use_pallas_paired_grouped_mm:
      packed = state_dict.pop(f"{prefix}w13_E2DF")
      w2 = state_dict.pop(f"{prefix}w2_EFD")
      state_dict[f"{prefix}w1_EFD"] = packed[:, 0].transpose(
          -2, -1
      ).contiguous()
      state_dict[f"{prefix}w2_EDF"] = w2.transpose(-2, -1).contiguous()
      state_dict[f"{prefix}w3_EFD"] = packed[:, 1].transpose(
          -2, -1
      ).contiguous()
      return

    for canonical_name, native_name in _CANONICAL_TO_NATIVE_WEIGHT.items():
      native = state_dict.pop(f"{prefix}{native_name}")
      state_dict[f"{prefix}{canonical_name}"] = native.transpose(
          -2, -1
      ).contiguous()

  @staticmethod
  def _native_weights_on_load(module, state_dict, prefix, *args) -> None:
    """Load canonical checkpoint tensors into kernel-native storage."""

    if module.use_pallas_paired_grouped_mm:
      w1_key = f"{prefix}w1_EFD"
      w3_key = f"{prefix}w3_EFD"
      if w1_key in state_dict and w3_key in state_dict:
        state_dict[f"{prefix}w13_E2DF"] = torch.stack(
            (
                state_dict.pop(w1_key).transpose(-2, -1),
                state_dict.pop(w3_key).transpose(-2, -1),
            ),
            dim=1,
        ).contiguous()
      w2_key = f"{prefix}w2_EDF"
      if w2_key in state_dict:
        state_dict[f"{prefix}w2_EFD"] = state_dict.pop(w2_key).transpose(
            -2, -1
        ).contiguous()
      return

    for canonical_name, native_name in _CANONICAL_TO_NATIVE_WEIGHT.items():
      canonical_key = f"{prefix}{canonical_name}"
      if canonical_key in state_dict:
        state_dict[f"{prefix}{native_name}"] = state_dict.pop(
            canonical_key
        ).transpose(-2, -1).contiguous()

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

    def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
      return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    if self.use_pallas_paired_grouped_mm:
      w13_E2KN = local_tensor(self.w13_E2DF)
      w2_EKN = local_tensor(self.w2_EFD)
    elif self.use_kernel_native_weight_layout:
      w1_EKN = local_tensor(self.w1_EDF)
      w2_EKN = local_tensor(self.w2_EFD)
      w3_EKN = local_tensor(self.w3_EDF)
    else:
      w1_EKN = local_tensor(self.w1_EFD).transpose(-2, -1)
      w2_EKN = local_tensor(self.w2_EDF).transpose(-2, -1)
      w3_EKN = local_tensor(self.w3_EFD).transpose(-2, -1)

    x_bf16 = x_RD.bfloat16()
    tile_m = self.pallas_grouped_mm_tile_m
    use_full_tgmm_tiling = self.pallas_grouped_mm_use_full_tgmm_tiling
    if self.use_pallas_paired_grouped_mm:
      gate_RF, up_RF = paired_segment_matmul_pallas_2d(
          x_bf16,
          w13_E2KN.bfloat16(),
          num_tokens_per_expert_E,
          tile_m=tile_m,
          emit_bf16_rounded_fp32_dweight=(
              self.emit_bf16_rounded_fp32_paired_dweight
          ),
      )
    else:
      gate_RF = segment_matmul_pallas_2d(
          x_bf16,
          w1_EKN.bfloat16(),
          num_tokens_per_expert_E,
          tile_m=tile_m,
          use_full_tgmm_tiling=use_full_tgmm_tiling,
      )
      up_RF = segment_matmul_pallas_2d(
          x_bf16,
          w3_EKN.bfloat16(),
          num_tokens_per_expert_E,
          tile_m=tile_m,
          use_full_tgmm_tiling=use_full_tgmm_tiling,
      )
    hidden_RF = F.silu(gate_RF) * up_RF
    return segment_matmul_pallas_2d(
        hidden_RF,
        w2_EKN.bfloat16(),
        num_tokens_per_expert_E,
        tile_m=tile_m,
        use_full_tgmm_tiling=use_full_tgmm_tiling,
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
