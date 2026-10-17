"""Ulysses context parallelism for the TPU lane.

Each attention layer all-to-alls q/k/v from token-sharded to head-sharded,
runs full-sequence attention on ``H / cp`` heads (by default the wrapped inner
attention; on TPU usually the Pallas splash kernel, installed with
``set_local_attention``), then all-to-alls the output back.

Tensor suffixes: ``T`` tokens, ``H`` heads, ``K`` qk head dim, ``V`` v head dim.
"""

from collections.abc import Callable
from dataclasses import dataclass, fields
import sys
from typing import Any

# pyformat: disable
# Import torch before spmd_types to avoid circular import with torch_tpu.
import torch
import spmd_types as spmd
# pyformat: enable
import torch.distributed._functional_collectives as funcol
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.attention import (
    BaseAttention,
    FlexInnerAttention,
    ScaledDotProductInnerAttention,
    VarlenInnerAttention,
    segment_ids_from_positions,
)
from torchtitan.models.common.cp_attention import CPInnerAttention

__all__ = [
    "UlyssesCPInnerAttention",
    "UlyssesCPScaledDotProductInnerAttention",
    "UlyssesCPFlexInnerAttention",
    "UlyssesCPVarlenInnerAttention",
    "configure_tpu_context_parallel",
    "install_tpu_cp_patches",
]

_TOKEN_DIM = 0
_HEAD_DIM = 1


def _type_checking() -> bool:
  return get_spmd_backend() == "spmd_types" and spmd.is_type_checking()


def _tracing() -> bool:
  # spmd.redistribute is an autograd.Function around dist.* collectives,
  # which Dynamo skips; use traceable functional collectives instead.
  return torch.compiler.is_compiling() or torch.compiler._is_non_strict_tracing()


def _all_to_all(
    x: torch.Tensor, group: Any, *, src_dim: int, dst_dim: int
) -> torch.Tensor:
  """Reshards ``x`` from ``S(src_dim)`` to ``S(dst_dim)`` over ``group``."""
  if not _tracing():
    return spmd.redistribute(
        x, group, src=spmd.S(src_dim), dst=spmd.S(dst_dim)
    )
  n = group.size()
  shape = list(x.shape)
  if shape[dst_dim] % n:
    raise ValueError(
        f"dim {dst_dim} (size {shape[dst_dim]}) is not divisible by cp={n}."
    )
  # [.., D, ..] -> [n, .., D / n, ..]: chunk i goes to rank i.
  chunked = shape[:dst_dim] + [n, shape[dst_dim] // n] + shape[dst_dim + 1 :]
  x = x.reshape(chunked).movedim(dst_dim, 0).contiguous()
  x = funcol.wait_tensor(funcol.all_to_all_single(x, None, None, group))
  # [n, ..] -> concatenate the received chunks along src_dim.
  out_shape = list(shape)
  out_shape[dst_dim] //= n
  out_shape[src_dim] *= n
  return x.movedim(0, src_dim).reshape(out_shape)


def _repeat_kv_heads(x: torch.Tensor, cp: int) -> torch.Tensor:
  """Repeats KV heads so that each of the ``cp`` head shards gets one.

  Needed when ``n_kv_heads < cp``, e.g. Qwen3.5-397B (2 KV heads) at cp=4.
  The result keeps the token sharding of ``x``.
  """
  n_kv = x.shape[_HEAD_DIM]
  if n_kv >= cp:
    return x
  if cp % n_kv:
    raise ValueError(f"cp ({cp}) must be a multiple of n_kv_heads ({n_kv}).")
  with spmd.no_typecheck():
    out = x.repeat_interleave(cp // n_kv, dim=_HEAD_DIM)
  if _type_checking():
    spmd.assert_type(out, spmd.get_local_type(x), spmd.get_partition_spec(x))
  return out


@spmd.no_typecheck()
def _full_sequence_segment_ids(
    positions: torch.Tensor, group: Any
) -> torch.Tensor:
  """Packed-document ``segment_ids`` of the whole (unsharded) sequence."""
  if _tracing():
    positions = funcol.wait_tensor(
        funcol.all_gather_tensor(positions, _TOKEN_DIM, group)
    )
  else:
    positions = spmd.redistribute(
        positions, group, src=spmd.S(_TOKEN_DIM), dst=spmd.R
    )
  return segment_ids_from_positions(positions)


class UlyssesCPInnerAttention(CPInnerAttention):
  """Ulysses (all-to-all) CP mixin for any single-device inner attention."""

  @classmethod
  def cp_shard(
      cls,
      input_dict: dict[str, Any],
      input_shardings: dict[str, Any] | None,
      cp_mesh: Any,
      load_balancer_type: str | None,
      ptrr_mask_key: str | None,
  ) -> dict[str, Any]:
    from torchtitan.distributed.context_parallel.api import prepare_context_parallel_input  # pylint: disable=g-import-not-at-top

    # Attention sees the full sequence, so masks stay unsharded, and there is
    # no per-rank causal imbalance to load-balance.
    attention_masks = input_dict.pop("attention_masks", None)
    prepare_context_parallel_input(
        input_dict, input_shardings, cp_mesh, None, None
    )
    if attention_masks is not None:
      input_dict["attention_masks"] = attention_masks
    return input_dict

  def set_local_attention(
      self, local_attn: Callable[..., torch.Tensor] | None
  ) -> None:
    """Replaces the full-sequence attention, e.g. with TPU splash attention."""
    object.__setattr__(self, "_local_attn_override", local_attn)

  def local_forward(self, q_THK, k_THK, v_THV, **kwargs) -> torch.Tensor:
    """The wrapped single-device inner attention."""
    return super().forward(  # pyrefly: ignore[missing-attribute]
        q_THK, k_THK, v_THV, **kwargs
    )

  def forward(
      self,
      q_THK: torch.Tensor,
      k_THK: torch.Tensor,
      v_THV: torch.Tensor,
      **kwargs,
  ) -> torch.Tensor:
    group = spmd_mesh_group(MeshAxisName.CP)
    if group is None:
      raise RuntimeError("CP attention requires an active CP mesh axis.")
    cp = group.size()
    k_THK, v_THV = _repeat_kv_heads(k_THK, cp), _repeat_kv_heads(v_THV, cp)
    # (T/cp, H, *) -> (T, H/cp, *).
    q_THK, k_THK, v_THV = (
        _all_to_all(x, group, src_dim=_TOKEN_DIM, dst_dim=_HEAD_DIM)
        for x in (q_THK, k_THK, v_THV)
    )
    if "enable_gqa" in kwargs:
      kwargs["enable_gqa"] = q_THK.shape[_HEAD_DIM] != k_THK.shape[_HEAD_DIM]
    # Decoder.forward derived segment_ids from the token-sharded positions;
    # the local attention needs them for the full sequence.
    positions = kwargs.pop("positions", None)
    if positions is not None:
      kwargs["segment_ids"] = _full_sequence_segment_ids(positions, group)

    local_attn = getattr(self, "_local_attn_override", None) or (
        self.local_forward
    )
    type_check = _type_checking()
    if type_check:
      q_type = spmd.get_local_type(q_THK), spmd.get_partition_spec(q_THK)
    # Kernels below (SDPA shim, Pallas) carry no spmd_types annotations.
    with spmd.no_typecheck():
      out_THV = local_attn(q_THK, k_THK, v_THV, **kwargs)
    if type_check and not spmd.has_local_type(out_THV):
      spmd.assert_type(out_THV, *q_type)
    # (T, H/cp, V) -> (T/cp, H, V).
    return _all_to_all(out_THV, group, src_dim=_HEAD_DIM, dst_dim=_TOKEN_DIM)


class UlyssesCPScaledDotProductInnerAttention(
    UlyssesCPInnerAttention, ScaledDotProductInnerAttention
):

  @dataclass(kw_only=True, slots=True)
  class Config(ScaledDotProductInnerAttention.Config):
    pass


class UlyssesCPFlexInnerAttention(UlyssesCPInnerAttention, FlexInnerAttention):

  @dataclass(kw_only=True, slots=True)
  class Config(FlexInnerAttention.Config):
    pass


class UlyssesCPVarlenInnerAttention(
    UlyssesCPInnerAttention, VarlenInnerAttention
):

  @dataclass(kw_only=True, slots=True)
  class Config(VarlenInnerAttention.Config):
    pass


# (single-device config, Ulysses CP config)
_CP_CONFIGS = (
    (
        ScaledDotProductInnerAttention.Config,
        UlyssesCPScaledDotProductInnerAttention.Config,
    ),
    (FlexInnerAttention.Config, UlyssesCPFlexInnerAttention.Config),
    (VarlenInnerAttention.Config, UlyssesCPVarlenInnerAttention.Config),
)


def _convert(cfg: Any, target: type[Any]) -> Any:
  return target(**{f.name: getattr(cfg, f.name) for f in fields(target)})


def _validate_heads(fqn: str, attn: Any, tp: int, cp: int) -> None:
  n_heads = attn.n_heads
  n_kv = getattr(attn, "n_kv_heads", None) or n_heads
  if n_heads % (tp * cp):
    raise ValueError(
        f"{fqn}: n_heads ({n_heads}) must be divisible by tp * cp ({tp * cp})."
    )
  # KV heads are TP-sharded, then repeated up to cp if there are fewer.
  local_kv = n_kv // tp
  if n_kv % tp or (local_kv % cp and cp % local_kv):
    raise ValueError(
        f"{fqn}: n_kv_heads ({n_kv}) / tp ({tp}) must divide or be divisible"
        f" by cp ({cp})."
    )


def configure_tpu_context_parallel(parallelism: Any, model: Any) -> None:
  """Switches the attention of ``model`` to Ulysses CP iff ``cp > 1``.

  For ``cp > 1`` this also selects the spmd_types backend (required for CP)
  and disables the load balancer (Ulysses attends over the full sequence).
  For ``cp == 1`` it reverts a previously converted config.
  """
  cp = parallelism.context_parallel_degree
  if cp > 1:
    install_tpu_cp_patches()
    parallelism.spmd_backend = "spmd_types"
    parallelism.context_parallel_load_balancer = None
  if model is None:
    return
  for fqn, attn, _, _ in model.traverse(BaseAttention.Config):
    inner = attn.inner_attention
    for base, ulysses in _CP_CONFIGS:
      if cp > 1 and type(inner) is base:
        attn.inner_attention = _convert(inner, ulysses)
      elif cp == 1 and type(inner) is ulysses:
        attn.inner_attention = _convert(inner, base)
    if cp > 1:
      _validate_heads(fqn, attn, parallelism.tensor_parallel_degree, cp)


_patched = False


def install_tpu_cp_patches() -> None:
  """Lets FSDP2 shard over ``(dp_shard, cp)`` on torch without spmd_types FSDP.

  Released torch (e.g. 2.13.0) has no ``FSDPParam._resolve_spmd_types_for_
  storage`` and rejects the multi-axis ``DataParallelMeshDims`` that
  ``resolve_fsdp_mesh`` returns when CP is on; give it the flattened
  ``dp_shard_cp`` mesh instead. No-op on torch versions that have it.
  """
  global _patched
  if _patched:
    return
  _patched = True
  from torch.distributed.device_mesh import DeviceMesh  # pylint: disable=g-import-not-at-top
  from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam  # pylint: disable=g-import-not-at-top
  import torchtitan.distributed.fsdp as fsdp_mod  # pylint: disable=g-import-not-at-top

  if hasattr(FSDPParam, "_resolve_spmd_types_for_storage"):
    return
  resolve = fsdp_mod.resolve_fsdp_mesh

  def resolve_flattened(parallel_dims):
    mesh, dims = resolve(parallel_dims)
    if (
        dims is None
        or not parallel_dims.cp_enabled
        or parallel_dims.tp_enabled
        or parallel_dims.ep_enabled
    ):
      return mesh, dims
    shard = mesh["dp_shard", "cp"]._flatten("dp_shard_cp")
    if parallel_dims.dp_replicate_enabled:
      shard = DeviceMesh._concatenate([mesh["dp_replicate"], shard])
    return shard, None

  # Model parallelize modules bind the function at import time.
  for name, mod in list(sys.modules.items()):
    if name.startswith("torchtitan.") and vars(mod).get(
        "resolve_fsdp_mesh"
    ) is resolve:
      mod.resolve_fsdp_mesh = resolve_flattened
