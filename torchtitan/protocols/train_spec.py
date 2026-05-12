"""[TO BE REMOVED] Compatibility layer for TPU model registration in Google3 experiments."""

from dataclasses import dataclass
from typing import Any, Callable, Literal, TypeAlias


class BaseModelArgs:
  """Placeholder base class for model args."""

  pass


ParallelizeFunction: TypeAlias = Callable[..., Any]


@dataclass
class MoEArgs:
  """Compatibility MoEArgs dataclass for TPU experiments."""

  num_experts: int = 1
  num_shared_experts: int = 0
  top_k: int = 1
  score_func: Literal["sigmoid", "softmax"] = "softmax"
  route_norm: bool = False
  route_scale: float = 1.0
  score_before_experts: bool = False
  _debug_force_load_balance: bool = False
  load_balance_coeff: float | None = None
  num_expert_groups: int | None = None
  num_limited_groups: int | None = None
  use_grouped_mm: bool = False


@dataclass
class TrainSpec:
  """Google-specific training specification for TPU experiments."""

  model_cls: type
  model_args: dict[str, Any]
  parallelize_fn: Callable
  pipelining_fn: Callable | None = None

  # Modern class-based configurations
  loss_config: Any | None = None
  optimizer_config: Any | None = None
  lr_scheduler_config: Any | None = None
  dataloader_config: Any | None = None
  tokenizer_config: Any | None = None
  validator_config: Any | None = None
  state_dict_adapter: type | None = None


_specs = {}


def register_train_spec(name: str, train_spec: TrainSpec):
  """Registers a TPU training specification."""
  _specs[name] = train_spec


def get_train_spec(name: str) -> TrainSpec:
  """Retrieves a registered TPU training specification."""
  return _specs[name]
