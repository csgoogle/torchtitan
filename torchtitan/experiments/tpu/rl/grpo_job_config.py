# Copyright 2026 The TorchTitan Authors. All Rights Reserved.

import dataclasses
from torchtitan.config.configs import TrainingConfig
import torchtitan.experiments.tpu.tpu_job_config


@dataclasses.dataclass

class SamplerConfig:
  max_new_tokens: int = 20
  temperature: float = 1.0
  top_k: int = 0
  use_vllm: bool = False  # vllm is not supported at this moment
  use_separate_sampler_model: bool = False
  distributed_strategy: str = "fsdp"
  use_fake_sampler: bool = False


@dataclasses.dataclass
class ReferenceConfig:
  use_reference_model: bool = True
  distributed_strategy: str = "fsdp"


@dataclasses.dataclass
class GRPOConfig:
  group_size: int = 4
  grpo_beta: float = 0.1
  ppo_clip_eps: float = 0.2
  ppo_epochs: int = 4


@dataclasses.dataclass
class ModelConfig:
  name: str = "qwen3_tpu"
  flavor: str = "4B"


@dataclasses.dataclass(kw_only=True, slots=True)
class GRPOTrainingConfig(TrainingConfig):
  dataset: str = "random"


@dataclasses.dataclass(kw_only=True, slots=True)
class GRPOJobConfig(torchtitan.experiments.tpu.tpu_job_config.TPUJobConfig):
  grpo: GRPOConfig = dataclasses.field(default_factory=GRPOConfig)
  sampler: SamplerConfig = dataclasses.field(default_factory=SamplerConfig)
  reference: ReferenceConfig = dataclasses.field(
      default_factory=ReferenceConfig
  )
  model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
  training: GRPOTrainingConfig = dataclasses.field(
      default_factory=GRPOTrainingConfig
  )

