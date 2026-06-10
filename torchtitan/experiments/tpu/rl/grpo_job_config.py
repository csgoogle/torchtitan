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
  vllm_gpu_memory_utilization: float = 0.4
  vllm_max_model_len: int = 1024
  # TODO: Support tensor_parallel_size > 1 for vLLM sampler.
  # Currently TPU-vLLM requires TP=1 in collocated setups to avoid crashing workers.
  vllm_tensor_parallel_size: int = 1
  # Set enforce_eager to True to avoid compiling for different sequence lengths 
  # and get to the sampling stage much sooner.
  vllm_enforce_eager: bool = True


@dataclasses.dataclass
class ReferenceConfig:
  use_reference_model: bool = True
  distributed_strategy: str = "fsdp"


@dataclasses.dataclass
class GRPOConfig:
  global_prompt_batch_size: int = 32 # Total prompts generated across the cluster per step
  group_size: int = 4 # Number of rollouts (completions) generated per prompt
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
class GRPOJobConfig(torchtitan.experiments.tpu.tpu_job_config.TPUTrainerConfig):
  grpo: GRPOConfig = dataclasses.field(default_factory=GRPOConfig)
  sampler: SamplerConfig = dataclasses.field(default_factory=SamplerConfig)
  reference: ReferenceConfig = dataclasses.field(
      default_factory=ReferenceConfig
  )
  model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
  training: GRPOTrainingConfig = dataclasses.field(
      default_factory=GRPOTrainingConfig
  )

