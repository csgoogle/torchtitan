# Copyright 2026 The TorchTitan Authors. All Rights Reserved.

import dataclasses
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config.configs import ParallelismConfig

from torchtitan.components.lr_scheduler import LRSchedulersContainer

from torchtitan.experiments.tpu.rl.grpo_job_config import (
    GRPOConfig,
    GRPOJobConfig,
    ModelConfig,
    ReferenceConfig,
    SamplerConfig,
    GRPOTrainingConfig,
)
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.experiments.tpu.qwen3 import parallelize_qwen3, qwen3_configs


def grpo_qwen3_0_6b() -> GRPOJobConfig:
  """Qwen3 0.6B model GRPO training with FSDP on GLP."""
  return GRPOJobConfig(
      model_spec=ModelSpec(
          name="qwen3_tpu",
          flavor="0.6B",
          model=qwen3_configs["0.6B"],
          parallelize_fn=parallelize_qwen3,
          pipelining_fn=None,
          post_optimizer_build_fn=None,
          state_dict_adapter=None,
      ),
      model=ModelConfig(name="qwen3_tpu", flavor="0.6B"),
      training=GRPOTrainingConfig(
          dataset="random",
          seq_len=512,
          steps=8,
          local_batch_size=2,
      ),
      parallelism=ParallelismConfig(
          data_parallel_shard_degree=-1,
          data_parallel_replicate_degree=1,
          fsdp_reshard_after_forward="never",
      ),
      sampler=SamplerConfig(
          use_vllm=False,
          max_new_tokens=128,
          use_separate_sampler_model=False,
          use_fake_sampler=False,
          vllm_gpu_memory_utilization=0.4,
          vllm_max_model_len=512,
      ),
      lr_scheduler=LRSchedulersContainer.Config(warmup_steps=4),
      metrics=MetricsProcessor.Config(log_freq=1),
      grpo=GRPOConfig(group_size=4, grpo_beta=0.1),
  )


def grpo_qwen3_1_7b() -> GRPOJobConfig:
  """Qwen3 1.7B model GRPO training with FSDP on TPU v7x."""
  return GRPOJobConfig(
      model_spec=ModelSpec(
          name="qwen3_tpu",
          flavor="1.7B",
          model=qwen3_configs["1.7B"],
          parallelize_fn=parallelize_qwen3,
          pipelining_fn=None,
          post_optimizer_build_fn=None,
          state_dict_adapter=None,
      ),
      model=ModelConfig(name="qwen3_tpu", flavor="1.7B"),
      training=GRPOTrainingConfig(
          dataset="random",
          seq_len=1024,
          steps=8,
          local_batch_size=4,
      ),
      parallelism=ParallelismConfig(
          data_parallel_shard_degree=-1,
          data_parallel_replicate_degree=1,
          fsdp_reshard_after_forward="never",
      ),
      sampler=SamplerConfig(
          use_vllm=False,
          max_new_tokens=128,
          use_separate_sampler_model=False,
          use_fake_sampler=False,
          vllm_gpu_memory_utilization=0.45,
          vllm_max_model_len=1024,
      ),
      lr_scheduler=LRSchedulersContainer.Config(warmup_steps=4),
      metrics=MetricsProcessor.Config(log_freq=1),
      grpo=GRPOConfig(group_size=4, grpo_beta=0.1),
  )


def grpo_qwen3_4b() -> GRPOJobConfig:
  """Qwen3 4B model GRPO training with FSDP on TPU v7x."""
  return GRPOJobConfig(
      model_spec=ModelSpec(
          name="qwen3_tpu",
          flavor="4B",
          model=qwen3_configs["4B"],
          parallelize_fn=parallelize_qwen3,
          pipelining_fn=None,
          post_optimizer_build_fn=None,
          state_dict_adapter=None,
      ),
      model=ModelConfig(name="qwen3_tpu", flavor="4B"),
      training=GRPOTrainingConfig(
          dataset="random",
          seq_len=1024,
          steps=8,
          local_batch_size=4,
      ),
      parallelism=ParallelismConfig(
          data_parallel_shard_degree=-1,
          data_parallel_replicate_degree=1,
          fsdp_reshard_after_forward="never",
      ),
      sampler=SamplerConfig(
          use_vllm=False,
          max_new_tokens=128,
          use_separate_sampler_model=False,
          use_fake_sampler=False,
          vllm_gpu_memory_utilization=0.5,
          vllm_max_model_len=1024,
      ),
      lr_scheduler=LRSchedulersContainer.Config(warmup_steps=4),
      metrics=MetricsProcessor.Config(log_freq=1),
      grpo=GRPOConfig(group_size=4, grpo_beta=0.1),
  )


