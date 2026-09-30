# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry


def gemma4_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_spec),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def gemma4_moe_debug() -> Trainer.Config:
    model_spec = model_registry("debugmodel_moe")
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_spec),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
            export_dtype="bfloat16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def gemma4_31b() -> Trainer.Config:
    model_spec = model_registry("31B")
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_spec),
        ),
        hf_assets_path="./assets/hf/gemma-4-31B-it",
        metrics=MetricsProcessor.Config(log_freq=10),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="bfloat16",
        ),
        activation_checkpoint=FullAC.Config(),
    )


def gemma4_26b_a4b() -> Trainer.Config:
    model_spec = model_registry("26B-A4B")
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_spec),
        ),
        hf_assets_path="./assets/hf/gemma-4-26B-A4B-it",
        metrics=MetricsProcessor.Config(log_freq=10),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="bfloat16",
        ),
        activation_checkpoint=FullAC.Config(),
    )
