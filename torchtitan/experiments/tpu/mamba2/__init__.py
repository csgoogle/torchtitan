# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mamba-2 TPU model definitions, training specifications, and configs."""

from torchtitan.components import loss
from torchtitan.components import lr_scheduler
from torchtitan.components import optimizer
from torchtitan.components import validate
from torchtitan.hf_datasets import text_datasets
from torchtitan.protocols import train_spec

from . import config_registry
from . import model
from . import parallelize

GatedRMSNorm = model.GatedRMSNorm
Mamba2Block = model.Mamba2Block
Mamba2Config = model.Mamba2Config
Mamba2LM = model.Mamba2LM
Mamba2Mixer = model.Mamba2Mixer
PallasMamba2Block = model.PallasMamba2Block
PallasMamba2Config = model.PallasMamba2Config
PallasMamba2LM = model.PallasMamba2LM
PallasMamba2Mixer = model.PallasMamba2Mixer

mamba2_configs = config_registry.mamba2_configs
model_registry = config_registry.model_registry
mamba2_debug = config_registry.mamba2_debug
mamba2_130m = config_registry.mamba2_130m
mamba2_370m = config_registry.mamba2_370m
mamba2_780m = config_registry.mamba2_780m
mamba2_1_3b = config_registry.mamba2_1_3b
mamba2_2_7b = config_registry.mamba2_2_7b
mamba2_2_7b_256k = config_registry.mamba2_2_7b_256k

parallelize_mamba2 = parallelize.parallelize_mamba2

mamba2_tpu_configs = {
    "mamba2_debug": mamba2_debug(),
    "mamba2_130m": mamba2_130m(),
    "mamba2_370m": mamba2_370m(),
    "mamba2_780m": mamba2_780m(),
    "mamba2_1_3b": mamba2_1_3b(),
    "mamba2_2_7b": mamba2_2_7b(),
    "mamba2_2_7b_256k": mamba2_2_7b_256k(),
}

__all__ = [
    "GatedRMSNorm",
    "Mamba2Block",
    "Mamba2Config",
    "Mamba2LM",
    "Mamba2Mixer",
    "PallasMamba2Block",
    "PallasMamba2Config",
    "PallasMamba2LM",
    "PallasMamba2Mixer",
    "mamba2_configs",
    "model_registry",
    "mamba2_tpu_configs",
    "parallelize_mamba2",
    "mamba2_debug",
    "mamba2_130m",
    "mamba2_370m",
    "mamba2_780m",
    "mamba2_1_3b",
    "mamba2_2_7b",
    "mamba2_2_7b_256k",
]

_mamba2_train_spec = train_spec.TrainSpec(
    model_cls=Mamba2LM,
    model_args=mamba2_tpu_configs,
    parallelize_fn=parallelize_mamba2,
    pipelining_fn=None,
    loss_config=loss.CrossEntropyLoss.Config(),
    optimizer_config=optimizer.default_adamw(lr=8e-4),
    lr_scheduler_config=lr_scheduler.LRSchedulersContainer.Config(
        warmup_steps=1000
    ),
    dataloader_config=text_datasets.HuggingFaceTextDataLoader.Config(
        dataset="c4_test"
    ),
    validator_config=validate.Validator.Config(enable=True),
    state_dict_adapter=None,
)

train_spec.register_train_spec(name="mamba2", train_spec=_mamba2_train_spec)
train_spec.register_train_spec(
    name="mamba2_pallas", train_spec=_mamba2_train_spec
)
