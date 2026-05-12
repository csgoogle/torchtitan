# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import MSELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

import torchtitan.models.flux as native_flux_models
from torchtitan.models.flux.flux_datasets import FluxDataLoader
from torchtitan.models.flux.validate import FluxValidator
from torchtitan.experiments.tpu.flux.infra.parallelize import parallelize_flux
from torchtitan.models.flux.model.model import FluxModel
from torchtitan.models.flux.model.state_dict_adapter import FluxStateDictAdapter

__all__ = [
    "flux_configs",
    "parallelize_flux",
]

flux_configs = {
    "flux-dev": native_flux_models.flux_configs["flux-dev"](),
    "flux-schnell": native_flux_models.flux_configs["flux-schnell"](),
    "flux-debug": native_flux_models.flux_configs["flux-debug"](),
}


# pytype: disable=wrong-arg-types
register_train_spec(
    name="flux_tpu",
    train_spec=TrainSpec(
        model_cls=FluxModel,
        model_args=flux_configs,
        parallelize_fn=parallelize_flux,
        pipelining_fn=None,
        loss_config=MSELoss.Config(),
        optimizer_config=OptimizersContainer.Config(lr=1e-4),
        lr_scheduler_config=LRSchedulersContainer.Config(warmup_steps=2),
        dataloader_config=FluxDataLoader.Config(),
        validator_config=FluxValidator.Config(enable=True),
        state_dict_adapter=FluxStateDictAdapter,
    )
)
# pytype: enable=wrong-arg-types
