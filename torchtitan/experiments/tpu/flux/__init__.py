# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import MSELoss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer, default_adamw
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

import torchtitan.models.flux as native_flux_models
from torchtitan.components.data import GrainDataLoader
from torchtitan.models.flux.flux_datasets import (
    DATASETS,
    FluxCollator,
    FluxValidationDatasetConfig,
)
from torchtitan.models.flux.validate import FluxValidator
from torchtitan.experiments.tpu.flux.infra.parallelize import parallelize_flux
from torchtitan.models.flux.model.model import FluxModel
from torchtitan.models.flux.model.state_dict_adapter import FluxStateDictAdapter

__all__ = [
    "flux_configs",
    "parallelize_flux",
]

# Upstream now stores ``(builder, max_context_length)`` tuples.
flux_configs = {
    flavor: builder()
    for flavor, (builder, _max_context_length) in native_flux_models.flux_configs.items()
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
        optimizer_config=default_adamw(lr=1e-4),
        lr_scheduler_config=LRSchedulersContainer.Config(warmup_steps=2),
        dataloader_config=GrainDataLoader.Config(
            dataset=DATASETS["cc12m-test"],
            collator=FluxCollator.Config(),
        ),
        validator_config=FluxValidator.Config(
            enable=True,
            dataloader=GrainDataLoader.Config(
                dataset=FluxValidationDatasetConfig(
                    dataset=DATASETS["cc12m-test-validation"],
                ),
                collator=FluxCollator.Config(),
            ),
        ),
        state_dict_adapter=FluxStateDictAdapter,
    )
)
# pytype: enable=wrong-arg-types
