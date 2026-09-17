# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan.models.flux as native_flux_models
from torchtitan.experiments.tpu.flux.infra.parallelize import parallelize_flux

__all__ = [
    "flux_configs",
    "parallelize_flux",
]

flux_configs = {
    "flux-dev": native_flux_models.flux_configs["flux-dev"](),
    "flux-schnell": native_flux_models.flux_configs["flux-schnell"](),
    "flux-debug": native_flux_models.flux_configs["flux-debug"](),
}
