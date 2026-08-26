# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Translating a torchtitan TPU config into scan parameters.

The rest of this package takes plain values (`peel_first_layer: bool`) so it
stays usable from a model that does not have torchtitan's config objects. This
module is the one place that reads a config, so both lanes -- fullgraph
(`llama3/infra/parallelize.py`, via `train.py`) and graph_trainer
(`graph_trainer/config.py`) -- resolve the same knob the same way.
"""

import os
from typing import Any


def resolve_peel_first_layer(train_config: Any) -> bool:
  """Whether layer 0 runs outside the scan.

  A JAX-exported kernel inside the scan body lowers to a `func.call` whose
  callee torch_tpu drops when it clones the body into the `stablehlo.while`
  region, so a splash-attention block needs layer 0 peeled out to put the callee
  back in the outer module. `TPU_SCAN_PEEL_FIRST_LAYER=0|1` forces it either
  way, so the peel's cost can be measured against an otherwise identical scan,
  and so the workaround can be switched off once torch_tpu carries the real fix.
  """
  peel_env = os.environ.get("TPU_SCAN_PEEL_FIRST_LAYER")
  if peel_env is not None:
    return peel_env == "1"
  sa_cfg = getattr(train_config, "splash_attention_kernel", None)
  return bool(
      getattr(sa_cfg, "use_splash_attention_kernel", False)
      or getattr(sa_cfg, "use_tokamax_splash_attention_kernel", False)
  )


__all__ = ["resolve_peel_first_layer"]
