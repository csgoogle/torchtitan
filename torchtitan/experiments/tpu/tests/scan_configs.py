# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test-only recipes for the scan work.

Reachable the same way as any shipped recipe, by fully qualified module path::

    --module=torchtitan.experiments.tpu.tests.scan_configs \
    --config=llama3_debugmodel_kernels

These live here rather than in `experiments/tpu/llama3/config_registry.py`
because nothing outside the tests runs them yet. Almost everything below *is*
expressible as a command-line flag -- all 22 non-model differences from
`llama3_debugmodel` parse fine -- so the only reason this is Python at all is
the model: `model_spec` is excluded from the tyro surface (see the comment in
`protocols/model_spec.py`), and both `--model_spec.flavor=...` and
`--model_spec.model.dim=...` are rejected.
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.validate import Validator
from torchtitan.config.configs import CompileConfig, ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.tpu.llama3.config_registry import llama3_debugmodel
from torchtitan.experiments.tpu.llama3.infra.parallelize import (
    parallelize_llama as tpu_parallelize_llama,
)
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.tpu_job_config import (
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
)
from torchtitan.models.llama3.config_registry import model_registry

_SPLASH_BLOCKS = (
    "q",
    "kv",
    "dkv",
    "kv_compute",
    "q_dkv",
    "kv_dkv",
    "kv_dkv_compute",
    "q_dq",
    "kv_dq",
)


def llama3_debugmodel_kernels() -> TPUTrainerConfig:
  """`llama3_debugmodel` with the kernel set that 8B/70B/405B actually run.

  `llama3_debugmodel` is plain SDPA with `SelectiveAC` and no kernels; `8B`,
  `70B` and `405B` use `FullAC`, splash attention and the Pallas loss. Anything
  validated only on `llama3_debugmodel` is therefore never once run against the
  code path it ships into -- it passes, then fails at 8B on a kernel the debug
  config does not contain. This closes that gap at debug cost, and splash in
  particular is what the scan work has to keep working.

  Sized for **all 8 chips of a v6e-8**, not one: FSDP is the thing under test,
  and at world size 1 there is nothing to gather.
  """
  config = llama3_debugmodel()
  # The one thing no flag can reach. `debugmodel` is head_dim=16 and splash
  # attention needs 128, so the flavour has to be selected in Python.
  config.model_spec = model_registry("debugmodel_kernels", attn_backend="sdpa")
  config.model_spec.parallelize_fn = tpu_parallelize_llama

  # The fused loss is selected the new way -- `PallasCrossEntropyLoss.Config`
  # plus the LM-head override below -- not by `loss_kernel.use_loss_kernel`.
  # The legacy flag is applied inside `build_tpu_loss`, which the graph trainer
  # never calls: `compute_annotated_loss` reads `job_config.loss` directly, so
  # the flag would patch the LM head into returning `(hidden, weight.T)` while
  # the loss stayed ordinary cross entropy, and the run dies in `pred.flatten`
  # on a tuple. The override swaps that LM head in, and has to be registered
  # before the model is constructed.
  config.loss = PallasCrossEntropyLoss.Config()
  config.override.imports.append(
      "torchtitan.experiments.tpu.overrides.fused_loss"
  )

  config.training.local_batch_size = 1
  config.parallelism = ParallelismConfig(use_simple_fsdp=True)
  config.activation_checkpoint = FullAC.Config()
  config.checkpoint = CheckpointManager.Config(
      enable=False, interval=500, last_save_model_only=True
  )
  config.validator = Validator.Config(enable=False, freq=500, steps=10)
  config.compile = CompileConfig(
      enable=True, backend="tpu", components=["model"]
  )
  config.tpu_config = TPUConfig(
      eager_mode="DEFER_AND_FUSE",
      use_graph_split=False,
      use_graph_trainer=True,
      compile_mode="aot_fx_trace",
  )
  # Block sizes are the sequence length quartered, as in the 8B recipe --
  # scaled down with `seq_len` rather than copied from it, since a block larger
  # than the sequence is not a valid splash configuration.
  config.splash_attention_kernel = SplashAttentionKernelConfig(
      use_splash_attention_kernel=True,
      **{f"sa_block_{n}": 512 for n in _SPLASH_BLOCKS},
  )
  return config
