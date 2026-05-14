r"""model training on TPU with torchax.

This is a simple example for using torchax to train llama3/qwen3 with:
- torchtitan config manager
- model from torchtitan
- dataloader from torchtitan
- sgd/adam/adamw optimizer
- distributed with fsdp + tp
- performance optimization: splash attention kernel, scan layers, offload
"""

import os
import sys
import typing
from typing import Any

from absl import app
import jax
import torchax
import torchtitan.config
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.jax import utils as jax_utils
from torchtitan.experiments.torchax import distributed
from torchtitan.experiments.torchax import gmm
from torchtitan.experiments.torchax import splash_attn
from torchtitan.experiments.torchax import torchax_job_config
from torchtitan.experiments.torchax import trainer
import torchtitan.tools.logging


P = jax.sharding.PartitionSpec
logger = torchtitan.tools.logging.logger


def _maybe_init_jax_distributed():
  """Initialise JAX's multi-host runtime when running under a multi-host
  launcher (GKE/XPK, etc.). No-op on single-host."""
  hostnames = os.environ.get('TPU_WORKER_HOSTNAMES')
  if not hostnames:
    return
  hosts = hostnames.split(',')
  if len(hosts) <= 1:
    return
  process_id = int(os.environ.get('TPU_WORKER_ID', '0'))
  coordinator = f"{hosts[0]}:8476"
  logger.info(
      'Initialising jax.distributed: num_processes=%d process_id=%d coordinator=%s',
      len(hosts), process_id, coordinator,
  )
  jax.distributed.initialize(
      coordinator_address=coordinator,
      num_processes=len(hosts),
      process_id=process_id,
  )


def main_train_loop(job_config: Any):
  """Main training loop for torchax."""
  torchtitan.tools.logging.init_logger()
  _maybe_init_jax_distributed()  # multi-host JAX init (no-op on single-host)
  assert job_config.torchax_config.use_torchax, 'use_torchax must be True'

  torchax.enable_globally()
  torchax.enable_performance_mode()

  logger.info('Running with config: %s', job_config)

  devices = jax.devices()
  if not devices:
    raise RuntimeError('No JAX devices found!')
  platform = devices[0].platform
  device_type = devices[0].device_kind
  accelerator = jax_utils.get_accelerator_short_name(device_type)
  logger.info(
      "Detected JAX device '%s', using accelerator type '%s' for metrics",
      device_type,
      accelerator,
  )

  num_global_devices = jax.device_count()
  num_local_devices = jax.local_device_count()
  num_hosts = jax.process_count()
  logger.info(
      'Running job on %s (%s) with %s num_global_devices, %s'
      ' num_local_devices, %s num_hosts, %s tpu_num_slices',
      platform.upper(),
      device_type,
      num_global_devices,
      num_local_devices,
      num_hosts,
      job_config.torchax_config.tpu_num_slices,
  )

  # only 2D sharding (fsdp x tp) is supported for now
  fsdp = num_global_devices // job_config.parallelism.tensor_parallel_degree

  # this is need for torchtitan metrics processor not used for sharding
  parallel_dims = distributed.TorchaxParallelDims(
      dp_shard=fsdp,
      dp_replicate=1,  # always using fsdp for better performance
      tp=job_config.parallelism.tensor_parallel_degree,
      cp=1,  # no context parallel for now
      pp=1,  # no pipeline parallel for now
      ep=1,  # no expert parallel for now
      world_size=num_global_devices,
  )

  # torchax sharding with JAX mesh API
  if platform == 'tpu' and job_config.torchax_config.tpu_num_slices > 1:
    # tp only through ICI, and fsdp through ICI & DCN
    dev_array = jax.experimental.mesh_utils.create_hybrid_device_mesh(
        (
            parallel_dims.dp_shard // job_config.torchax_config.tpu_num_slices,
            parallel_dims.tp,
        ),
        (job_config.torchax_config.tpu_num_slices, 1),
        devices,
        process_is_granule=False,
        allow_split_physical_axes=True,
    )
    mesh = jax.sharding.Mesh(dev_array, ('fsdp', 'tp'))
  else:
    # Standard mesh for GPU or single-slice TPU
    mesh = jax.make_mesh(
        (parallel_dims.dp_shard, parallel_dims.tp),
        ('fsdp', 'tp'),
        axis_types=(jax.sharding.AxisType.Auto,) * len(('fsdp', 'tp')),
    )

  env = torchax.default_env()
  env._mesh = mesh  # this is the mesh used by flash attention pallas kernel

  if platform == 'tpu':
    logger.info('Setup TPU-specific kernel overrides.')
    splash_attn.declare_splash_attention(
        env, mesh, job_config.splash_attention_kernel
    )
    gmm.declare_gmm_kernel(env, mesh)

  with mesh:
    torchax_trainer = trainer.TorchaxTrainer(
        mesh,
        parallel_dims,
        job_config,
        accelerator,
        num_global_devices,
        num_local_devices,
    )
    return torchax_trainer.train()


def main(argv):
  # Args are passed via ``sys.argv`` (set in __main__ from absl-filtered args).
  # ConfigManager parses ``--module=...`` + ``--config=...`` to load the base
  # config from the per-model ``config_registry``, then applies any remaining
  # CLI overrides via ``tyro.cli`` with ``default=loaded_config``. This is the
  # same launch surface the torch_tpu lane uses (see ``experiments/tpu/gmain``).
  del argv
  config = typing.cast(
      torchax_job_config.TorchaxJobConfig,
      ConfigManager().parse_args(sys.argv[1:]),
  )
  assert main_train_loop(config), 'training is not successful'


if __name__ == '__main__':
  # absl owns ``sys.argv[0]`` parsing. Forward all flags to ``main`` via
  # ``ConfigManager`` rather than absl flags. ``flags.FLAGS(args, known_only=True)``
  # consumes only the absl-registered flags and leaves the rest for us.
  from absl import flags
  app.run(
      main,
      flags_parser=(
          lambda args: flags.FLAGS(args, known_only=True)
      ),
  )
