r"""Pure JAX training for LLM models on TPU.

Parallel to torchtitan/experiments/torchax/train_minimal.py but uses pure
JAX / Flax NNX instead of torchax (the PyTorch-on-JAX wrapper).

Run llama3 8B on v6e-4:
```
python -m torchtitan.experiments.jax.train_minimal \
    --model.name=llama3 \
    --model.flavor=8B \
    --training.dataset_path=tests/assets/c4_test \
    --model.hf_assets_path=tests/assets/tokenizer \
    --training.seq_len=2048 \
    --training.global_batch_size=4 \
    --training.steps=10 \
    --jax_config.use_scan \
    --activation_checkpoint.mode=full
```
"""

import dataclasses
import sys
from typing import Any

from absl import app
import jax
import torchtitan.config
from torchtitan.experiments.jax import jax_job_config
from torchtitan.experiments.jax import trainer as jax_trainer
from torchtitan.experiments.jax import utils as jax_utils
import torchtitan.tools.logging
import tyro


P = jax.sharding.PartitionSpec
logger = torchtitan.tools.logging.logger


def _maybe_init_jax_distributed():
  """Initialise JAX's multi-host runtime when running under a multi-host
  launcher (GKE/XPK, SLURM, etc.). No-op on a single-host VM — libtpu's
  local discovery already gives us all chips.

  Multi-host is detected via the ``TPU_WORKER_HOSTNAMES`` env var exported
  by Cloud-TPU-on-GKE; it holds a comma-separated list of worker hostnames
  (len > 1 ⇒ multi-host). Rank and host count are inferred from
  ``TPU_WORKER_ID`` / the same hostnames list, which are guaranteed to be
  set on every TPU GKE pod.
  """
  import os
  hostnames = os.environ.get('TPU_WORKER_HOSTNAMES')
  if not hostnames:
    return
  hosts = hostnames.split(',')
  if len(hosts) <= 1:
    return
  process_id = int(os.environ.get('TPU_WORKER_ID', '0'))
  coordinator = f"{hosts[0]}:8476"
  logger.info(
      'Initialising jax.distributed: num_processes=%d process_id=%d '
      'coordinator=%s',
      len(hosts), process_id, coordinator,
  )
  jax.distributed.initialize(
      coordinator_address=coordinator,
      num_processes=len(hosts),
      process_id=process_id,
  )


def main_train_loop(job_config: Any):
  torchtitan.tools.logging.init_logger()

  # Must run before any jax.devices() / jit call so the TPU runtime starts
  # with the full multi-host mesh instead of a single-host slice.
  _maybe_init_jax_distributed()

  logger.info('Running with config: %s', job_config)

  devices = jax.devices()
  if not devices:
    raise RuntimeError('No JAX devices found.')
  platform = devices[0].platform
  device_kind = devices[0].device_kind
  accelerator = jax_utils.get_accelerator_short_name(device_kind)
  logger.info(
      "Detected JAX device '%s', accelerator type '%s'",
      device_kind,
      accelerator,
  )

  num_global_devices = jax.device_count()
  num_local_devices = jax.local_device_count()
  logger.info(
      'Devices: %d global, %d local', num_global_devices, num_local_devices
  )

  tp = job_config.parallelism.tensor_parallel_degree
  if tp <= 0:
    tp = 1
  fsdp = num_global_devices // tp

  if platform == 'tpu' and job_config.jax_config.tpu_num_slices > 1:
    dev_array = jax.experimental.mesh_utils.create_hybrid_device_mesh(
        (fsdp // job_config.jax_config.tpu_num_slices, tp),
        (job_config.jax_config.tpu_num_slices, 1),
        devices,
        process_is_granule=False,
        allow_split_physical_axes=True,
    )
    mesh = jax.sharding.Mesh(dev_array, ('fsdp', 'tp'))
  else:
    mesh = jax.make_mesh(
        (fsdp, tp),
        ('fsdp', 'tp'),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )

  logger.info('Mesh: fsdp=%d tp=%d', fsdp, tp)

  with mesh:
    t = jax_trainer.JaxTrainer(
        mesh=mesh,
        job_config=job_config,
        accelerator=accelerator,
        num_global_devices=num_global_devices,
        num_local_devices=num_local_devices,
    )
    return t.train()


def _parse_flags():
  tyro_args = []
  absl_args = [sys.argv[0]]

  config_fields = {
      f.name for f in dataclasses.fields(jax_job_config.JaxJobConfig)
  }

  def is_tyro_arg(arg):
    if not arg.startswith('--'):
      return False
    key = arg[2:].split('=')[0]
    top = key.split('.')[0]
    return top in config_fields

  for arg in sys.argv[1:]:
    if is_tyro_arg(arg):
      tyro_args.append(arg)
    else:
      absl_args.append(arg)
  return tyro_args, absl_args


def main(argv):
  assert main_train_loop(_global_config), 'training failed'


if __name__ == '__main__':
  tyro_args, absl_args = _parse_flags()
  try:
    _global_config = tyro.cli(jax_job_config.JaxJobConfig, args=tyro_args)
  except SystemExit as e:
    if e.code != 0:
      print(f'tyro argument parsing failed: {e.code}')
    sys.exit(e.code)

  sys.argv = absl_args
  app.run(main)
