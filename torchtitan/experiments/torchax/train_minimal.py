r"""model training on TPU with torchax.

This is a simple example for using torchax to train llama3/qwen3 with:
- torchtitan config manager
- model from torchtitan
- dataloader from torchtitan
- sgd/adam/adamw optimizer
- distributed with fsdp + tp
- performance optimization: splash attention kernel, scan layers, offload

Run it with the bazel test hack:
Llama3 8B training (yield MFU ~48%):
```
bazel test //third_party/py/torchtitan/experiments/torchax:torchax_train_test_tpu_tpu_4x2x1 \
    --test_arg=-- \
    --test_arg=--model.name=llama3 \
    --test_arg=--model.flavor=8B \
    --test_arg=--training.dataset=c4_test \
    --test_arg=--training.seq_len=2048 \
    --test_arg=--training.global_batch_size=8 \
    --test_arg=--training.steps=8 \
    --test_arg=--model.hf_assets_path="third_party/py/torchtitan/tests/assets/tokenizer" \
    --test_arg=--training.dataset_path="third_party/py/torchtitan/tests/assets/c4_test" \
    --test_arg=--parallelism.tensor_parallel_degree=1 \
    --test_arg=--torchax_config.use_torchax \
    --test_arg=--torchax_config.use_scan \
    --test_arg=--training.enable_cpu_offload
```
Llama3 8B training on 2xH100 GPU (MFU ~26%):
```
bazel test //third_party/py/torchtitan/experiments/torchax:torchax_train_test_gpu_h100x2 \
    --test_arg=-- \
    --test_arg=--model.name=llama3 \
    --test_arg=--model.flavor=8B \
    --test_arg=--training.dataset=c4_test \
    --test_arg=--training.seq_len=2048 \
    --test_arg=--training.global_batch_size=2 \
    --test_arg=--training.steps=8 \
    --test_arg=--model.hf_assets_path="third_party/py/torchtitan/tests/assets/tokenizer" \
    --test_arg=--training.dataset_path="third_party/py/torchtitan/tests/assets/c4_test" \
    --test_arg=--parallelism.tensor_parallel_degree=1 \
    --test_arg=--torchax_config.use_torchax \
    --test_arg=--torchax_config.use_scan \
    --test_arg=--training.enable_cpu_offload
```
Qwen3 30B-A3B (override layer from 48 to 8, so ~5.5B-A1B, MFU ~25%):
```
bazel test //third_party/py/torchtitan/experiments/torchax:torchax_train_test_tpu_tpu_4x2x1 \
    --test_arg=--alsologtostderr \
    --test_arg=--xprof_end_2_end_upload \
    --test_arg=--xprof_host_trace_level=2 \
    --test_arg=-- \
    --test_arg=--model.name=qwen3 \
    --test_arg=--model.flavor=30B-A3B \
    --test_arg=--training.dataset=c4_test \
    --test_arg=--training.seq_len=8192 \
    --test_arg=--training.global_batch_size=8 \
    --test_arg=--training.steps=8 \
    --test_arg=--model.hf_assets_path="third_party/py/torchtitan/tests/assets/tokenizer" \
    --test_arg=--training.dataset_path="third_party/py/torchtitan/tests/assets/c4_test" \
    --test_arg=--parallelism.tensor_parallel_degree=1 \
    --test_arg=--torchax_config.model_layer_override=8 \
    --test_arg=--torchax_config.use_torchax \
    --test_arg=--torchax_config.use_scan \
    --test_arg=--training.enable_cpu_offload
```
Deepseek_v3 16B (override layer from 27 to 11, so ~6.3B-A1.3B, MFU ~23%):
```
bazel test //third_party/py/torchtitan/experiments/torchax:torchax_train_test_tpu_tpu_4x2x1 \
    --test_arg=-- \
    --test_arg=--model.name=deepseek_v3 \
    --test_arg=--model.flavor=16B \
    --test_arg=--training.dataset=c4_test \
    --test_arg=--training.seq_len=8192 \
    --test_arg=--training.global_batch_size=8 \
    --test_arg=--training.steps=8 \
    --test_arg=--model.hf_assets_path="third_party/py/torchtitan/tests/assets/tokenizer" \
    --test_arg=--training.dataset_path="third_party/py/torchtitan/tests/assets/c4_test" \
    --test_arg=--parallelism.tensor_parallel_degree=1 \
    --test_arg=--torchax_config.model_layer_override=5 \
    --test_arg=--torchax_config.use_torchax \
    --test_arg=--torchax_config.use_scan \
    --test_arg=--training.enable_cpu_offload
```
Run it with launcher:
```
launcher launch third_party/py/torchtitan/launcher/xm_launch.py -- \
    --xm_resource_alloc=cloud-dynamic/cmcs-xm \
    --target=//third_party/py/torchtitan/experiments/torchax:train_minimal  \
    --platform=vl=4x2 \
    -- \
    --model.name=llama3 \
    --model.flavor=8B \
    --training.dataset=fake \
    --training.seq_len=4096 \
    --training.global_batch_size=8 \
    --training.steps=25 \
    --parallelism.tensor_parallel_degree=2 \
    --torchax_config.use_torchax \
    --torchax_config.use_scan \
    --training.enable_cpu_offload
```
"""

import dataclasses
import os
import sys
from typing import Any

from absl import app
import jax
import torch_xla2
import torchtitan.config
from torchtitan.experiments.torchax import distributed
from torchtitan.experiments.torchax import gmm
from torchtitan.experiments.torchax import splash_attn
from torchtitan.experiments.torchax import torchax_job_config
from torchtitan.experiments.torchax import trainer
import torchtitan.tools.logging
import tyro


P = jax.sharding.PartitionSpec
logger = torchtitan.tools.logging.logger


def _get_accelerator_short_name(device_kind: str) -> str:
  """Parses JAX device kind to get accelerator short name like v5e, h100."""
  device_lower = device_kind.lower()
  if 'v4' in device_lower:
    return 'v4'
  # v5e is reported as 'TPU v5 lite'
  if 'v5 lite' in device_lower:
    return 'v5e'
  # v5p is reported as 'TPU v5'
  if device_kind == 'TPU v5':
    return 'v5p'
  if 'v6' in device_lower:
    return 'v6e'
  if 'tpu7x' in device_lower:
    return 'v7x'
  if 'h100' in device_lower:
    return 'h100'
  if 'a100' in device_lower:
    return 'a100'
  raise RuntimeError(
      f'Could not determine accelerator type from JAX device_kind: {device_kind}'
  )


def main_train_loop(job_config: Any):
  """Main training loop for torchax."""
  torchtitan.tools.logging.init_logger()
  assert job_config.torchax_config.use_torchax, 'use_torchax must be True'

  torch_xla2.enable_globally()
  torch_xla2.enable_performance_mode()

  logger.info('Running with config: %s', job_config)

  devices = jax.devices()
  if not devices:
    raise RuntimeError('No JAX devices found!')
  platform = devices[0].platform
  device_type = devices[0].device_kind
  accelerator = _get_accelerator_short_name(
      device_type
  )
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

  # investigate libtpu env vars
  for k, v in os.environ.items():
    logger.info('libtpu env var: %s=%s', k, v)

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
      etp=1,  # no expert tensor parallel for now
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

  env = torch_xla2.default_env()
  env._mesh = mesh  # this is the mesh used by flash attention pallas kernel

  if platform == 'tpu':
    logger.info('Setup TPU-specific kernel overrides.')
    splash_attn.declare_splash_attention(env, mesh)
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


def _parse_flags():
  """Parse flags into tyro args and absl args."""
  tyro_args = []
  absl_args = [sys.argv[0]]

  # Get top-level field names from the TorchaxJobConfig dataclass
  config_fields = {
      f.name for f in dataclasses.fields(torchax_job_config.TorchaxJobConfig)
  }

  def is_tyro_arg(arg):
    if not arg.startswith('--'):
      return False
    key = arg[2:].split('=')[0]
    if '.' in key:
      top_level = key.split('.')[0]
      return top_level in config_fields
    else:
      return key in config_fields

  # Separate args based on whether they look like tyro args for TorchaxJobConfig
  for arg in sys.argv[1:]:
    if is_tyro_arg(arg):
      tyro_args.append(arg)
    else:
      absl_args.append(arg)
  return tyro_args, absl_args


def main(argv):
  # _global_torchtitan_config is populated before app.run is called
  assert main_train_loop(
      _global_torchtitan_config
  ), 'training is not successful'


if __name__ == '__main__':
  tyro_args, absl_args = _parse_flags()

  # Parse tyro-specific arguments to torchtitan config
  try:
    _global_torchtitan_config = tyro.cli(
        torchax_job_config.TorchaxJobConfig, args=tyro_args
    )
  except SystemExit as e:
    if e.code != 0:
      print(f'tyro argument parsing failed. Error code: {e.code}')
    sys.exit(e.code)

  # Replace sys.argv with only the arguments NOT handled by tyro
  sys.argv = absl_args

  app.run(main)
