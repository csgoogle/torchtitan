import functools

import jax
import torch
import torch_xla2
import torch_xla2.train
import torchtitan.distributed
import torchtitan.tools.logging


logger = torchtitan.tools.logging.logger


def sharded_device_put(
    tensor: jax.Array, sharding, num_global_devices, num_local_devices
) -> jax.Array:
  """Places a tensor onto the device with the specified sharding."""
  if isinstance(tensor, tuple):
    return tuple(
        sharded_device_put(t, sharding, num_global_devices, num_local_devices)
        for t in tensor
    )

  if num_global_devices == num_local_devices:
    return jax.device_put(tensor, sharding)

  # Multi-host setup: Each host handles only its accessible devices.
  shape = tensor.shape
  x_split = [
      jax.device_put(tensor[i], device)
      for device, i in sharding.addressable_devices_indices_map(shape).items()
  ]
  return jax.make_array_from_single_device_arrays(shape, sharding, x_split)


def _process_sharding_name(name):
  """Replace integers in param name with *.

  presumably all layers should have the same sharding.

  Args:
    name: The parameter name string.

  Returns:
    The processed parameter name string with integers replaced by '*'.
  """

  def is_integer(t):
    try:
      int(t)
      return True
    except:
      return False

  tokens = name.split('.')
  for i, t in enumerate(tokens):
    if is_integer(t):
      tokens[i] = '*'
  return '.'.join(tokens)


def _make_weight_shard(weight_meta, slice_index):
  """Creates a single shard of a weight with deterministic random initialization.

  This function is designed to be used with `jax.make_array_from_callback`.
  It generates a JAX array for a specific slice of a larger weight, ensuring
  that each shard is initialized with unique, deterministic random values.

  Args:
    weight_meta: The metadata (e.g., shape, dtype) of the full weight.
    slice_index: A tuple of slice objects defining the portion of the weight
      this shard represents.

  Returns:
    A JAX array containing the randomly initialized shard.
  """
  # 1. Determine the specific shape for this shard
  shard_meta = weight_meta[slice_index]
  logger.debug(
      '.... working on shard at index=%s with shape=%s',
      slice_index,
      shard_meta.shape,
  )

  # 2. Deterministic Seeding:
  # Generate a unique random key based on the slice_index.
  # This ensures every device initializes with different noise (or same if desired).
  # Note: Converting slice_index to a stable hash integer for seeding.
  seed = hash(tuple((s.start, s.stop, s.step) for s in slice_index)) % (
      2**31 - 1
  )
  key = jax.random.PRNGKey(seed)

  # 3. Handle Dtype Mapping (Torch -> JAX)
  # Ensure we allocate directly in bfloat16 if the model is bfloat16
  dtype_map = {
      torch.bfloat16: jax.numpy.bfloat16,
      torch.float16: jax.numpy.float16,
      torch.float32: jax.numpy.float32,
      torch.complex64: jax.numpy.complex64,
      torch.complex128: jax.numpy.complex128,
  }
  jax_dtype = dtype_map.get(shard_meta.dtype, jax.numpy.bfloat16)

  # 4. Generate directly on the device using JAX
  # This allocates ONLY the memory needed for the shard, directly on XLA/TPU
  # Reduce the std to 0.01 to avoid blow up the range of activations.
  # TODO(jialeic): better weight initialization like glorot_normal.
  return jax.random.normal(key, shard_meta.shape, dtype=jax_dtype) * 0.01


def create_sharded_weights(model, mesh, sharding_map):
  """Creates sharded JAX arrays for model weights based on a sharding map.

  Args:
    model: The PyTorch model whose state_dict will be used.
    mesh: The JAX device mesh to use for sharding.
    sharding_map: A dictionary mapping weight name patterns to sharding specs
      (e.g., ('fsdp', 'tp')).

  Returns:
    A dictionary mapping weight names to jax.Array objects, sharded according
    to the provided sharding_map. Weights not found in the sharding_map
    (after processing the name) are skipped.
  """
  res = {}
  env = torch_xla2.default_env()
  for name, weight_meta in model.state_dict().items():
    sharding_spec = sharding_map.get(_process_sharding_name(name))
    if sharding_spec is None:
      logger.warning('Skipping weight: %s', name)
      continue
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(*sharding_spec)
    )
    logger.info(
        'Initializing weight %s with shape=%s dtype=%s sharding=%s....',
        name,
        weight_meta.shape,
        weight_meta.dtype,
        sharding
    )
    res[name] = env.j2t_iso(
        jax.make_array_from_callback(
            weight_meta.shape,
            sharding,
            functools.partial(_make_weight_shard, weight_meta),
        )
    )
  return res


class ModelWithScan(torch.nn.Module):
  """Wrapper for transformer models that supports single or split scans."""

  def __init__(
      self, old_transformer, checkpoint_policy, embedding_constants_key
  ):
    super().__init__()
    self.tok_embeddings = old_transformer.tok_embeddings
    self.norm = old_transformer.norm
    self.output = old_transformer.output
    self.embedding_constants_key = embedding_constants_key

    self.register_buffer(
        embedding_constants_key,
        getattr(old_transformer, embedding_constants_key),
    )

    self.single_scan = True
    n_dense_layers = getattr(old_transformer.model_args, 'n_dense_layers', 0)
    all_layers = list(old_transformer.layers.values())

    if n_dense_layers == 0 or n_dense_layers == len(all_layers):
      # keep the original scanned module under "layers" name prefix
      self.layers = torch_xla2.train.ScannedModule(
          all_layers, checkpoint_policy
      )
    else:
      self.single_scan = False
      # assuming always dense layers before moe layers
      dense_layers_list = all_layers[:n_dense_layers]
      moe_layers_list = all_layers[n_dense_layers:]
      logger.info(
          'Creating scan model with %d dense layers and %d moe layers',
          len(dense_layers_list),
          len(moe_layers_list),
      )

      self.layers_dense = torch_xla2.train.ScannedModule(
          dense_layers_list, checkpoint_policy
      )
      self.layers_moe = torch_xla2.train.ScannedModule(
          moe_layers_list, checkpoint_policy
      )

  def forward(self, tokens: torch.Tensor):
    h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
    emb_const = getattr(self, self.embedding_constants_key)

    if self.single_scan:
      h = self.layers(h, emb_const, None)
    else:
      h = self.layers_dense(h, emb_const, None)
      h = self.layers_moe(h, emb_const, None)

    h = self.norm(h) if self.norm else h
    output = self.output(h) if self.output else h
    return output


class TorchaxParallelDims(torchtitan.distributed.ParallelDims):
  """Torchax hack for torchtitan.distributed.ParallelDims:

  Set non_data_parallel_size=1 to be able to correct compute MFU.
    The reason for this hack is torchax use a global dataloader for its
    single-controller training scheme while torchtitan assumes a distributed
    dataloader for its distributed training scheme.
  """

  @property
  def non_data_parallel_size(self):
    return 1
