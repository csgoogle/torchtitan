import collections
import functools

import jax
import torch
import torchax

from torchtitan.experiments.jax.distributed import _match_path
from torchtitan.experiments.jax.distributed import sharded_device_put
import torchtitan.distributed
import torchtitan.tools.logging


logger = torchtitan.tools.logging.logger


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
  # Reduce the std to 0.001 to avoid blow up the range of activations.
  # TODO(jialeic): better weight initialization like glorot_normal.
  return jax.random.normal(key, shard_meta.shape, dtype=jax_dtype) * 0.001


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
  env = torchax.default_env()
  for name, weight_meta in model.state_dict().items():
    sharding_spec = _match_path(name, sharding_map)
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


class ConformerScannedModule(torch.nn.Module):
  """A PyTorch module wrapper that runs Conformer layers in a JAX loop (scan).

  Why a custom class is needed for Conformer:
  Conformer contains BatchNorm layers, which have both parameters (floating-point
  weights/biases) and stateful buffers (like integer step counters and running stats).

  Standard scanning classes assume all layer variables are parameters. This fails
  on Conformer for two reasons:
  1. PyTorch crashes if you try to make integer counters (int64) a Parameter.
  2. Treating BatchNorm running averages as parameters is incorrect because
     the optimizer would try to train them via gradients.

  To fix this, this class:
  - Keeps true trainable weights inside `self.params` (ParameterDict).
  - Registers BatchNorm counters and statistics as non-trainable `buffers`.
  - Executes them in the JAX scan loop, and copies the JAX-updated running stats
    back to the PyTorch buffers at the end of each step.
  """

  def __init__(self, module_list, checkpoint_policy=None):
    super().__init__()
    assert module_list
    self.c = torchax.train.Container()
    self.c.one_mod = module_list[0]  # pyrefly: ignore[missing-attribute]
    self.checkpoint_policy = checkpoint_policy
    weights = self._stack_layer_weights(module_list)
    self.layer_weights_keys = list(self.c.one_mod.state_dict().keys())  # pyrefly: ignore[missing-attribute]

    self.params = torch.nn.ParameterDict()
    for k, v in weights.items():
      new_name = self._param_name_new(k)
      if k in dict(self.c.one_mod.named_parameters()):  # pyrefly: ignore[missing-attribute]
        if v.is_floating_point() or v.is_complex():
          self.params[new_name] = torch.nn.Parameter(v)
        else:
          self.register_buffer(new_name, v)
      else:
        self.register_buffer(new_name, v)

  def _stack_layer_weights(self, module_list):
    temp = collections.defaultdict(list)
    for m in module_list:
      for k, v in m.state_dict().items():
        temp[k].append(v)
    return {k: torch.stack(v) for k, v in temp.items()}

  def _param_name_new(self, old):
    return '___'.join(old.split('.'))

  def forward(self, input_tensor, **kwargs):
    state = {}
    for k in self.layer_weights_keys:
      new_name = self._param_name_new(k)
      if new_name in self.params:
        state[k] = self.params[new_name]
      else:
        state[k] = getattr(self, new_name)

    scan = torchax.interop.torch_view(jax.lax.scan)

    def eval_one_layer(args, weight):
      (h,) = args
      newh = torch.func.functional_call(self.c.one_mod, weight, (h,), kwargs)  # pyrefly: ignore[missing-attribute]
      return (newh,), weight

    if self.checkpoint_policy is None:
      _eval_one_layer = eval_one_layer
    else:
      _eval_one_layer = torchax.interop.gradient_checkpoint(
          eval_one_layer,
          kwargs={'policy': self.checkpoint_policy},
      )
    (result,), updated_state = scan(_eval_one_layer, (input_tensor,), state)

    for k, v in updated_state.items():
      new_name = self._param_name_new(k)
      if new_name not in self.params:
        getattr(self, new_name).copy_(v)

    return result


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
      self.layers = torchax.train.ScannedModule(
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

      self.layers_dense = torchax.train.ScannedModule(
          dense_layers_list, checkpoint_policy
      )
      self.layers_moe = torchax.train.ScannedModule(
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


class ConformerModelWithScan(torch.nn.Module):
  """Wrapper specifically for Conformer model that supports scanning."""

  def __init__(self, old_transformer, checkpoint_policy):
    super().__init__()
    self.tok_embeddings = old_transformer.embedding
    self.norm = None
    self.output = old_transformer.fc

    all_layers = list(old_transformer.conformer_layers)
    self.layers = ConformerScannedModule(all_layers, checkpoint_policy)

  def forward(self, tokens: torch.Tensor):
    h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
    h = self.layers(h)
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
