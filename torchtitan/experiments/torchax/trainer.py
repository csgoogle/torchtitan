"""Minimal trainer for torch titan experiments."""

import collections
import time
import typing

import importlib

import jax
import optax
import torch
import torchax
import torch_xla2.amp as torchax_amp
import torch_xla2.train  # required for ``torchax.train.make_train_step``
from torchtitan.components.dataloader import DataloaderExhaustedError
import torchtitan.components.tokenizer
from torchtitan.experiments.torchax import data_utils
from torchtitan.experiments.jax.metrics import JaxMetricsProcessor
from torchtitan.experiments.torchax import distributed
from torchtitan.experiments.torchax import jit_utils
from torchtitan.experiments.torchax import moe_utils
import torchtitan.tools.logging


# Per-model wrappers under torchtitan.experiments.torchax.* may depend on
# upstream model module layouts (e.g. torchtitan.models.<m>.model.args) that
# drift independently per-lane. Import them on demand inside setup_model so a
# breakage in one lane doesn't take down the trainer for every other lane.
_TORCHAX_MODEL_MODULES = {
    'llama3': 'torchtitan.experiments.torchax.llama3',
    'qwen3': 'torchtitan.experiments.torchax.qwen3',
    'deepseek_v3': 'torchtitan.experiments.torchax.deepseek_v3',
    'redacted': 'torchtitan.experiments.torchax.redacted',
    'redacted': 'torchtitan.experiments.torchax.redacted',
    'conformer': 'torchtitan.experiments.torchax.conformer',
}
from [redacted]._ops.segment_matmul import interface as redacted_segment_matmul_interface
from [redacted].layers import functional as redacted_functional

_original_segment_matmul = redacted_segment_matmul_interface.segment_matmul


# Custom JAX-compatible implementation of [redacted]'s segment_matmul to avoid ConcretizationTypeError
def _custom_segment_matmul_jax(segmented_input, group_sizes, weight):
  env = torchax.default_env()
  if (
      hasattr(segmented_input, "jax")
      and hasattr(group_sizes, "jax")
      and hasattr(weight, "jax")
  ):
    X = segmented_input.jax()  # [T, B, D_IN]
    G = group_sizes.jax()  # [T, E]
    W = weight.jax()  # [T, E, D_IN, D_OUT]

    T = X.shape[0]
    TotalTokens = X.shape[1]
    E = G.shape[1]

    borders = jax.numpy.cumsum(G, axis=-1)  # [T, E]
    token_indices = jax.numpy.arange(TotalTokens)  # [TotalTokens]

    if E > 1:
      # Broadcasted comparison to find expert_ids per token
      expert_ids = jax.numpy.sum(
          token_indices[None, :, None] >= borders[:, None, :-1], axis=-1
      )  # [T, TotalTokens]
    else:
      expert_ids = jax.numpy.zeros((T, TotalTokens), dtype=jax.numpy.int32)

    track_indices = jax.numpy.arange(T)[:, None]  # [T, 1]
    gathered_weights = W[
        track_indices, expert_ids, :, :
    ]  # [T, TotalTokens, D_IN, D_OUT]

    # Batched matmul via einsum
    res_jax = jax.numpy.einsum("tbi,tbio->tbo", X, gathered_weights)

    return env.j2t_iso(res_jax)
  else:
    return _original_segment_matmul(segmented_input, group_sizes, weight)


# Custom override for take_along_dim to support int32 indices on TPU
def _custom_take_along_dim(input_tensor, indices, dim, out=None):
  env = torchax.default_env()
  if hasattr(input_tensor, "jax") and hasattr(indices, "jax"):
    input_jax = input_tensor.jax()
    indices_jax = indices.jax()
    res_jax = jax.numpy.take_along_axis(input_jax, indices_jax, axis=dim)
    return env.j2t_iso(res_jax)
  else:
    with torchax.disable_temporarily():
      return torch.take_along_dim(input_tensor, indices, dim, out=out)



P = jax.sharding.PartitionSpec
logger = torchtitan.tools.logging.logger


def is_moe_model(job_config) -> bool:
  """Returns true if the model is a MoE model.

  Args:
    job_config: The job configuration object.
  """
  if job_config.model_spec.name == 'llama3':
    return False
  elif job_config.model_spec.name == 'redacted':
    return False
  elif job_config.model_spec.name == 'redacted':
    # We use the model's own get_nparams_and_flops via the simple branch;
    # bypassing the moe_utils path which doesn't know about [redacted]'s MoE
    # parameter naming convention.
    return False
  elif job_config.model_spec.name == 'qwen3':
    # Naming convention for qwen3 moe model seems to be xB-AyB,
    # i.e. x Billion total parameters with y Billion Active parameters
    return 'B-A' in job_config.model_spec.flavor or 'moe' in job_config.model_spec.flavor
  elif job_config.model_spec.name == 'deepseek_v3':
    return True
  elif job_config.model_spec.name == 'conformer':
    return False
  else:
    raise ValueError(f'Unsupported model: {job_config.model_spec.name}')


class TorchaxTrainer:

  def __init__(
      self,
      mesh,
      parallel_dims,
      job_config,
      accelerator: str,
      num_global_devices: int,
      num_local_devices: int,
  ):
    torchax.default_env().override_op_definition(
        torch.take_along_dim, _custom_take_along_dim
    )
    redacted_segment_matmul_interface.segment_matmul = _custom_segment_matmul_jax
    redacted_functional.segment_matmul = _custom_segment_matmul_jax

    self.mesh = mesh
    self.parallel_dims = parallel_dims
    self.accelerator = accelerator
    self.num_global_devices = num_global_devices
    self.num_local_devices = num_local_devices
    self.x_sharding = jax.sharding.NamedSharding(self.mesh, P('fsdp'))
    self.replicated = jax.sharding.NamedSharding(self.mesh, P())

    self.job_config = job_config

  def setup_dataloader(self, job_config):
    """Sets up the dataloader."""
    # TorchAx uses a single controller for all devices, so the single
    # DataLoader must fetch the data for all replicas globally.
    # Torchtitan defines `local_batch_size` as the PER CHIP batch size.
    # So we temporarily override `local_batch_size` to be the global batch size.
    expected_global_batch_size = job_config.training.global_batch_size
    if expected_global_batch_size <= 0:
      expected_global_batch_size = (
          job_config.training.local_batch_size
          * self.parallel_dims.dp_shard
          * self.parallel_dims.dp_replicate
      )

    original_local_batch_size = job_config.training.local_batch_size
    job_config.training.local_batch_size = expected_global_batch_size

    dataset = job_config.dataloader.dataset
    if not dataset or dataset.startswith('fake'):
      # fake dataloader producing random ints, no tokenizer
      tokenizer = None
      train_loader = data_utils.fake_dataloader(
          job_config.training.steps,
          job_config.training.seq_len,
          job_config.training.global_batch_size,
      )
      logger.warning('Using fake data loader.')
    else:
      tokenizer = typing.cast(
          torchtitan.components.tokenizer.HuggingFaceTokenizer,
          job_config.tokenizer.build(
              tokenizer_path=job_config.hf_assets_path,
          ),
      )

      train_loader = job_config.dataloader.build(
          dp_world_size=1,
          dp_rank=0,
          tokenizer=tokenizer,
          seq_len=job_config.training.seq_len,
          local_batch_size=job_config.training.local_batch_size,
      )

    job_config.training.local_batch_size = original_local_batch_size
    return tokenizer, train_loader

  def setup_model(self, job_config):
    """Sets up the model."""

    # TODO(jialeic): Can we do something like
    # [internal link]

    if job_config.model_spec.name not in _TORCHAX_MODEL_MODULES:
      raise ValueError(f'Unsupported model: {job_config.model_spec.name}')
    torchax_model = importlib.import_module(
        _TORCHAX_MODEL_MODULES[job_config.model_spec.name]
    )

    if job_config.torchax_config.use_scan:
      # using scan the individial weights will have shape (num_layers, w, h)
      if is_moe_model(job_config):
        sharding_map = torchax_model.sharding_map_scan_moe
      elif 'lora' in job_config.model_spec.flavor and hasattr(
          torchax_model, 'sharding_map_scan_lora'
      ):
        if job_config.torchax_config.use_ddp_sharding and hasattr(
            torchax_model, 'sharding_map_scan_lora_ddp'
        ):
          sharding_map = torchax_model.sharding_map_scan_lora_ddp
          logger.info('Using DDP-all sharding: base model replicated, no fsdp all-gather')
        else:
          sharding_map = torchax_model.sharding_map_scan_lora
      else:
        sharding_map = torchax_model.sharding_map_scan
    else:
      sharding_map = torchax_model.sharding_map_original

    self.sharding_map = sharding_map

    model_args = torchax_model.args[job_config.model_spec.flavor]
    # Note: torchtitan's upstream config did not specify this value
    if model_args.vocab_size is None:
      logger.warning('vocab_size is None, using 128256 for llama3.')
      model_args.vocab_size = 128256
    if hasattr(model_args, 'max_seq_len'):
      # Some ModelArgs (llama3/qwen3) hold ``max_seq_len`` to size RoPE/attn
      # caches; redacted/redacted handle that internally and don't expose it.
      model_args.max_seq_len = job_config.training.seq_len
    self.model_args = model_args

    if job_config.torchax_config.model_layer_override is not None:
      current_layers = getattr(
          model_args, 'n_layers', getattr(model_args, 'num_layers', None)
      )
      logger.warning(
          'Overriding layer from %s to %s for debug',
          current_layers,
          job_config.torchax_config.model_layer_override,
      )
      if hasattr(model_args, 'n_layers'):
        model_args.n_layers = job_config.torchax_config.model_layer_override
      elif hasattr(model_args, 'num_layers'):
        model_args.num_layers = job_config.torchax_config.model_layer_override
    if hasattr(model_args, 'use_flex_attn'):
      logger.info(
          'Setting model_args.use_flex_attn to False since we are using'
          ' splash attention kernel for TPU.'
      )
      model_args.use_flex_attn = False

    logger.info('Finial model args: %s', model_args)

    # Map job_config.training.dtype (Literal['bfloat16','float32']) to the
    # torch master-weight dtype. Default bf16 preserves the prior hardcoded
    # behavior; setting training.dtype=float32 keeps the torchtitan model
    # parameters in fp32 (relying on mixed-precision autocast / manual casts
    # in the forward pass for bf16 matmul compute).
    _dtype_map = {'bfloat16': torch.bfloat16, 'float32': torch.float32}
    param_dtype = _dtype_map[job_config.training.dtype]
    logger.info('torchax model param dtype: %s', param_dtype)
    torch.set_default_dtype(param_dtype)

    # Note: because a single device don't have enough HBM memory
    # nor enough CPU memory to hold the parameters. We instantiate
    # the model on meta then manually initialize then shard each param
    with torch.device('meta'):
      model = torchax_model.model(model_args)

    # Sharding for embedding constants
    with torch.device('cpu'):
      if job_config.model_spec.name == 'llama3':
        embedding_constants = model._precompute_freqs_cis()
        embedding_constants_key = 'freqs_cis'
      elif job_config.model_spec.name == 'qwen3':
        embedding_constants = model._precompute_rope_cache()
        embedding_constants_key = 'rope_cache'
      elif job_config.model_spec.name == 'deepseek_v3':
        # New-style upstream: DeepSeekV3Model uses the shared
        # torchtitan.models.common.rope.RoPE; the freqs_cis cache lives at
        # ``model.rope.cache``. Build the RoPE on CPU (model itself is on
        # meta) and pull its precomputed cache. Replaces an older import of
        # ``deepseek_v3.model.model.precompute_freqs_cis`` from the
        # pre-restructure upstream layout.
        from torchtitan.models.common.rope import RoPE
        embedding_constants = RoPE(model_args.rope).cache
        embedding_constants_key = 'freqs_cis'
      elif job_config.model_spec.name == 'redacted':
        embedding_constants = None
        embedding_constants_key = None
      elif job_config.model_spec.name == 'redacted':
        # REDACTED PT MoE handles RoPE internally; no precomputed embedding constants needed.
        embedding_constants = None
        embedding_constants_key = None
      elif job_config.model_spec.name == 'conformer':
        embedding_constants = None
        embedding_constants_key = None
      else:
        raise ValueError(f'No embedding constant for: {job_config.model_spec.name}')

    # Remat: default to recompute everything inside transformer block
    if job_config.activation_checkpoint.mode == 'full':
      checkpoint_policy = jax.checkpoint_policies.nothing_saveable
    elif job_config.activation_checkpoint.mode == 'selective':
      checkpoint_policy = jax.checkpoint_policies.dots_saveable
    elif job_config.activation_checkpoint.mode == 'everything':
      checkpoint_policy = jax.checkpoint_policies.everything_saveable
    elif job_config.activation_checkpoint.mode == 'offload_dot':
      checkpoint_policy = jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    elif job_config.activation_checkpoint.mode == 'nothing':
      checkpoint_policy = None
    else:
      checkpoint_policy = jax.checkpoint_policies.nothing_saveable

    if job_config.training.enable_cpu_offload and job_config.torchax_config.offload_keys:
      offload_keys = job_config.torchax_config.offload_keys.split('|')
      logger.info('Offloading keys: %s to host.', offload_keys)
      checkpoint_policy = (
            jax.checkpoint_policies.save_and_offload_only_these_names(
                names_which_can_be_saved=[],
                names_which_can_be_offloaded=offload_keys,
                offload_src='device',
                offload_dst='pinned_host',
            )
        )

    if job_config.torchax_config.use_scan:
      if job_config.model_spec.name == 'redacted':
        redacted_model = model.model
        seg0 = redacted_model.layers.segment_0
        seg0_layers = list(seg0.layers)
        scanned_0 = distributed.redactedScannedModule(
            seg0_layers[:-1], checkpoint_policy
        )
        new_modules = collections.OrderedDict()
        new_side_keys = collections.defaultdict(list)
        new_output_keys = {}

        new_modules['scan_0'] = scanned_0
        new_side_keys['scan_0'] = [('**kwargs', '**kwargs')]

        new_modules['last_0'] = seg0_layers[-1]
        last_name = list(seg0._named_layers_dict.keys())[-1]
        new_side_keys['last_0'] = seg0.side_input_keys[last_name]
        if last_name in seg0.side_output_keys:
          new_output_keys['last_0'] = seg0.side_output_keys[last_name]

        seg0._modules = new_modules
        seg0._side_input_keys = new_side_keys
        seg0._side_output_keys = new_output_keys

        if hasattr(redacted_model.layers, 'segment_1'):
          seg1 = redacted_model.layers.segment_1
          seg1_layers = list(seg1.layers)
          scanned_1 = distributed.redactedScannedModule(
              seg1_layers, checkpoint_policy
          )

          new_modules_1 = collections.OrderedDict()
          new_side_keys_1 = collections.defaultdict(list)
          new_output_keys_1 = {}

          new_modules_1['scan_1'] = scanned_1
          new_side_keys_1['scan_1'] = [('**kwargs', '**kwargs')]

          seg1._modules = new_modules_1
          seg1._side_input_keys = new_side_keys_1
          seg1._side_output_keys = new_output_keys_1
      if job_config.model_spec.name == 'conformer':
        model = distributed.ConformerModelWithScan(
            model, checkpoint_policy
        )
      elif job_config.model_spec.name != 'redacted':
        model = distributed.ModelWithScan(
            model, checkpoint_policy, embedding_constants_key
        )

    state_dict = dict(model.state_dict())
    if embedding_constants_key is not None:
      state_dict.pop(embedding_constants_key)
    state_dict = distributed.create_sharded_weights(model, self.mesh, sharding_map)
    replicated = jax.sharding.NamedSharding(self.mesh, P())

    if embedding_constants_key is not None:
      state_dict[embedding_constants_key] = embedding_constants.to(
          'jax'
      ).apply_jax(jax.device_put, replicated)
    model.load_state_dict(state_dict, assign=True)
    return model, checkpoint_policy

  def setup_optimizer(self, job_config, jittable_mod):
    """Sets up the optimizer and its state."""

    if job_config.optimizer.name.lower() == 'sgd':
      jax_optimizer = optax.sgd(job_config.optimizer.lr)
      opt_state = torchax.interop.torch_view(jax_optimizer.init(torchax.interop.jax_view(jittable_mod.params)))

    elif job_config.optimizer.name.lower() in ('adam', 'adamw'):
      if job_config.optimizer.name.lower() == 'adam':
        jax_optimizer = optax.adam(
            learning_rate=job_config.optimizer.lr,
            b1=job_config.optimizer.beta1 or 0.9,
            b2=job_config.optimizer.beta2 or 0.95,
            eps=job_config.optimizer.eps or 1e-08,
        )
      else:
        # AdamW usually requires weight decay
        jax_optimizer = optax.adamw(
            learning_rate=job_config.optimizer.lr,
            b1=job_config.optimizer.beta1 or 0.9,
            b2=job_config.optimizer.beta2 or 0.95,
            eps=job_config.optimizer.eps or 1e-08,
            weight_decay=job_config.optimizer.weight_decay or 0.1,
        )

      jax_params = torchax.interop.jax_view(jittable_mod.params)
      abstract_opt_state = jax.eval_shape(jax_optimizer.init, jax_params)

      def _get_opt_sharding(path, x):
        """Determines the sharding for a given optimizer state parameter.

        Args:
          path: The path to the current node in the JAX tree.
          x: The value of the node (unused, but required by tree_map_with_path).

        Returns:
          A jax.sharding.NamedSharding object specifying the sharding for the
          parameter.
        """
        param_name = None
        for node in path:
          if isinstance(node, jax.tree_util.DictKey):
            if node.key in self.sharding_map:
              param_name = node.key
              break

        if param_name:
          spec = self.sharding_map[param_name]
          return jax.sharding.NamedSharding(self.mesh, P(*spec))

        # Default to replicated for scalars (like count) or unmapped params
        return self.replicated

      # Use tree_map_with_path to access the dictionary keys (param names)
      opt_state_sharding = jax.tree_util.tree_map_with_path(
          _get_opt_sharding, abstract_opt_state
      )

      # Initialize directly into Sharded Memory
      init_fn = jax.jit(jax_optimizer.init, out_shardings=opt_state_sharding)
      opt_state = torchax.interop.torch_view(init_fn(jax_params))
    else:
      raise ValueError(
          f'Unsupported optimizer type: {job_config.optimizer.name}'
      )
    return jax_optimizer, opt_state

  def train(self):
    xla_env = torchax.default_env()
    jax.config.update('jax_enable_x64', False)
    xla_env._mesh = self.mesh
    xla_env.use_flash_attention = True

    job_config = self.job_config

    model, checkpoint_policy = self.setup_model(job_config)

    jittable_mod = torchax.interop.JittableModule(model)

    # model_fn is responsible to shard if needed
    # to do FSDP one shards the first input args and output on the batch dimension
    use_autocast = (
        job_config.training.mixed_precision_param == 'bfloat16'
        and job_config.model_spec.name == 'conformer'
    )
    autocast_dtype = torch.bfloat16 if use_autocast else None

    def model_fn(weights, buffers, args):
      if autocast_dtype is not None:
        with torchax_amp.autocast('jax', dtype=autocast_dtype):
          return jittable_mod.functional_call('forward', weights, buffers, args)
      else:
        return jittable_mod.functional_call('forward', weights, buffers, args)

    loss_fn = job_config.loss.build(compile_config=job_config.compile)

    jax_optimizer, opt_state = self.setup_optimizer(
        job_config, jittable_mod
    )

    _, train_dataloader = self.setup_dataloader(job_config)

    metrics_processor = JaxMetricsProcessor(
        job_config,
        self.accelerator,
        self.num_global_devices,
        tpu_megacore=job_config.torchax_config.tpu_megacore,
        log_freq=job_config.metrics.log_freq,
        parallel_dims=self.parallel_dims,
    )

    if is_moe_model(job_config):
      # This is a hack to compute model active paramters
      # When using the jax scan the model name is changed from `moe.experts` to
      # `moe___experts` so we need to use a different naming convention to
      # extract the nparams
      if job_config.model_spec.name == 'qwen3':
        # [internal link]
        head_dims = 2 * self.model_args.head_dim  # pytype: disable=attribute-error
      elif job_config.model_spec.name == 'deepseek_v3':
        # [internal link]
        head_dims = (
            self.model_args.qk_nope_head_dim  # pytype: disable=attribute-error
            + self.model_args.qk_rope_head_dim  # pytype: disable=attribute-error
            + self.model_args.v_head_dim  # pytype: disable=attribute-error
        )
      else:
        raise ValueError(f'Unsupported MoE model: {job_config.model_spec.name}')

      _, metrics_processor.num_flops_per_token = (
          moe_utils.get_moe_model_nparams_and_flops(
              self.model_args, model, head_dims, job_config.training.seq_len
          )
      )
    else:
      if hasattr(self.model_args, 'get_nparams_and_flops'):
        _, metrics_processor.num_flops_per_token = (
            self.model_args.get_nparams_and_flops(
                model, job_config.training.seq_len
            )
        )
      else:
        # Fallback if get_nparams_and_flops is not implemented (e.g., redacted)
        metrics_processor.num_flops_per_token = 0

    if job_config.torchax_config.split_compile:
      grad_step, opt_step = jit_utils.make_split_train_step(
          model_fn,
          loss_fn,
          jax_optimizer,
          remat_policy=checkpoint_policy,
      )
      train_step = None  # compiled later on first step
    else:
      train_step = torchax.train.make_train_step(
          model_fn,
          loss_fn,
          jax_optimizer,
          remat_policy=checkpoint_policy,
      )

    logger.info(
        'Start training %s - %s %s with %s',
        job_config.model_spec.name,
        job_config.model_spec.flavor,
        '(override layer to'
          f' {job_config.torchax_config.model_layer_override})'
          if job_config.torchax_config.model_layer_override
          else '',
        job_config.optimizer.name,
    )

    if job_config.profiler.enable_profiling:
      jax.profiler.start_trace(job_config.profiler.save_traces_folder)

    data_iterator = iter(train_dataloader)
    step = -1

    while True:
      step += 1
      data_load_start = time.perf_counter()
      try:
        batch = next(data_iterator)
      except StopIteration as ex:
        raise DataloaderExhaustedError() from ex

      inputs, labels = batch[0]['input'], batch[1]
      ntokens_batch = labels.numel()

      # Move them to jax device
      inputs = inputs.to('jax')
      labels = labels.to('jax')

      # Shard them on batch dim for fsdp
      inputs.apply_jax_(
          distributed.sharded_device_put,
          self.x_sharding,
          self.num_global_devices,
          self.num_local_devices,
      )
      labels.apply_jax_(
          distributed.sharded_device_put,
          self.x_sharding,
          self.num_global_devices,
          self.num_local_devices,
      )

      metrics_processor.ntokens_since_last_log += ntokens_batch
      metrics_processor.data_loading_times.append(
          time.perf_counter() - data_load_start
      )

      if step == 0:
        # Compile step specifically for the first iteration
        if job_config.torchax_config.split_compile:
          train_step = jit_utils.compile_split_step_func(
              grad_step,
              opt_step,
              jittable_mod.params,
              jittable_mod.buffers,
              opt_state,
              inputs,
              labels,
              self.mesh,
          )
        else:
          train_step = jit_utils.compile_step_func(
              train_step,
              jittable_mod.params,
              jittable_mod.buffers,
              opt_state,
              inputs,
              labels,
              self.mesh,
          )

      # (Removed jax.block_until_ready here to allow async overlap)

      logger.debug('Model input shape: %s', inputs.shape)

      loss, jittable_mod.params, opt_state = train_step(
          jittable_mod.params, jittable_mod.buffers, opt_state, inputs, labels
      )

      # Only block and log periodically to avoid stalling the TPU hardware pipeline.
      if metrics_processor.should_log(step + 1):
        # wait for iteration to finish to measure time
        torchax.interop.call_jax(
            jax.block_until_ready, (loss, jittable_mod.params)
        )

        # Torchtitan cross entropy returns the sum across all tokens.
        # Normalize by token count for log readability (loss ~10 rather than ~700k).
        norm_loss = loss.item() / ntokens_batch

        metrics_processor.log(
            step + 1,
            norm_loss,
            norm_loss,
            grad_norm = -1.0,  # grad_norm is not supported for now
        )

      if step >= job_config.training.steps - 1:
        # Prevent premature exit before the last step finishes
        torchax.interop.call_jax(
            jax.block_until_ready, (loss, jittable_mod.params)
        )
        break

    if job_config.profiler.enable_profiling:
      jax.profiler.stop_trace()

    return True
