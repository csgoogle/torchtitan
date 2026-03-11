"""Minimal trainer for torch titan experiments."""

import time
import typing

import jax
import optax
import torch
import torch_xla2
from torch_xla2.interop import jax_view
from torch_xla2.interop import JittableModule
from torch_xla2.interop import torch_view
import torch_xla2.train
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.experiments.torchax import data_utils
from torchtitan.experiments.torchax import deepseek_v3 as torchax_dsv3
from torchtitan.experiments.torchax import distributed
from torchtitan.experiments.torchax import jit_utils
from torchtitan.experiments.torchax import llama3 as torchax_llama3
from torchtitan.experiments.torchax import metrics
from torchtitan.experiments.torchax import moe_utils
from torchtitan.experiments.torchax import qwen3 as torchax_qwen3
import torchtitan.tools.logging


P = jax.sharding.PartitionSpec
logger = torchtitan.tools.logging.logger


def is_moe_model(job_config) -> bool:
  """Returns true if the model is a MoE model.

  Args:
    job_config: The job configuration object.
  """
  if job_config.model.name == 'llama3':
    return False
  elif job_config.model.name == 'qwen3':
    # Naming convention for qwen3 moe model seems to be xB-AyB,
    # i.e. x Billion total parameters with y Billion Active parameters
    return 'B-A' in job_config.model.flavor or 'moe' in job_config.model.flavor
  elif job_config.model.name == 'deepseek_v3':
    return True
  else:
    raise ValueError(f'Unsupported model: {job_config.model.name}')


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

    self.mesh = mesh
    self.parallel_dims = parallel_dims
    self.accelerator = accelerator
    self.num_global_devices = num_global_devices
    self.num_local_devices = num_local_devices
    self.x_sharding = jax.sharding.NamedSharding(self.mesh, P('fsdp'))
    self.replicated = jax.sharding.NamedSharding(self.mesh, P())

    self.job_config = job_config
    self.train_spec = torchtitan.protocols.train_spec.get_train_spec(
        job_config.model.name
    )

  def setup_dataloader(self, job_config):
    """Sets up the dataloader."""
    if (
        job_config.training.dataset is None
        or job_config.training.dataset.startswith('fake')
    ):
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
          self.train_spec.build_tokenizer_fn(job_config)
          if self.train_spec.build_tokenizer_fn is not None
          else None,
      )
      train_loader = self.train_spec.build_dataloader_fn(
          dp_world_size=1,
          dp_rank=0,
          tokenizer=tokenizer,
          job_config=job_config,
      )
    return tokenizer, train_loader

  def setup_model(self, job_config):
    """Sets up the model."""

    # TODO(jialeic): Can we do something like
    # [internal link]

    if job_config.model.name == 'llama3':
      torchax_model = torchax_llama3
    elif job_config.model.name == 'qwen3':
      torchax_model = torchax_qwen3
    elif job_config.model.name == 'deepseek_v3':
      torchax_model = torchax_dsv3
    else:
      raise ValueError(f'Unsupported model: {job_config.model.name}')

    if job_config.torchax_config.use_scan:
      # using scan the individial weights will have shape (num_layers, w, h)
      if is_moe_model(job_config):
        sharding_map = torchax_model.sharding_map_scan_moe
      else:
        sharding_map = torchax_model.sharding_map_scan
    else:
      sharding_map = torchax_model.sharding_map_original

    self.sharding_map = sharding_map

    model_args = torchax_model.args[job_config.model.flavor]
    # Note: torchtitan's upstream config did not specify this value
    if model_args.vocab_size is None:
      logger.warning('vocab_size is None, using 128256 for llama3.')
      model_args.vocab_size = 128256
    model_args.max_seq_len = job_config.training.seq_len
    self.model_args = model_args

    if job_config.torchax_config.model_layer_override is not None:
      logger.warning(
          'Overriding layer from %s to %s for debug',
          model_args.n_layers,
          job_config.torchax_config.model_layer_override,
      )
      model_args.n_layers = (
          job_config.torchax_config.model_layer_override
      )
    if hasattr(model_args, 'use_flex_attn'):
      logger.info(
          'Setting model_args.use_flex_attn to False since we are using'
          ' splash attention kernel for TPU.'
      )
      model_args.use_flex_attn = False

    logger.info('Finial model args: %s', model_args)

    # Note: because a single device don't have enough HBM memory
    # nor enough CPU memory to hold the parameters. We instantiate
    # the model on meta then manually initialize then shard each param
    torch.set_default_dtype(torch.bfloat16)
    with torch.device('meta'):
      model = torchax_model.model(model_args)

    # Sharding for embedding constants
    with torch.device('cpu'):
      if job_config.model.name == 'llama3':
        embedding_constants = model._precompute_freqs_cis()
        embedding_constants_key = 'freqs_cis'
      elif job_config.model.name == 'qwen3':
        embedding_constants = model._precompute_rope_cache()
        embedding_constants_key = 'rope_cache'
      elif job_config.model.name == 'deepseek_v3':
        # HACK: deepseek model does not have a _precompute_freqs_cis wrapping
        # the global precompute_freqs_cis function as in llama3. Consider
        # make this change upstream to torchtitan deepseek model definition
        from torchtitan.models.deepseek_v3.model.model import precompute_freqs_cis
        embedding_constants = precompute_freqs_cis(model_args)
        embedding_constants_key = 'freqs_cis'
      else:
        raise ValueError(f'No embedding constant for: {job_config.model.name}')

    # Remat: default to recompute everything inside transformer block
    checkpoint_policy = jax.checkpoint_policies.nothing_saveable

    if job_config.torchax_config.use_scan:
      if (
          job_config.training.enable_cpu_offload
          and job_config.torchax_config.offload_keys
      ):
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
      model = distributed.ModelWithScan(
          model, checkpoint_policy, embedding_constants_key
      )

    state_dict = dict(model.state_dict())
    state_dict.pop(embedding_constants_key)
    state_dict = distributed.create_sharded_weights(model, self.mesh, sharding_map)
    replicated = jax.sharding.NamedSharding(self.mesh, P())

    state_dict[embedding_constants_key] = embedding_constants.to('jax').apply_jax(
        jax.device_put, replicated
    )
    model.load_state_dict(state_dict, assign=True)
    return model, checkpoint_policy

  def setup_optimizer(self, job_config, jittable_mod):
    """Sets up the optimizer and its state."""

    if job_config.optimizer.name.lower() == 'sgd':
      jax_optimizer = optax.sgd(job_config.optimizer.lr)
      opt_state = torch_view(jax_optimizer.init(jax_view(jittable_mod.params)))

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

      jax_params = jax_view(jittable_mod.params)
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
      opt_state = torch_view(init_fn(jax_params))
    else:
      raise ValueError(
          f'Unsupported optimizer type: {job_config.optimizer.name}'
      )
    return jax_optimizer, opt_state

  def train(self):
    xla_env = torch_xla2.default_env()
    jax.config.update('jax_enable_x64', False)
    xla_env._mesh = self.mesh
    xla_env.use_flash_attention = True

    job_config = self.job_config

    model, checkpoint_policy = self.setup_model(job_config)

    jittable_mod = JittableModule(model)

    # model_fn is responsible to shard if needed
    # to do FSDP one shards the first input args and output on the batch dimension
    def model_fn(weights, buffers, args):
      return jittable_mod.functional_call('forward', weights, buffers, args)

    loss_fn = self.train_spec.build_loss_fn(job_config)

    jax_optimizer, opt_state = self.setup_optimizer(
        job_config, jittable_mod
    )

    _, train_dataloader = self.setup_dataloader(job_config)

    metrics_processor = metrics.TorchaxMetricsProcessor(
        job_config,
        self.parallel_dims,
        self.accelerator,
        self.num_global_devices,
    )

    if is_moe_model(job_config):
      # This is a hack to compute model active paramters
      # When using the jax scan the model name is changed from `moe.experts` to
      # `moe___experts` so we need to use a different naming convention to
      # extract the nparams
      if job_config.model.name == 'qwen3':
        # [internal link]
        head_dims = 2 * self.model_args.head_dim  # pytype: disable=attribute-error
      elif job_config.model.name == 'deepseek_v3':
        # [internal link]
        head_dims = (
            self.model_args.qk_nope_head_dim  # pytype: disable=attribute-error
            + self.model_args.qk_rope_head_dim  # pytype: disable=attribute-error
            + self.model_args.v_head_dim  # pytype: disable=attribute-error
        )
      else:
        raise ValueError(f'Unsupported MoE model: {job_config.model.name}')

      _, metrics_processor.num_flops_per_token = (
          moe_utils.get_moe_model_nparams_and_flops(
              self.model_args, model, head_dims, job_config.training.seq_len
          )
      )
    else:
      _, metrics_processor.num_flops_per_token = (
          self.model_args.get_nparams_and_flops(
              model, job_config.training.seq_len
          )
      )

    train_step = torch_xla2.train.make_train_step(
        model_fn,
        loss_fn,
        jax_optimizer,
        remat_policy=checkpoint_policy
        or jax.checkpoint_policies.nothing_saveable,
    )

    logger.info(
        'Start training %s - %s %s with %s',
        job_config.model.name,
        job_config.model.flavor,
        '(override layer to'
          f' {job_config.torchax_config.model_layer_override})'
          if job_config.torchax_config.model_layer_override
          else '',
        job_config.optimizer.name,
    )

    if job_config.profiling.enable_profiling:
      jax.profiler.start_trace(job_config.profiling.save_traces_folder)

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
        train_step = jit_utils.compile_step_func(
            train_step,
            jittable_mod.params,
            jittable_mod.buffers,
            opt_state,
            inputs,
            labels,
            self.mesh,
        )

      # Ensure previous async ops are done before starting the next step
      jax.block_until_ready((jittable_mod.params, opt_state))

      logger.debug(f'Model input shape: %s', inputs.shape)

      loss, jittable_mod.params, opt_state = train_step(
          jittable_mod.params, jittable_mod.buffers, opt_state, inputs, labels
      )
      # wait for iteration to finish to measure time
      torch_xla2.interop.call_jax(
          jax.block_until_ready, (loss, jittable_mod.params)
      )

      metrics_processor.log(
          step + 1,
          loss.item(),
          loss.item(),
          grad_norm = -1.0,  # grad_norm is not supported for now
      )

      if step >= job_config.training.steps - 1:
        break

    if job_config.profiling.enable_profiling:
      jax.profiler.stop_trace()

    return True
