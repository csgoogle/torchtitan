"""JAX trainer for Llama3 and other models.

Mirrors the structure of torchax/trainer.py but uses pure JAX / Flax NNX
instead of the torchax PyTorch-on-JAX wrapper.

Training loop:
  1. Model defined in Flax NNX; parameters split into (graphdef, state).
  2. Optimizer state managed by nnx.Optimizer (integrates optax).
  3. Train step compiled with nnx.jit + nnx.value_and_grad.
  4. Inputs sharded on FSDP axis; parameters sharded per sharding_map.
"""

import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.experiments.jax import data_utils
from torchtitan.experiments.jax import distributed
from torchtitan.experiments.jax import jax_profiling
from torchtitan.experiments.jax import metrics as jax_metrics
from torchtitan.experiments.jax import splash_attn
from torchtitan.experiments.jax import llama3 as jax_llama3
import torchtitan.tools.logging


logger = torchtitan.tools.logging.logger

P = jax.sharding.PartitionSpec


class JaxTrainer:

    def __init__(
        self,
        mesh: jax.sharding.Mesh,
        job_config: Any,
        accelerator: str,
        num_global_devices: int,
        num_local_devices: int,
    ):
        self.mesh = mesh
        self.job_config = job_config
        self.accelerator = accelerator
        self.num_global_devices = num_global_devices
        self.num_local_devices = num_local_devices

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def setup_dataloader(self, job_config):
        """Build a dataloader that returns (inputs, labels) numpy arrays."""
        expected_global_batch_size = job_config.training.global_batch_size
        if expected_global_batch_size <= 0:
            expected_global_batch_size = job_config.training.local_batch_size

        if (
            job_config.training.dataset is None
            or job_config.training.dataset.startswith('fake')
        ):
            train_loader = data_utils.fake_dataloader(
                job_config.training.steps,
                job_config.training.seq_len,
                expected_global_batch_size,
            )
            logger.warning('Using fake data loader.')
            return train_loader

        # Real dataset via torchtitan's data pipeline.
        import torchtitan.protocols.train_spec as ts_mod
        train_spec = ts_mod.get_train_spec('llama3')

        tokenizer = (
            train_spec.build_tokenizer_fn(job_config)
            if train_spec.build_tokenizer_fn is not None
            else None
        )
        original_local = job_config.training.local_batch_size
        job_config.training.local_batch_size = expected_global_batch_size
        torch_loader = train_spec.build_dataloader_fn(
            dp_world_size=1, dp_rank=0,
            tokenizer=tokenizer,
            job_config=job_config,
        )
        job_config.training.local_batch_size = original_local
        return data_utils.torch_loader_to_jax(torch_loader)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def setup_model(self, job_config):
        """Instantiate and shard the Flax NNX model."""
        model_name = job_config.model.name   # e.g. 'llama3'
        model_flavor = job_config.model.flavor  # e.g. '8B'
        jax_config = job_config.jax_config
        use_scan = jax_config.use_scan

        if model_name == 'llama3':
            jax_model_mod = jax_llama3
        else:
            raise ValueError(f'Unsupported model: {model_name}')

        model_args = jax_model_mod.args[model_flavor]
        model_args.max_seq_len = job_config.training.seq_len

        if jax_config.model_layer_override is not None:
            logger.warning(
                'Overriding n_layers from %d to %d.',
                model_args.n_layers, jax_config.model_layer_override,
            )
            model_args.n_layers = jax_config.model_layer_override

        # Choose activation checkpoint policy.
        ac_mode = job_config.activation_checkpoint.mode
        if ac_mode == 'full':
            checkpoint_policy = jax.checkpoint_policies.nothing_saveable
        elif ac_mode == 'selective':
            checkpoint_policy = jax.checkpoint_policies.dots_saveable
        elif ac_mode == 'nothing':
            checkpoint_policy = None
        else:
            checkpoint_policy = jax.checkpoint_policies.nothing_saveable

        # Build splash attention if on TPU.
        devices = jax.devices()
        platform = devices[0].platform if devices else 'cpu'
        attn_fn = None
        if platform == 'tpu':
            attn_fn = splash_attn.make_splash_attention_fn(
                self.mesh, jax_config
            )

        sharding_map = (
            jax_model_mod.sharding_map_scan if use_scan
            else jax_model_mod.sharding_map_original
        )

        logger.info('Building model %s-%s (scan=%s) ...', model_name, model_flavor, use_scan)

        # Instantiate on CPU so we don't OOM before sharding.
        with jax.default_device(jax.devices('cpu')[0]):
            model = jax_model_mod.Transformer(
                model_args,
                use_scan=use_scan,
                attn_fn=attn_fn,
                checkpoint_policy=checkpoint_policy,
                rngs=nnx.Rngs(0),
            )

        # Extract state, apply sharding, merge back.
        graphdef, state = nnx.split(model)
        logger.info('Applying sharding to %d parameter tensors ...', len(jax.tree_util.tree_leaves(state)))
        state = distributed.apply_sharding_to_state(state, sharding_map, self.mesh)
        model = nnx.merge(graphdef, state)

        self.model_args = model_args
        return model

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def setup_optimizer(self, job_config, model):
        opt_name = job_config.optimizer.name.lower()
        lr = job_config.optimizer.lr
        b1 = job_config.optimizer.beta1 or 0.9
        b2 = job_config.optimizer.beta2 or 0.95
        eps = job_config.optimizer.eps or 1e-8
        wd = job_config.optimizer.weight_decay or 0.1

        if opt_name == 'sgd':
            tx = optax.sgd(lr)
        elif opt_name == 'adam':
            tx = optax.adam(lr, b1=b1, b2=b2, eps=eps)
        elif opt_name == 'adamw':
            tx = optax.adamw(lr, b1=b1, b2=b2, eps=eps, weight_decay=wd)
        else:
            raise ValueError(f'Unsupported optimizer: {opt_name}')

        return nnx.Optimizer(model, tx, wrt=nnx.Param)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        job_config = self.job_config
        model = self.setup_model(job_config)
        optimizer = self.setup_optimizer(job_config, model)

        train_loader = self.setup_dataloader(job_config)

        metrics_processor = jax_metrics.JaxMetricsProcessor(
            job_config,
            self.accelerator,
            self.num_global_devices,
            tpu_megacore=job_config.jax_config.tpu_megacore,
            log_freq=job_config.metrics.log_freq,
        )
        # Approximate flops: 6 * n_params * seq_len (dense model estimate).
        n_params = sum(
            x.size for x in jax.tree_util.tree_leaves(
                nnx.state(model, nnx.Param)
            )
        )
        metrics_processor.num_flops_per_token = 6 * n_params

        @nnx.jit
        def train_step(model, optimizer: nnx.Optimizer, inputs, labels):
            def loss_fn(model):
                logits = model(inputs)  # [B, S, vocab]
                # Flatten for cross-entropy.
                B, S, V = logits.shape
                logits_2d = logits.reshape(B * S, V)
                labels_1d = labels.reshape(B * S)
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits_2d, labels_1d
                ).sum()
                return loss

            loss, grads = nnx.value_and_grad(
                loss_fn,
                argnums=nnx.DiffState(0, nnx.Param),
            )(model)
            optimizer.update(model, grads)
            return loss

        logger.info('Starting training loop ...')

        with jax_profiling.maybe_enable_profiling(
            job_config.profiling,
            global_step=0,
            base_folder=job_config.job.dump_folder,
        ) as profiler:
            step = -1
            for inputs_np, labels_np in train_loader:
                step += 1
                if step >= job_config.training.steps:
                    break

                data_load_start = time.perf_counter()
                inputs = jnp.array(inputs_np, dtype=jnp.int32)
                labels = jnp.array(labels_np, dtype=jnp.int32)

                # Shard batch on FSDP axis.
                inputs = distributed.shard_input(
                    inputs, self.mesh, self.num_global_devices, self.num_local_devices
                )
                labels = distributed.shard_input(
                    labels, self.mesh, self.num_global_devices, self.num_local_devices
                )

                ntokens = labels.size
                metrics_processor.ntokens_since_last_log += ntokens
                metrics_processor.data_loading_times.append(
                    time.perf_counter() - data_load_start
                )

                with jax.named_scope('train_step'):
                    loss = train_step(model, optimizer, inputs, labels)

                if profiler is not None:
                    jax.block_until_ready(loss)
                    profiler.step()

                if metrics_processor.should_log(step + 1):
                    jax.block_until_ready(loss)
                    metrics_processor.log(step + 1, float(loss) / ntokens)

        return True
