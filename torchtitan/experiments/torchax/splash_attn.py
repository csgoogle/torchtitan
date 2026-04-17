"""Torchax wrapper around the shared JAX splash-attention builder.

The actual kernel construction lives in
``torchtitan.experiments.jax.splash_attn``; this module just plugs the
resulting callable into torchax's scaled_dot_product_attention op override.
"""

import functools

import jax
import torch
import torchax

from torchtitan.experiments.jax.splash_attn import build_splash_attention_callable
import torchtitan.tools.logging


P = jax.sharding.PartitionSpec
logger = torchtitan.tools.logging.logger


def tpu_splash_attention(
    mesh,
    q_sharding,
    apply_shard_map,
    torchax_config,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    decoder_segment_ids: jax.Array | None,
    attn_logits_soft_cap: float | None = None,
) -> jax.Array:
  """Build and invoke TPU Splash Attention for one call.

  Thin wrapper kept for back-compat; prefer caching
  ``build_splash_attention_callable`` at setup time.
  """
  attn_fn = build_splash_attention_callable(
      mesh,
      q_sharding,
      torchax_config,
      apply_shard_map=apply_shard_map,
      attn_logits_soft_cap=attn_logits_soft_cap,
  )
  return attn_fn(query, key, value, decoder_segment_ids)


def declare_splash_attention(env, mesh, torchax_config):
  """Override torch's scaled_dot_product_attention with TPU Splash Attention.

  Args:
    env: torchax environment object used to override op definitions.
    mesh: JAX device mesh for sharding.
    torchax_config: TorchaxConfig with sa_* fields.

  Returns:
    True on success, False otherwise.
  """
  try:
    q_sharding = P('fsdp', 'tp', None, None)
    attention = build_splash_attention_callable(
        mesh, q_sharding, torchax_config, apply_shard_map=True
    )

    def custom_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
      #  batch, num of head, seq, dim
      jk, jq, jv = torchax.interop.jax_view((query, key, value))
      res = attention(jk, jq, jv, None)
      return torchax.interop.torch_view(res)

    env.override_op_definition(
        torch.nn.functional.scaled_dot_product_attention, custom_attention
    )
    return True
  except Exception as e:
    logger.error('Failed to declare splash attention: %s', e)
    return False
