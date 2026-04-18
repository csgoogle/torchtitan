import jax
import torch
import functools
import torchax
import torchtitan.tools.logging

from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask


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
  """TPU Flash Attention."""
  if decoder_segment_ids is not None:
    decoder_segment_ids = splash_attention_kernel.SegmentIds(
        decoder_segment_ids, decoder_segment_ids)

  def wrap_flash_attention(query, key, value, decoder_segment_ids):
    if decoder_segment_ids is not None:
      assert (
          query.shape[2] == decoder_segment_ids.q.shape[1]
      ), "Sharding along sequence dimension not allowed in tpu kernel attention"
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=min(torchax_config.sa_block_q, query.shape[2]),
        block_kv=min(torchax_config.sa_block_kv, key.shape[2]),
        block_kv_compute=min(torchax_config.sa_block_kv_compute, key.shape[2]),
        block_q_dkv=min(torchax_config.sa_block_q_dkv, query.shape[2]),
        block_kv_dkv=min(torchax_config.sa_block_kv_dkv, key.shape[2]),
        block_kv_dkv_compute=min(torchax_config.sa_block_kv_dkv_compute, query.shape[2]),
        block_q_dq=None if torchax_config.sa_use_fused_bwd_kernel else min(
            torchax_config.sa_block_q_dq, query.shape[2]),
        block_kv_dq=None if torchax_config.sa_use_fused_bwd_kernel else min(
            torchax_config.sa_block_kv_dq, query.shape[2]),
        use_fused_bwd_kernel=torchax_config.sa_use_fused_bwd_kernel,
        q_layout=splash_attention_kernel.QKVLayout[torchax_config.sa_q_layout],
        k_layout=splash_attention_kernel.QKVLayout[torchax_config.sa_k_layout],
        v_layout=splash_attention_kernel.QKVLayout[torchax_config.sa_v_layout],
    )

    mask = splash_attention_mask.CausalMask(
        shape=(query.shape[2], query.shape[2]))

    # Create multi-head mask
    multi_head_mask = splash_attention_mask.MultiHeadMask(
        masks=(mask,) * query.shape[1])
    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        head_shards=1,
        q_seq_shards=1,
        block_sizes=block_sizes,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    return jax.vmap(splash_kernel)(
        query, key, value, segment_ids=decoder_segment_ids)

  if apply_shard_map:
    wrap_flash_attention = jax.experimental.shard_map.shard_map(
        wrap_flash_attention,
        mesh=mesh,
        in_specs=(
            q_sharding,
            q_sharding,
            q_sharding,
            None,
        ),
        out_specs=q_sharding,
        check_rep=False,
    )

  x = wrap_flash_attention(query, key, value, decoder_segment_ids)
  return x


def declare_splash_attention(env, mesh, torchax_config):
  """Overrides torch's scaled_dot_product_attention with a TPU Splash Attention.

  This function attempts to replace the default
  `torch.nn.functional.scaled_dot_product_attention` with a JAX-based
  TPU-optimized Splash Attention kernel.

  Args:
    env: The environment object used to override op definitions.
    mesh: The JAX device mesh for sharding.
    torchax_config: The TorchaxConfig object with block sizes.

  Returns:
    True if the override was successful, False otherwise.
  """

  try:
    partition = P('fsdp', 'tp', None, None)
    attention = functools.partial(
        tpu_splash_attention, mesh, partition, True, torchax_config
    )
    attention = jax.jit(attention)

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
