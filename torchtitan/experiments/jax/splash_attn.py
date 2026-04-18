"""Splash Attention kernel setup for JAX experiment.

Identical to the torchax version but without the torchax env dependency —
returns a plain JAX callable instead of overriding a torch op.
"""

import functools

import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
import torchtitan.tools.logging

logger = torchtitan.tools.logging.logger

P = jax.sharding.PartitionSpec


def make_splash_attention_fn(mesh, jax_config):
    """Build a splash-attention callable for the given config and mesh.

    Args:
        mesh: JAX device mesh.
        jax_config: JaxConfig with sa_block_* fields.

    Returns:
        attn_fn(query, key, value, segment_ids) -> jax.Array
        where tensors are shaped [batch, n_heads, seq, head_dim].
        Returns None if construction fails (falls back to default SDPA).
    """
    try:
        q_sharding = P('fsdp', 'tp', None, None)

        def _wrap_flash_attention(query, key, value, segment_ids):
            if segment_ids is not None:
                segment_ids = splash_attention_kernel.SegmentIds(
                    segment_ids, segment_ids
                )
            block_sizes = splash_attention_kernel.BlockSizes(
                block_q=min(jax_config.sa_block_q, query.shape[2]),
                block_kv=min(jax_config.sa_block_kv, key.shape[2]),
                block_kv_compute=min(jax_config.sa_block_kv_compute, key.shape[2]),
                block_q_dkv=min(jax_config.sa_block_q_dkv, query.shape[2]),
                block_kv_dkv=min(jax_config.sa_block_kv_dkv, key.shape[2]),
                block_kv_dkv_compute=min(
                    jax_config.sa_block_kv_dkv_compute, query.shape[2]
                ),
                block_q_dq=None if jax_config.sa_use_fused_bwd_kernel else min(
                    jax_config.sa_block_q_dq, query.shape[2]
                ),
                block_kv_dq=None if jax_config.sa_use_fused_bwd_kernel else min(
                    jax_config.sa_block_kv_dq, query.shape[2]
                ),
                use_fused_bwd_kernel=jax_config.sa_use_fused_bwd_kernel,
                q_layout=splash_attention_kernel.QKVLayout[jax_config.sa_q_layout],
                k_layout=splash_attention_kernel.QKVLayout[jax_config.sa_k_layout],
                v_layout=splash_attention_kernel.QKVLayout[jax_config.sa_v_layout],
            )
            mask = splash_attention_mask.CausalMask(
                shape=(query.shape[2], query.shape[2])
            )
            multi_head_mask = splash_attention_mask.MultiHeadMask(
                masks=(mask,) * query.shape[1]
            )
            kernel = splash_attention_kernel.make_splash_mha(
                mask=multi_head_mask,
                head_shards=1,
                q_seq_shards=1,
                block_sizes=block_sizes,
            )
            return jax.vmap(kernel)(query, key, value, segment_ids=segment_ids)

        sharded_attn = jax.experimental.shard_map.shard_map(
            _wrap_flash_attention,
            mesh=mesh,
            in_specs=(q_sharding, q_sharding, q_sharding, None),
            out_specs=q_sharding,
            check_rep=False,
        )
        attn_fn = jax.jit(sharded_attn)
        logger.info('Splash attention kernel enabled.')
        return attn_fn
    except Exception as e:
        logger.error('Failed to build splash attention: %s — falling back to SDPA.', e)
        return None
