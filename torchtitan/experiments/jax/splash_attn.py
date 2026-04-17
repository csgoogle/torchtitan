"""Splash Attention kernel setup shared by jax/ and torchax/.

- ``build_splash_attention_callable`` constructs a jitted splash-attention
  callable with a given mesh / q-sharding / config. torchax/splash_attn.py
  uses this under the hood when overriding torch's SDPA op.
- ``make_splash_attention_fn`` is the jax-experiment convenience wrapper
  (hardcodes ``P('fsdp', 'tp', None, None)`` for q-sharding).
"""

import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

import torchtitan.tools.logging

logger = torchtitan.tools.logging.logger

P = jax.sharding.PartitionSpec


def _build_block_sizes(config, query_shape, key_shape):
    return splash_attention_kernel.BlockSizes(
        block_q=min(config.sa_block_q, query_shape[2]),
        block_kv=min(config.sa_block_kv, key_shape[2]),
        block_kv_compute=min(config.sa_block_kv_compute, key_shape[2]),
        block_q_dkv=min(config.sa_block_q_dkv, query_shape[2]),
        block_kv_dkv=min(config.sa_block_kv_dkv, key_shape[2]),
        block_kv_dkv_compute=min(config.sa_block_kv_dkv_compute, query_shape[2]),
        block_q_dq=None if config.sa_use_fused_bwd_kernel else min(
            config.sa_block_q_dq, query_shape[2]
        ),
        block_kv_dq=None if config.sa_use_fused_bwd_kernel else min(
            config.sa_block_kv_dq, query_shape[2]
        ),
        use_fused_bwd_kernel=config.sa_use_fused_bwd_kernel,
        q_layout=splash_attention_kernel.QKVLayout[config.sa_q_layout],
        k_layout=splash_attention_kernel.QKVLayout[config.sa_k_layout],
        v_layout=splash_attention_kernel.QKVLayout[config.sa_v_layout],
    )


def build_splash_attention_callable(
    mesh,
    q_sharding,
    config,
    *,
    apply_shard_map: bool = True,
    attn_logits_soft_cap: float | None = None,
):
    """Return a jitted, sharded splash-attention callable.

    Signature of the returned callable:
        attn_fn(query, key, value, segment_ids) -> jax.Array
    where tensors are [batch, n_heads, seq, head_dim].
    """

    def _wrap_flash_attention(query, key, value, segment_ids):
        if segment_ids is not None:
            segment_ids = splash_attention_kernel.SegmentIds(
                segment_ids, segment_ids
            )
            assert query.shape[2] == segment_ids.q.shape[1], (
                'Sharding along sequence dimension not allowed in tpu kernel attention'
            )
        block_sizes = _build_block_sizes(config, query.shape, key.shape)
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
            attn_logits_soft_cap=attn_logits_soft_cap,
        )
        return jax.vmap(kernel)(query, key, value, segment_ids=segment_ids)

    fn = _wrap_flash_attention
    if apply_shard_map:
        fn = jax.experimental.shard_map.shard_map(
            fn,
            mesh=mesh,
            in_specs=(q_sharding, q_sharding, q_sharding, None),
            out_specs=q_sharding,
            check_rep=False,
        )
    return jax.jit(fn)


def make_splash_attention_fn(mesh, jax_config):
    """Build a splash-attention callable for the jax experiment.

    Returns ``None`` on failure (caller should fall back to default SDPA).
    """
    try:
        q_sharding = P('fsdp', 'tp', None, None)
        attn_fn = build_splash_attention_callable(
            mesh, q_sharding, jax_config, apply_shard_map=True
        )
        logger.info('Splash attention kernel enabled.')
        return attn_fn
    except Exception as e:
        logger.error(
            'Failed to build splash attention: %s — falling back to SDPA.', e
        )
        return None
