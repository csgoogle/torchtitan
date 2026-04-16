"""Sharding maps for Llama3 JAX model.

Keys match Flax NNX parameter paths (the GetAttrKey('.value') trailing segment
is stripped in distributed.py before matching).

Examples:
  tok_embeddings/embedding    → the embedding matrix
  layers/0/attention/wq/kernel → query weight for layer 0 (non-scan)
  layers/attention_wq_kernel   → stacked query weight (scan)

Partition specs: tuples of axis names ('fsdp', 'tp') or None.
Empty tuple () = replicated.  None = not sharded on that axis.
"""

# ---------------------------------------------------------------------------
# Non-scan: one entry per layer (layers is a Python list of TransformerBlocks).
# NNX stores them under integer keys: layers/0/attention/wq/kernel etc.
# ---------------------------------------------------------------------------

sharding_map_original = {
    # Embedding replicated to avoid gather-with-sharded-table issues in JAX.
    'tok_embeddings/embedding': (),
    'layers/*/attention/wq/kernel': ('fsdp', None),
    'layers/*/attention/wk/kernel': ('fsdp', None),
    'layers/*/attention/wv/kernel': ('fsdp', None),
    'layers/*/attention/wo/kernel': (None, 'fsdp'),
    'layers/*/feed_forward/w1/kernel': ('fsdp', None),
    'layers/*/feed_forward/w2/kernel': (None, 'fsdp'),
    'layers/*/feed_forward/w3/kernel': ('fsdp', None),
    'layers/*/attention_norm/weight': ('fsdp',),
    'layers/*/ffn_norm/weight': ('fsdp',),
    'norm/weight': ('fsdp',),
    'output/kernel': ('fsdp', None),
    'freqs_cis': (),  # RoPE table — replicated on all devices
}


# ---------------------------------------------------------------------------
# Scan: ScannedTransformerBlocks stores each stacked parameter as a named
# nnx.Param attribute derived from the block path (e.g. attention/wq/kernel
# → attribute name attention_wq_kernel). The leading None covers the layer
# axis added by stacking.
# ---------------------------------------------------------------------------

sharding_map_scan = {
    # Embedding replicated to avoid gather-with-sharded-table issues in JAX.
    'tok_embeddings/embedding': (),
    # Stacked weights: first axis = layer index (unsharded), rest = weight dims.
    'layers/attention_wq_kernel': (None, 'fsdp', None),
    'layers/attention_wk_kernel': (None, 'fsdp', None),
    'layers/attention_wv_kernel': (None, 'fsdp', None),
    'layers/attention_wo_kernel': (None, None, 'fsdp'),
    'layers/feed_forward_w1_kernel': (None, 'fsdp', None),
    'layers/feed_forward_w2_kernel': (None, None, 'fsdp'),
    'layers/feed_forward_w3_kernel': (None, 'fsdp', None),
    'layers/attention_norm_weight': (None, 'fsdp'),
    'layers/ffn_norm_weight': (None, 'fsdp'),
    'norm/weight': ('fsdp',),
    'output/kernel': ('fsdp', None),
    'freqs_cis': (),  # RoPE table — replicated on all devices
}
