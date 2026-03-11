"""Sharding map for qwen3 model."""

sharding_map_original = {}

sharding_map_scan = {
    'rope_cache': (),
    'tok_embeddings.weight': ('tp', 'fsdp'),
    'norm.weight': ('fsdp',),
    'output.weight': ('tp', 'fsdp'),
    # dense layers
    'layers_dense.params.attention___wo___weight': (None, 'fsdp', 'tp'),
    'layers_dense.params.attention___wq___weight': (None, 'tp', 'fsdp'),
    'layers_dense.params.attention___kv_norm___weight': (None, 'fsdp'),
    'layers_dense.params.attention___wkv_a___weight': (None, 'tp', 'fsdp'),
    'layers_dense.params.attention___wkv_b___weight': (None, 'fsdp', 'tp'),
    'layers_dense.params.attention_norm___weight': (None, 'fsdp'),
    'layers_dense.params.attention___q_norm___weight': (None, 'fsdp'),
    'layers_dense.params.attention___wq_a___weight': (None, 'tp', 'fsdp'),
    'layers_dense.params.attention___wq_b___weight': (None, 'fsdp', 'tp'),
    'layers_dense.params.ffn_norm___weight': (None, 'fsdp'),
    'layers_dense.params.feed_forward___w1___weight': (None, 'tp', 'fsdp'),
    'layers_dense.params.feed_forward___w2___weight': (None, 'fsdp', 'tp'),
    'layers_dense.params.feed_forward___w3___weight': (None, 'tp', 'fsdp'),
    # moe layers
    'layers_moe.params.attention___wo___weight': (None, 'fsdp', 'tp'),
    'layers_moe.params.attention___wq___weight': (None, 'tp', 'fsdp'),
    'layers_moe.params.attention___kv_norm___weight': (None, 'fsdp'),
    'layers_moe.params.attention___wkv_a___weight': (None, 'tp', 'fsdp'),
    'layers_moe.params.attention___wkv_b___weight': (None, 'fsdp', 'tp'),
    'layers_moe.params.attention_norm___weight': (None, 'fsdp'),
    'layers_moe.params.attention___q_norm___weight': (None, 'fsdp'),
    'layers_moe.params.attention___wq_a___weight': (None, 'tp', 'fsdp'),
    'layers_moe.params.attention___wq_b___weight': (None, 'fsdp', 'tp'),
    'layers_moe.params.ffn_norm___weight': (None, 'fsdp'),
    'layers_moe.params.moe___expert_bias': (None, None),
    'layers_moe.params.moe___experts___w1': (None, None, 'tp', 'fsdp'),
    'layers_moe.params.moe___experts___w2': (None, None, 'fsdp', 'tp'),
    'layers_moe.params.moe___experts___w3': (None, None, 'tp', 'fsdp'),
    'layers_moe.params.moe___router___gate___weight': (None, None, 'fsdp'),
    'layers_moe.params.moe___shared_experts___w1___weight': (None, 'tp', 'fsdp'),
    'layers_moe.params.moe___shared_experts___w2___weight': (None, 'fsdp', 'tp'),
    'layers_moe.params.moe___shared_experts___w3___weight': (None, 'tp', 'fsdp'),
}

sharding_map_scan_moe = sharding_map_scan

