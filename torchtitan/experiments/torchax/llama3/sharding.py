"""Sharding map for llama3 model."""

sharding_map_original = {
    'freqs_cis': (),
    'tok_embeddings.weight': ('fsdp', 'tp'),
    'layers.*.attention.wo.weight': ('fsdp', 'tp'),
    'layers.*.attention.wq.weight': ('tp', 'fsdp'),
    'layers.*.attention.wk.weight': ('tp', 'fsdp'),
    'layers.*.attention.wv.weight': ('tp', 'fsdp'),
    'layers.*.feed_forward.w1.weight': ('tp', 'fsdp'),
    'layers.*.feed_forward.w2.weight': ('fsdp', 'tp'),
    'layers.*.feed_forward.w3.weight': ('tp', 'fsdp'),
    'layers.*.attention_norm.weight': ('fsdp',),
    'layers.*.ffn_norm.weight': ('fsdp',),
    'norm.weight': ('fsdp',),
    'output.weight': ('tp', 'fsdp'),
}


sharding_map_scan = {
    'freqs_cis': (),
    'tok_embeddings.weight': ('tp', 'fsdp'),
    'layers.params.attention___wo___weight': (None, 'fsdp', 'tp'),
    'layers.params.attention___wq___weight': (None, 'tp', 'fsdp'),
    'layers.params.attention___wk___weight': (None, 'tp', 'fsdp'),
    'layers.params.attention___wv___weight': (None, 'tp', 'fsdp'),
    'layers.params.feed_forward___w1___weight': (None, 'tp', 'fsdp'),
    'layers.params.feed_forward___w2___weight': (None, 'fsdp', 'tp'),
    'layers.params.feed_forward___w3___weight': (None, 'tp', 'fsdp'),
    'layers.params.attention_norm___weight': (None, 'fsdp'),
    'layers.params.ffn_norm___weight': (None, 'fsdp'),
    'norm.weight': ('fsdp',),
    'output.weight': ('tp', 'fsdp'),
}

sharding_map_scan_moe = {}
