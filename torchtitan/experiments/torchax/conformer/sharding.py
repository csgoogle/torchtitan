"""Sharding map for conformer model."""

sharding_map_original = {
    'embedding.weight': ('fsdp', 'tp'),
    'fc.weight': ('tp', 'fsdp'),
    'fc.bias': ('fsdp',),
    'conformer_layers.*.ffn1.sequential.0.weight': ('fsdp',),
    'conformer_layers.*.ffn1.sequential.0.bias': ('fsdp',),
    'conformer_layers.*.ffn1.sequential.1.weight': ('tp', 'fsdp'),
    'conformer_layers.*.ffn1.sequential.1.bias': ('tp',),
    'conformer_layers.*.ffn1.sequential.4.weight': ('fsdp', 'tp'),
    'conformer_layers.*.ffn1.sequential.4.bias': ('fsdp',),
    
    'conformer_layers.*.self_attn_layer_norm.weight': ('fsdp',),
    'conformer_layers.*.self_attn_layer_norm.bias': ('fsdp',),
    
    'conformer_layers.*.self_attn.in_proj.weight': ('tp', 'fsdp'),
    'conformer_layers.*.self_attn.in_proj.bias': ('tp',),
    'conformer_layers.*.self_attn.out_proj.weight': ('fsdp', 'tp'),
    'conformer_layers.*.self_attn.out_proj.bias': ('fsdp',),
    
    'conformer_layers.*.conv_module.layer_norm.weight': ('fsdp',),
    'conformer_layers.*.conv_module.layer_norm.bias': ('fsdp',),
    
    'conformer_layers.*.conv_module.sequential.0.weight': ('fsdp',),
    'conformer_layers.*.conv_module.sequential.0.bias': ('fsdp',),
    'conformer_layers.*.conv_module.sequential.2.weight': ('fsdp',),
    'conformer_layers.*.conv_module.sequential.2.bias': ('fsdp',),
    'conformer_layers.*.conv_module.sequential.3.weight': ('fsdp',),
    'conformer_layers.*.conv_module.sequential.3.bias': ('fsdp',),
    'conformer_layers.*.conv_module.sequential.3.running_mean': ('fsdp',),
    'conformer_layers.*.conv_module.sequential.3.running_var': ('fsdp',),
    'conformer_layers.*.conv_module.sequential.3.num_batches_tracked': (),
    'conformer_layers.*.conv_module.sequential.5.weight': ('fsdp',),
    'conformer_layers.*.conv_module.sequential.5.bias': ('fsdp',),

    'conformer_layers.*.ffn2.sequential.0.weight': ('fsdp',),
    'conformer_layers.*.ffn2.sequential.0.bias': ('fsdp',),
    'conformer_layers.*.ffn2.sequential.1.weight': ('tp', 'fsdp'),
    'conformer_layers.*.ffn2.sequential.1.bias': ('tp',),
    'conformer_layers.*.ffn2.sequential.4.weight': ('fsdp', 'tp'),
    'conformer_layers.*.ffn2.sequential.4.bias': ('fsdp',),
    
    'conformer_layers.*.final_layer_norm.weight': ('fsdp',),
    'conformer_layers.*.final_layer_norm.bias': ('fsdp',),
}

sharding_map_scan = sharding_map_original
