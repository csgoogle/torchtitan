"""Sharding map for Conformer model in torchax."""

# Regex patterns to match parameter names in conformer state_dict
sharding_map_original = {
    r".*embedding\.weight": ("fsdp", "tp"),

    # FeedForward Modules
    r".*ffn1\.sequential\.0\.weight": ("fsdp",),
    r".*ffn1\.sequential\.0\.bias": ("fsdp",),
    r".*ffn1\.sequential\.1\.weight": ("tp", "fsdp"),
    r".*ffn1\.sequential\.1\.bias": ("tp",),
    r".*ffn1\.sequential\.4\.weight": ("fsdp", "tp"),
    r".*ffn1\.sequential\.4\.bias": ("fsdp",),

    r".*ffn2\.sequential\.0\.weight": ("fsdp",),
    r".*ffn2\.sequential\.0\.bias": ("fsdp",),
    r".*ffn2\.sequential\.1\.weight": ("tp", "fsdp"),
    r".*ffn2\.sequential\.1\.bias": ("tp",),
    r".*ffn2\.sequential\.4\.weight": ("fsdp", "tp"),
    r".*ffn2\.sequential\.4\.bias": ("fsdp",),

    # Self Attention Layer Norm
    r".*self_attn_layer_norm\.weight": ("fsdp",),
    r".*self_attn_layer_norm\.bias": ("fsdp",),

    # Attention Projections (PatchedMHA)
    r".*self_attn\.in_proj\.weight": ("tp", "fsdp"),
    r".*self_attn\.in_proj\.bias": ("tp",),
    r".*self_attn\.out_proj\.weight": ("fsdp", "tp"),
    r".*self_attn\.out_proj\.bias": ("fsdp",),

    # Convolution Module
    r".*conv_module\.layer_norm\.weight": ("fsdp",),
    r".*conv_module\.layer_norm\.bias": ("fsdp",),
    r".*conv_module\.sequential\.0\.weight": ("tp", "fsdp", None),
    r".*conv_module\.sequential\.0\.bias": ("tp",),
    r".*conv_module\.sequential\.2\.weight": ("fsdp", None, None),  # depthwise
    r".*conv_module\.sequential\.2\.bias": ("fsdp",),
    r".*conv_module\.sequential\.3\.weight": ("fsdp",),  # BatchNorm1d
    r".*conv_module\.sequential\.3\.bias": ("fsdp",),
    r".*conv_module\.sequential\.3\.running_mean": ("fsdp",),
    r".*conv_module\.sequential\.3\.running_var": ("fsdp",),
    r".*conv_module\.sequential\.3\.num_batches_tracked": (),
    r".*conv_module\.sequential\.5\.weight": ("fsdp", "tp", None),
    r".*conv_module\.sequential\.5\.bias": ("fsdp",),

    # Final Layer Norm
    r".*final_layer_norm\.weight": ("fsdp",),
    r".*final_layer_norm\.bias": ("fsdp",),

    # FC classification layer
    r"fc\.weight": ("tp", "fsdp"),
    r"fc\.bias": ("tp",),
}

sharding_map_scan = {
    "tok_embeddings.weight": (),
    "output.weight": (),
    "output.bias": (),

    # FeedForward 1
    "layers.params.ffn1___sequential___0___weight": (),
    "layers.params.ffn1___sequential___0___bias": (),
    "layers.params.ffn1___sequential___1___weight": (),
    "layers.params.ffn1___sequential___1___bias": (),
    "layers.params.ffn1___sequential___4___weight": (),
    "layers.params.ffn1___sequential___4___bias": (),

    # Self Attention
    "layers.params.self_attn_layer_norm___weight": (),
    "layers.params.self_attn_layer_norm___bias": (),
    "layers.params.self_attn___in_proj___weight": (),
    "layers.params.self_attn___in_proj___bias": (),
    "layers.params.self_attn___out_proj___weight": (),
    "layers.params.self_attn___out_proj___bias": (),

    # Convolution Module
    "layers.params.conv_module___layer_norm___weight": (),
    "layers.params.conv_module___layer_norm___bias": (),
    "layers.params.conv_module___sequential___0___weight": (),
    "layers.params.conv_module___sequential___0___bias": (),
    "layers.params.conv_module___sequential___2___weight": (),
    "layers.params.conv_module___sequential___2___bias": (),
    "layers.params.conv_module___sequential___3___weight": (),
    "layers.params.conv_module___sequential___3___bias": (),
    "layers.conv_module___sequential___3___running_mean": (),
    "layers.conv_module___sequential___3___running_var": (),
    "layers.conv_module___sequential___3___num_batches_tracked": (),
    "layers.params.conv_module___sequential___5___weight": (),
    "layers.params.conv_module___sequential___5___bias": (),

    # FeedForward 2
    "layers.params.ffn2___sequential___0___weight": (),
    "layers.params.ffn2___sequential___0___bias": (),
    "layers.params.ffn2___sequential___1___weight": (),
    "layers.params.ffn2___sequential___1___bias": (),
    "layers.params.ffn2___sequential___4___weight": (),
    "layers.params.ffn2___sequential___4___bias": (),

    # Final Layer Norm
    "layers.params.final_layer_norm___weight": (),
    "layers.params.final_layer_norm___bias": (),
}
sharding_map_scan_moe = {}
