"""Locate the parts of a HuggingFace model that torchtitan needs by name.

`HFTransformerModel` exposes `tok_embeddings` / `layers` / `norm` / `lm_head`
so the rest of torchtitan can find them. Those properties used to hardcode the
Llama nesting (`model.model.layers`), so anything shaped differently raised
`AttributeError` before the first step -- OPT keeps its blocks at
`model.decoder.layers`, GPT-2 at `transformer.h`.

This module resolves each part in two passes: a table of known nestings, then
a structural fallback that looks for the thing itself (the longest ModuleList
of same-class blocks is the decoder stack; the sole nn.Embedding under the
backbone is the token embedding). The fallback is what lets an architecture
nobody enumerated still work.
"""

from torch import nn

# Ordered attribute chains, most common first. Each entry is tried with
# getattr from the wrapped HF model.
_LAYER_PATHS = (
    ("model", "layers"),            # Llama, Qwen, Mistral, Gemma
    ("model", "decoder", "layers"), # OPT, BART decoder
    ("transformer", "h"),           # GPT-2, Falcon
    ("decoder", "layers"),
    ("model", "encoder", "layers"),
    ("encoder", "layer"),           # BERT family
    ("layers",),
)
_EMBED_PATHS = (
    ("model", "embed_tokens"),
    ("model", "decoder", "embed_tokens"),
    ("transformer", "wte"),
    ("embeddings", "word_embeddings"),
    ("model", "embeddings", "word_embeddings"),
    ("shared",),
)
_NORM_PATHS = (
    ("model", "norm"),
    ("model", "decoder", "final_layer_norm"),
    ("transformer", "ln_f"),
    ("model", "final_layernorm"),
    ("norm",),
)
_HEAD_PATHS = (
    ("lm_head",),
    ("cls", "predictions", "decoder"),
    ("score",),
    ("qa_outputs",),
    ("classifier",),
)


def _walk(root, path):
    obj = root
    for part in path:
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _resolve(root, paths):
    for path in paths:
        found = _walk(root, path)
        if found is not None:
            return found, path
    return None, None


def find_layers(root):
    """The decoder/encoder block stack.

    Falls back to the longest ModuleList/ModuleDict whose entries share a
    class -- that is what a transformer block stack looks like structurally,
    whatever it is called.
    """
    found, path = _resolve(root, _LAYER_PATHS)
    if found is not None:
        return found, path

    best, best_path = None, None
    for name, mod in root.named_modules():
        if not isinstance(mod, (nn.ModuleList, nn.ModuleDict)):
            continue
        entries = list(mod.values()) if isinstance(mod, nn.ModuleDict) else list(mod)
        if len(entries) < 2:
            continue
        if len({type(e) for e in entries}) != 1:
            continue
        if best is None or len(entries) > len(best):
            best, best_path = mod, tuple(name.split("."))
    return best, best_path


def find_embedding(root):
    found, path = _resolve(root, _EMBED_PATHS)
    if found is not None:
        return found, path
    # Structural fallback: the widest nn.Embedding is the token table.
    best, best_path = None, None
    for name, mod in root.named_modules():
        if isinstance(mod, nn.Embedding):
            if best is None or mod.num_embeddings > best.num_embeddings:
                best, best_path = mod, tuple(name.split("."))
    return best, best_path


def find_norm(root):
    return _resolve(root, _NORM_PATHS)


def find_head(root):
    return _resolve(root, _HEAD_PATHS)


def set_at(root, path, value):
    """Assign back through a path returned by one of the finders."""
    if not path:
        raise AttributeError("no resolved path to assign through")
    obj = root
    for part in path[:-1]:
        obj = getattr(obj, part)
    setattr(obj, path[-1], value)


def find_backbone(root):
    """The module that owns the block stack (model / transformer / decoder).

    Derived from wherever the stack was found, so it needs no separate table.
    """
    _, path = find_layers(root)
    if not path or len(path) < 2:
        return None, None
    return _walk(root, path[:-1]), path[:-1]


def find_rotary(root):
    """Rotary embedding module, or (None, None) for learned-position models."""
    backbone, bpath = find_backbone(root)
    for holder, hpath in ((backbone, bpath), (root, ())):
        if holder is not None and hasattr(holder, "rotary_emb"):
            return getattr(holder, "rotary_emb"), tuple(hpath) + ("rotary_emb",)
    return None, None
