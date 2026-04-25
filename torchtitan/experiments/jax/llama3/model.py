"""Llama3 8B model in pure JAX using Flax NNX.

Architecture:
  - RMSNorm
  - Rotary Positional Embedding (RoPE)
  - Grouped Query Attention (GQA) with optional splash attention
  - SwiGLU FFN
  - Scan-based layer iteration via jax.lax.scan
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx


# ---------------------------------------------------------------------------
# Model args
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 128256
    ffn_dim_multiplier: float = 1.3
    multiple_of: int = 1024
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_seq_len: int = 8192

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def hidden_dim(self) -> int:
        hidden = int(4 * self.dim * 2 / 3)
        hidden = int(self.ffn_dim_multiplier * hidden)
        return self.multiple_of * ((hidden + self.multiple_of - 1) // self.multiple_of)

    @property
    def n_rep(self) -> int:
        """Number of times each KV head is repeated for GQA."""
        return self.n_heads // self.n_kv_heads


# Pre-defined configs
llama3_debug = ModelArgs(
    dim=64,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    vocab_size=2048,
    ffn_dim_multiplier=1.0,
    multiple_of=16,
)

llama3_8b = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=128256,
    ffn_dim_multiplier=1.3,
    multiple_of=1024,
    norm_eps=1e-5,
    rope_theta=500000.0,
)

llama3_70b = ModelArgs(
    dim=8192,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    vocab_size=128256,
    ffn_dim_multiplier=1.3,
    multiple_of=4096,
    norm_eps=1e-5,
    rope_theta=500000.0,
)


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------

def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> jax.Array:
    """Precompute RoPE frequency matrix.

    Returns:
        freqs_cis: [max_seq_len, head_dim // 2, 2] where last dim is (cos, sin).
    """
    freqs = 1.0 / (
        theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)  # [max_seq_len, head_dim // 2]
    return jnp.stack([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)  # [..., 2]


def apply_rotary_emb(
    x: jax.Array,
    freqs_cis: jax.Array,
) -> jax.Array:
    """Apply RoPE to query or key tensor.

    Args:
        x: [batch, seq_len, n_heads, head_dim]
        freqs_cis: [seq_len, head_dim // 2, 2] — (cos, sin) packed in last dim
    Returns:
        Rotated x with same shape.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]   # [batch, seq, heads, head_dim//2]
    x2 = x[..., half:]

    cos = freqs_cis[..., 0]  # [seq, head_dim//2]
    sin = freqs_cis[..., 1]

    # Broadcast: [1, seq, 1, head_dim//2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    rotated = jnp.concatenate(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1
    )
    return rotated.astype(x.dtype)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class RMSNorm(nnx.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        *,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        # Scale weight stored at the model's master-weight dtype. The forward
        # pass upcasts to fp32 for the rms math regardless, so bf16 storage
        # here is safe under training.dtype=bfloat16.
        self.weight = nnx.Param(jnp.ones(dim, dtype=param_dtype))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        # Cast to float32 for numerical stability, then back.
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps)
        return (x_f32 / rms * self.weight[...].astype(jnp.float32)).astype(x.dtype)


class Attention(nnx.Module):
    """Multi-head attention with Grouped Query Attention (GQA)."""

    def __init__(
        self,
        args: ModelArgs,
        attn_fn: Optional[Callable] = None,
        *,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = args.n_rep
        self.head_dim = args.head_dim

        self.wq = nnx.Linear(
            args.dim, args.n_heads * args.head_dim,
            use_bias=False, dtype=jnp.bfloat16, param_dtype=param_dtype,
            rngs=rngs,
        )
        self.wk = nnx.Linear(
            args.dim, args.n_kv_heads * args.head_dim,
            use_bias=False, dtype=jnp.bfloat16, param_dtype=param_dtype,
            rngs=rngs,
        )
        self.wv = nnx.Linear(
            args.dim, args.n_kv_heads * args.head_dim,
            use_bias=False, dtype=jnp.bfloat16, param_dtype=param_dtype,
            rngs=rngs,
        )
        self.wo = nnx.Linear(
            args.n_heads * args.head_dim, args.dim,
            use_bias=False, dtype=jnp.bfloat16, param_dtype=param_dtype,
            rngs=rngs,
        )
        # Optional override: splash attention or other kernel.
        self._attn_fn = attn_fn

    def __call__(
        self,
        x: jax.Array,
        freqs_cis: jax.Array,
    ) -> jax.Array:
        batch, seq_len, _ = x.shape

        q = self.wq(x).reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis[:seq_len])
        k = apply_rotary_emb(k, freqs_cis[:seq_len])

        if self._attn_fn is not None:
            # Splash attention supports GQA natively (num_kv_heads <
            # num_q_heads). Pass K/V with n_kv_heads directly; the kernel
            # handles the mapping internally, saving the ~4x materialized
            # K/V when we'd otherwise repeat to match n_heads.
            q_t = q.transpose(0, 2, 1, 3)  # [B, H_q, S, D]
            k_t = k.transpose(0, 2, 1, 3)  # [B, H_kv, S, D]
            v_t = v.transpose(0, 2, 1, 3)
            out = self._attn_fn(q_t, k_t, v_t, None)
            out = out.transpose(0, 2, 1, 3)  # [B, S, H_q, D]
        else:
            # Fallback path: no native GQA, so repeat K/V to match Q head count.
            k = jnp.repeat(k, self.n_rep, axis=2)
            v = jnp.repeat(v, self.n_rep, axis=2)
            # Standard scaled dot-product attention (causal).
            scale = 1.0 / math.sqrt(self.head_dim)
            # [B, S, H, D] → compute attn scores
            scores = jnp.einsum('bshd,bthd->bsht', q, k) * scale
            # Causal mask
            mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            scores = jnp.where(mask[None, :, None, :], scores, jnp.finfo(scores.dtype).min)
            attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
            out = jnp.einsum('bsht,bthd->bshd', attn, v)

        out = out.reshape(batch, seq_len, self.n_heads * self.head_dim)
        return self.wo(out)


class FeedForward(nnx.Module):
    """SwiGLU feed-forward network with separate gate, up, and down projections."""

    def __init__(
        self,
        args: ModelArgs,
        *,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.w1 = nnx.Linear(
            args.dim, args.hidden_dim,
            use_bias=False, dtype=jnp.bfloat16, param_dtype=param_dtype,
            rngs=rngs,
        )
        self.w2 = nnx.Linear(
            args.hidden_dim, args.dim,
            use_bias=False, dtype=jnp.bfloat16, param_dtype=param_dtype,
            rngs=rngs,
        )
        self.w3 = nnx.Linear(
            args.dim, args.hidden_dim,
            use_bias=False, dtype=jnp.bfloat16, param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        args: ModelArgs,
        attn_fn: Optional[Callable] = None,
        *,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.attention = Attention(
            args, attn_fn=attn_fn, param_dtype=param_dtype, rngs=rngs,
        )
        self.feed_forward = FeedForward(args, param_dtype=param_dtype, rngs=rngs)
        self.attention_norm = RMSNorm(
            args.dim, eps=args.norm_eps, param_dtype=param_dtype, rngs=rngs,
        )
        self.ffn_norm = RMSNorm(
            args.dim, eps=args.norm_eps, param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(self, x: jax.Array, freqs_cis: jax.Array) -> jax.Array:
        # Name the layer input so remat policies can target it for
        # save/offload. Harmless no-op when the remat policy doesn't reference it.
        x = jax.ad_checkpoint.checkpoint_name(x, "decoder_layer_input")
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Scan-based layer stack
# ---------------------------------------------------------------------------

def _get_block_param_paths(state: 'nnx.State') -> list[str]:
    """Return path strings (e.g. 'attention/wq/kernel') for each leaf in state.

    Strips the trailing '.value' GetAttrKey that comes from nnx.Variable's
    pytree registration.
    """
    paths: list[str] = []

    def _collect(path, leaf):
        parts = []
        for k in path:
            if isinstance(k, jax.tree_util.DictKey):
                parts.append(str(k.key))
            elif isinstance(k, jax.tree_util.GetAttrKey):
                if k.name != 'value':
                    parts.append(k.name)
            elif isinstance(k, jax.tree_util.SequenceKey):
                parts.append(str(k.idx))
            else:
                parts.append(str(k))
        paths.append('/'.join(p for p in parts if p))
        return leaf

    jax.tree_util.tree_map_with_path(_collect, state)
    return paths


class ScannedTransformerBlocks(nnx.Module):
    """Runs N transformer blocks using jax.lax.scan with stacked parameters.

    Each per-block parameter is stacked along axis 0 and stored as a named
    nnx.Param attribute (e.g. ``attention_wq_kernel``). This gives NNX full
    visibility for gradient computation and sharding while keeping the XLA
    graph compact.

    Implementation:
      1. Create all N blocks with independent random seeds.
      2. Flatten each block's NNX state; jax.tree_util leaves are raw arrays.
      3. Stack arrays along axis 0 and store as named nnx.Param attributes.
      4. In forward, collect arrays in canonical order, run jax.lax.scan.
         Each scan step unflatten → nnx.merge to get a single-layer block.
    """

    def __init__(
        self,
        args: ModelArgs,
        n_layers: int,
        attn_fn: Optional[Callable] = None,
        checkpoint_policy=None,
        *,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.n_layers = n_layers
        self.checkpoint_policy = checkpoint_policy

        # Reference block: defines graphdef (static structure).
        ref = TransformerBlock(
            args, attn_fn=attn_fn, param_dtype=param_dtype, rngs=rngs,
        )
        ref_graphdef, ref_state = nnx.split(ref)
        self._graphdef = ref_graphdef

        # Flatten: leaves are raw jax.Arrays (nnx.Variable is a pytree whose
        # leaf is the underlying value).
        ref_leaves, treedef = jax.tree_util.tree_flatten(ref_state)
        self._treedef = treedef

        # Derive attribute names from the block's state paths.
        # E.g. 'attention/wq/kernel' → 'attention_wq_kernel'.
        param_paths = _get_block_param_paths(ref_state)
        attr_names = [p.replace('/', '_') for p in param_paths]
        self._param_attr_names: list[str] = attr_names  # static Python list

        # Create remaining blocks and collect raw leaf values.
        all_blocks = [ref] + [
            TransformerBlock(
                args,
                attn_fn=attn_fn,
                param_dtype=param_dtype,
                rngs=nnx.Rngs(i),
            )
            for i in range(1, n_layers)
        ]
        all_values: list[list] = []
        for block in all_blocks:
            _, state = nnx.split(block)
            leaves, _ = jax.tree_util.tree_flatten(state)
            all_values.append(leaves)  # leaves are raw jax.Arrays

        # Stack each parameter across layers: shape [n_layers, *param_shape].
        n_params = len(ref_leaves)
        stacked = [
            jnp.stack([all_values[b][p] for b in range(n_layers)], axis=0)
            for p in range(n_params)
        ]

        # Store as named nnx.Param attributes so NNX tracks them for gradient
        # computation and sharding (no plain-list storage, which NNX rejects).
        for name, arr in zip(attr_names, stacked):
            setattr(self, name, nnx.Param(arr))

    def __call__(self, x: jax.Array, freqs_cis: jax.Array) -> jax.Array:
        graphdef = self._graphdef
        treedef = self._treedef
        checkpoint_policy = self.checkpoint_policy

        # Collect stacked arrays in the same canonical order as treedef.
        stacked_arrays = [getattr(self, name)[...] for name in self._param_attr_names]

        def scan_fn(carry, layer_arrays):
            # layer_arrays: list of single-layer arrays sliced by jax.lax.scan.
            # treedef.unflatten reconstructs an nnx.State with Param wrappers,
            # which nnx.merge uses to build a single TransformerBlock.
            layer_state = treedef.unflatten(layer_arrays)
            block = nnx.merge(graphdef, layer_state)
            return block(carry, freqs_cis), None

        if checkpoint_policy is not None:
            scan_fn = jax.checkpoint(scan_fn, policy=checkpoint_policy)

        x, _ = jax.lax.scan(scan_fn, x, stacked_arrays)
        return x


# ---------------------------------------------------------------------------
# Full Transformer
# ---------------------------------------------------------------------------

class Transformer(nnx.Module):
    def __init__(
        self,
        args: ModelArgs,
        use_scan: bool = True,
        attn_fn: Optional[Callable] = None,
        checkpoint_policy=None,
        *,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        # param_dtype is the master-weight storage dtype (driven by
        # job_config.training.dtype: float32 → fp32 master + bf16 compute,
        # bfloat16 → fully-bf16 model). Compute dtype stays bf16 in both
        # cases; when param_dtype=bf16 this collapses to a pure-bf16 model.
        self.args = args
        self.tok_embeddings = nnx.Embed(
            num_embeddings=args.vocab_size,
            features=args.dim,
            dtype=jnp.bfloat16,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        if use_scan:
            self.layers = ScannedTransformerBlocks(
                args,
                n_layers=args.n_layers,
                attn_fn=attn_fn,
                checkpoint_policy=checkpoint_policy,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        else:
            self.layers = nnx.List([
                TransformerBlock(
                    args, attn_fn=attn_fn, param_dtype=param_dtype,
                    rngs=nnx.Rngs(i),
                )
                for i in range(args.n_layers)
            ])
        self.norm = RMSNorm(
            args.dim, eps=args.norm_eps, param_dtype=param_dtype, rngs=rngs,
        )
        self.output = nnx.Linear(
            args.dim, args.vocab_size,
            use_bias=False, dtype=jnp.bfloat16, param_dtype=param_dtype,
            rngs=rngs,
        )
        # Precomputed RoPE table — replicated across all devices.
        self.freqs_cis = precompute_freqs_cis(
            args.head_dim, args.max_seq_len, args.rope_theta
        )

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            tokens: [batch, seq_len] int32 token ids.
        Returns:
            logits: [batch, seq_len, vocab_size] float32.
        """
        # nnx.Embed uses a gather under the hood.  When the embedding table is
        # replicated and tokens are batch-sharded, JAX cannot infer the output
        # sharding automatically.  Work around this by temporarily replicating
        # tokens before the lookup, then re-sharding the result on the batch
        # axis so the rest of the network sees batch-parallel activations.
        tokens_rep = jax.lax.with_sharding_constraint(
            tokens, jax.sharding.PartitionSpec()
        )
        x = self.tok_embeddings(tokens_rep)  # [B, S, dim], replicated
        x = x.astype(jnp.bfloat16)
        x = jax.lax.with_sharding_constraint(
            x, jax.sharding.PartitionSpec('fsdp', None, None)
        )
        freqs_cis = self.freqs_cis[:x.shape[1]]

        if isinstance(self.layers, ScannedTransformerBlocks):
            x = self.layers(x, freqs_cis)
        else:
            for layer in self.layers:
                x = layer(x, freqs_cis)

        x = self.norm(x)
        # Keep logits in compute dtype (bf16). Was .astype(fp32) which pins
        # a f32[B,S,V] copy live through backward. Downstream loss uses a
        # max-stabilized bf16 logsumexp which is numerically adequate.
        return self.output(x)
