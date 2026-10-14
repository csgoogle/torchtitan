from typing import Any

import jax
from jax.experimental import shard_map
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from torch import nn
from torchtitan.tools.logging import logger


def _local_count_tokens_jax(experts_shard, num_experts):
  """Computes histogram of tokens per expert on the LOCAL shard only.

  Returns shape [1, num_experts] to match P('fsdp', None) rank requirements.
  """
  experts_flat = experts_shard.reshape(-1)
  counts = jnp.bincount(experts_flat, length=num_experts).astype(jnp.float32)
  # Expand dims to satisfy Rank 2 spec P('fsdp', None)
  return counts[None, :]


def _local_permute_jax(x_local, experts_local, top_scores_local, top_k):
  """Sorts and permutes tokens locally within the shard.

  x_local: [LocalBatch, Dim]
  """
  expert_ids_flat = experts_local.reshape(-1)

  # Local Sort
  sort_indices = jnp.argsort(expert_ids_flat, stable=True)

  # Permute (Local Gather)
  if top_k > 1:
    # We need to expand x to match the size of expert_ids (which is Token * TopK)
    # x_local: [Tokens, Dim] -> [Tokens, TopK, Dim] -> [Tokens*TopK, Dim]
    x_expanded = jnp.repeat(x_local, top_k, axis=0)
    routed_input = x_expanded[sort_indices]
  else:
    routed_input = x_local[sort_indices]

  top_scores_sorted = top_scores_local.reshape(-1)[sort_indices]

  return routed_input, sort_indices, top_scores_sorted


def _local_unpermute_jax(routed_output, sort_indices, original_shape, top_k):
  """Scatters results back to original positions locally."""
  # routed_output: [LocalTokens*TopK, Dim]

  local_tokens = routed_output.shape[0] // top_k
  dim = routed_output.shape[1]

  out_flat = jnp.zeros((local_tokens, dim), dtype=routed_output.dtype)

  # Map sort_indices back to original input indices (0..LocalTokens-1)
  original_indices = sort_indices // top_k

  # Scatter Add (Local)
  out_flat = out_flat.at[original_indices].add(routed_output)

  return out_flat


def make_count_runner(mesh, num_experts):
  return jax.jit(
      shard_map.shard_map(
          lambda e: _local_count_tokens_jax(e, num_experts),
          mesh=mesh,
          # In: Sharded Batch. Out: Sharded "Batch of Counts" (Rank 2)
          in_specs=P("fsdp", None),
          out_specs=P("fsdp", None),
          check_rep=False,
      )
  )


def make_permute_runner(mesh, top_k):
  return jax.jit(
      shard_map.shard_map(
          lambda x, e, s: _local_permute_jax(x, e, s, top_k),
          mesh=mesh,
          in_specs=(P("fsdp", None), P("fsdp", None), P("fsdp", None)),
          # Fixed Specs:
          # routed_input: Rank 2 [Batch, Dim] -> P('fsdp', None)
          # sort_indices: Rank 1 [Batch]      -> P('fsdp')
          # top_scores:   Rank 1 [Batch]      -> P('fsdp')
          out_specs=(P("fsdp", None), P("fsdp"), P("fsdp")),
          check_rep=False,
      )
  )


def make_unpermute_runner(mesh, top_k):
  return jax.jit(
      shard_map.shard_map(
          lambda r, s, o_shp: _local_unpermute_jax(r, s, o_shp, top_k),
          mesh=mesh,
          # Fixed Specs:
          # routed_output: Rank 2 [Batch, Dim] -> P('fsdp', None)
          # sort_indices:  Rank 1 [Batch]      -> P('fsdp')
          # output_shape:  Rank 2 [Batch, Dim] -> P('fsdp', None) (Dummy)
          in_specs=(P("fsdp", None), P("fsdp"), P("fsdp", None)),
          out_specs=P("fsdp", None),
          check_rep=False,
      )
  )


# TODO(jialeic): Remove this function once we upstream the fix to torchtitan
# This is a temporary solution to compute the nparams and flops for MoE models.
# When using the jax scan the model name is changed from `moe.experts` to
# `moe___experts` so we need to use a different naming convention to
# extract the nparams
def get_moe_model_nparams_and_flops(
    model_args: Any,
    model: nn.Module,
    head_dims: int,
    seq_len: int,
) -> tuple[int, float]:
  """Calculate nparams and nflops for MoE models.

  Args:
      model_args: BaseModelArgs object containing model configuration parameters
        including MoE settings.
      model: nn.Module representing the MoE model.
      head_dims: The sum of qk and v head dimensions.
      seq_len: The sequence length in training configs.

  Returns:
      Tuple of (nparams, num_flops_per_token):
          nparams: Total number of model parameters including all experts.
          num_flops_per_token: Estimated number of floating point operations per
          token
                              based on active parameters only.
  """
  nparams_embedding = 0
  nparams_moe_router = 0
  nparams_shared_experts = 0
  nparams_experts = 0
  nparams_dense = 0

  for name, p in model.named_parameters():
    if "embedding" in name:
      nparams_embedding += p.numel()
      nparams_dense += p.numel()
    elif "moe.shared_experts" in name or "moe___shared_experts" in name:
      nparams_shared_experts += p.numel()
    elif "moe.router" in name or "moe___router" in name:
      nparams_moe_router += p.numel()
    elif "moe.experts" in name or "moe___experts" in name:
      nparams_experts += p.numel()
    else:
      nparams_dense += p.numel()

  nparams_sparse = nparams_moe_router + nparams_shared_experts + nparams_experts
  nparams = nparams_dense + nparams_sparse
  nparams_sparse_active = (
      nparams_moe_router
      + nparams_shared_experts
      + nparams_experts
      * model_args.moe_args.top_k
      // model_args.moe_args.num_experts
  )

  logger.info(
      f"Total parameter count: dense {nparams_dense:,}, sparse"
      f" {nparams_sparse:,}, active {nparams_dense + nparams_sparse_active:,}"
  )

  num_flops_per_token = (
      6 * (nparams_dense - nparams_embedding + nparams_sparse_active)
      + 6 * model_args.n_layers * model_args.n_heads * head_dims * seq_len
  )

  # If weight tying is enabled, subtract embedding parameters from total count
  if (
      hasattr(model_args, "enable_weight_tying")
      and model_args.enable_weight_tying
  ):
    nparams = nparams - nparams_embedding

  return nparams, num_flops_per_token
