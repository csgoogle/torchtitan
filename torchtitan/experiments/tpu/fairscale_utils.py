"""Utility functions for applying Fairscale Tensor Parallelism."""

from absl import logging
from fairscale.nn import model_parallel
from fairscale.nn.model_parallel import layers as fairscale_layers
import torch
from torch import nn


def replace_embedding(
    module: nn.Module,
    child_name: str,
    emb_layer: nn.Embedding,
    *,
    rank: int,
    world_size: int,
):
  """Replaces nn.Embedding with fairscale_layers.ParallelEmbedding."""
  num_embeddings = emb_layer.num_embeddings
  embedding_dim = emb_layer.embedding_dim
  device = emb_layer.weight.device

  if embedding_dim % world_size != 0:
    raise ValueError(
        f"embedding_dim ({embedding_dim}) must be divisible by world_size"
        f" ({world_size}) for {child_name}"
    )
  new_layer = fairscale_layers.ParallelEmbedding(
      num_embeddings, embedding_dim, init_method=lambda w: None
  ).to(device)

  # Copy weights to the parallel embedding layer.
  hidden_chunk_size = embedding_dim // world_size
  start_col = rank * hidden_chunk_size
  end_col = start_col + hidden_chunk_size
  with torch.no_grad():
    new_layer.weight.data.copy_(emb_layer.weight.data[:, start_col:end_col])
  setattr(module, child_name, new_layer)
  logging.info("Replaced %s with ParallelEmbedding", child_name)


def replace_colwise_linear(
    module: nn.Module,
    child_name: str,
    linear_layer: nn.Linear,
    *,
    rank: int,
    world_size: int,
    gather_output: bool = False,
):
  """Replaces nn.Linear with fairscale_layers.ColumnParallelLinear."""
  in_features = linear_layer.in_features
  out_features = linear_layer.out_features
  has_bias = linear_layer.bias is not None
  device = linear_layer.weight.device

  if out_features % world_size != 0:
    raise ValueError(
        f"out_features ({out_features}) must be divisible by world_size"
        f" ({world_size}) for {child_name}"
    )
  new_layer = fairscale_layers.ColumnParallelLinear(
      in_features,
      out_features,
      bias=has_bias,
      gather_output=gather_output,
      init_method=lambda w: None,
  ).to(device)

  # Copy weights to the parallel linear layer.
  out_chunk_size = out_features // world_size
  start_row = rank * out_chunk_size
  end_row = start_row + out_chunk_size
  with torch.no_grad():
    new_layer.weight.data.copy_(linear_layer.weight.data[start_row:end_row, :])
    if has_bias:
      new_layer.bias.data.copy_(linear_layer.bias.data[start_row:end_row])
  setattr(module, child_name, new_layer)
  logging.info(
      "Replaced %s with ColumnParallelLinear(gather_output=%s)",
      child_name,
      gather_output,
  )


def replace_rowwise_linear(
    module: nn.Module,
    child_name: str,
    linear_layer: nn.Linear,
    *,
    rank: int,
    world_size: int,
):
  """Replaces nn.Linear with fairscale_layers.RowParallelLinear."""
  in_features = linear_layer.in_features
  out_features = linear_layer.out_features
  has_bias = linear_layer.bias is not None
  device = linear_layer.weight.device

  if in_features % world_size != 0:
    raise ValueError(
        f"in_features ({in_features}) must be divisible by world_size"
        f" ({world_size}) for {child_name}"
    )
  new_layer = fairscale_layers.RowParallelLinear(
      in_features,
      out_features,
      bias=has_bias,
      input_is_parallel=True,
      init_method=lambda w: None,
  ).to(device)

  # Copy weights to the parallel linear layer.
  in_chunk_size = in_features // world_size
  start_col = rank * in_chunk_size
  end_col = start_col + in_chunk_size
  with torch.no_grad():
    new_layer.weight.data.copy_(linear_layer.weight.data[:, start_col:end_col])
    if has_bias:
      new_layer.bias.data.copy_(linear_layer.bias.data)
  setattr(module, child_name, new_layer)
  logging.info("Replaced %s with RowParallelLinear", child_name)
