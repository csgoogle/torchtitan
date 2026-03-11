"""Apply tensor parallelism to Llama3 models using Fairscale.

Uses internal/third_party/py/torchtitan/models/llama3/infra/parallelize.py 
as a guide.
"""
from absl import logging
from fairscale.nn import model_parallel
from fairscale.nn.model_parallel import layers as fairscale_layers
import torch
from torch import nn
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.tpu import fairscale_utils


def parallelize_llama(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: torchtitan.config.JobConfig,
    rank: int):
  """Applies Fairscale Tensor Parallelism to a Llama3 model.

  Args:
    model: The PyTorch model to parallelize.
    parallel_dims: Contains the parallel dimensions, including tensor
      parallelism world size.
    job_config: The job configuration.
    rank: The model parallel rank.

  Returns:
    The parallelized PyTorch model.
  """

  # NOTE: Fairscale is not using 'cp', but we keep the check for consistency with
  # DTensor implementation.
  assert (
      job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
  ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}).
        """

  if parallel_dims.tp_enabled:
    if not model_parallel.initialize.model_parallel_is_initialized():
      model_parallel.initialize.initialize_model_parallel(parallel_dims.tp)

    logging.info(
        "Model parallel world size is %s. Applying Fairscale pure-TP.",
        parallel_dims.tp
    )

    model = apply_tp(model, world_size=parallel_dims.tp, rank=rank)

  if parallel_dims.fsdp_enabled:
    raise RuntimeError("Fairscale FSDP not yet added for Llama3 models")

  return model


def apply_tp(model: nn.Module, world_size: int, rank: int):
  """Apply Fairscale Tensor Parallelism to a Llama-style model.

  Mimics the sharding strategy of apply_tp of
  internal/third_party/py/torchtitan/models/llama3/infra/parallelize.py

  Args:
    model: The PyTorch model to parallelize.
    world_size: The model parallel world size.
    rank: The model parallel rank.

  Returns:
    The parallelized PyTorch model.
  """
  logging.info("Applying Fairscale TP to %s.", model.__class__.__name__)
  logging.info(
      "NOTE: Fairscale apply_tp implements pure-TP (unlike DTensor apply_tp,"
      " which implements hybrid TP+SP)."
  )
  logging.debug("model before TP application: %s", model)
  # Shard top-level modules
  # NOTE: Omits norm layer parallelization (out of scope for standard TP)
  if hasattr(model, "tok_embeddings"):
    fairscale_utils.replace_embedding(
        model,
        "tok_embeddings",
        model.tok_embeddings,
        rank=rank,
        world_size=world_size,
    )
  if hasattr(model, "output"):
    # Shard vocab dim (ColwiseParallel in torchtitan parallelize.py)
    # gather_output=True to get full logits for cross_entropy on each rank
    fairscale_utils.replace_colwise_linear(
        model,
        "output",
        model.output,
        rank=rank,
        world_size=world_size,
        gather_output=True,
    )

  # Shard transformer blocks
  if hasattr(model, "layers"):
    for _, transformer_block in model.layers.named_children():
      # Attention
      if hasattr(transformer_block, "attention"):
        attn = transformer_block.attention
        # If n_heads is not set, or incompatible with world_size, skip TP.
        if not hasattr(attn, "n_heads"):
          logging.warning("Attention module has no 'n_heads', skipping TP.")
          continue
        if attn.n_heads % world_size != 0:
          raise ValueError(
              f"n_heads ({attn.n_heads}) must be divisible by "
              f"tensor parallel world_size ({world_size})"
          )
        if hasattr(attn, "n_kv_heads"):
          if attn.n_kv_heads % world_size != 0:
            raise ValueError(
                f"n_kv_heads ({attn.n_kv_heads}) not divisible by world_size"
                f" ({world_size}). No fallback available for replication or"
                " MQA."
            )

        if hasattr(attn, "wq"):
          fairscale_utils.replace_colwise_linear(
              attn,
              "wq",
              attn.wq,
              rank=rank,
              world_size=world_size,
              gather_output=False,
          )
        if hasattr(attn, "wk"):
          fairscale_utils.replace_colwise_linear(
              attn,
              "wk",
              attn.wk,
              rank=rank,
              world_size=world_size,
              gather_output=False,
          )
        if hasattr(attn, "wv"):
          fairscale_utils.replace_colwise_linear(
              attn,
              "wv",
              attn.wv,
              rank=rank,
              world_size=world_size,
              gather_output=False,
          )
        if hasattr(attn, "wo"):
          fairscale_utils.replace_rowwise_linear(
              attn, "wo", attn.wo, rank=rank, world_size=world_size
          )
      # Feed Forward
      if hasattr(transformer_block, "feed_forward"):
        ffn = transformer_block.feed_forward
        if hasattr(ffn, "w1"):
          fairscale_utils.replace_colwise_linear(
              ffn,
              "w1",
              ffn.w1,
              rank=rank,
              world_size=world_size,
              gather_output=False,
          )
        if hasattr(ffn, "w2"):
          fairscale_utils.replace_rowwise_linear(
              ffn, "w2", ffn.w2, rank=rank, world_size=world_size
          )
        if hasattr(ffn, "w3"):
          fairscale_utils.replace_colwise_linear(
              ffn,
              "w3",
              ffn.w3,
              rank=rank,
              world_size=world_size,
              gather_output=False,
          )
  logging.debug("model after TP application: %s", model)
  return model
