"""Apply tensor parallelism to Qwen3 models using Fairscale.

Uses internal/third_party/py/torchtitan/models/qwen3/infra/parallelize.py
as a guide.
"""

from absl import logging
from fairscale.nn import data_parallel
from fairscale.nn import model_parallel
import torch
from torch import nn
import torch.distributed as dist
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.tpu import fairscale_utils


def parallelize_qwen3(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: torchtitan.config.JobConfig,
    rank: int,
):
  """Apply tensor parallelism and/or FSDP to a Qwen3 model.

  Note: Fairscale FSDP will only work on CUDA devices.
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

    model = apply_non_moe_tp(model, world_size=parallel_dims.tp, rank=rank)

  if parallel_dims.fsdp_enabled:
    if not model_parallel.initialize.model_parallel_is_initialized():
      model_parallel.initialize.initialize_model_parallel(parallel_dims.tp)
    try:
      all_ranks = list(range(parallel_dims.world_size))
      dp_group = dist.new_group(ranks=all_ranks)

      apply_fsdp(
          model,
          dp_group=dp_group,
          pp_enabled=parallel_dims.pp_enabled,
          cpu_offload=job_config.training.enable_cpu_offload,
          reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
      )
    except Exception as e:

        logging.error(f"Fairscale FSDP application failed: {e}")
        raise RuntimeError(
            f"Failed to initialize Fairscale FSDP (Error: {e}).\n"
            "NOTE: Fairscale FSDP APIs is strictly CUDA-dependent. "
            "If you are running on CPU or TPU, this failure could be expected."
        ) from e

  return model


def apply_non_moe_tp(model: nn.Module, world_size: int, rank: int):
  """Apply Fairscale Tensor Parallelism to a Qwen3-style model's dense layers."""
  logging.info(
      "Applying Fairscale TP to model %s with world_size %d, rank %d",
      model.__class__.__name__,
      world_size,
      rank,
  )
  logging.info(
      "NOTE: Fairscale apply_non_moe_tp implements pure-TP (unlike DTensor apply_tp,"
      " which implements hybrid TP+SP)."
  )
  # Shard top-level modules
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
    for layer_id, transformer_block in model.layers.named_children():
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
      # Check for a dense feed_forward block before sharding.
      # This skips MoE blocks, which require EP.
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
      elif hasattr(transformer_block, "moe"):
        logging.info("Layer %s is an MoE block, skipping TP for MLP.", layer_id)
  return model


def apply_fsdp(
    model: nn.Module,
    dp_group: dist.ProcessGroup,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
  """
  Apply data parallelism (via Fairscale FSDP) to the model by wrapping
  submodules.

  Args:
      model (nn.Module): The model to apply data parallelism to.
      dp_group (dist.ProcessGroup): The process group for data parallelism.
      pp_enabled (bool): Whether pipeline parallelism is enabled.
      cpu_offload (bool, optional): Whether to offload model parameters to CPU.
        Defaults to False.
      reshard_after_forward_policy (str, optional): The policy to use for
        resharding after forward pass. Defaults to "default".
          Other options: "never", "always".
  """
  # Base FSDP configuration dictionary
  fsdp_config = {
      "process_group": dp_group,
      "cpu_offload": cpu_offload,
  }

  # Determine the reshard_after_forward policy based on the string input
  match reshard_after_forward_policy:
    case "always":
      reshard_after_forward = True
    case "never":
      reshard_after_forward = False
    case "default":
      # For PP, by default do not reshard after forward to avoid per-microbatch
      # all-gathers, which can be expensive and non-overlapped
      reshard_after_forward = not pp_enabled
    case _:
      raise ValueError(
          f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
      )

  # Apply FSDP to the embedding layer
  if hasattr(model, "tok_embeddings"):
    logging.info("Wrapping tok_embeddings with Fairscale FSDP")
    model.tok_embeddings = data_parallel.FullyShardedDataParallel(
        model.tok_embeddings,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )

  # Apply FSDP to each transformer block
  if hasattr(model, "layers"):
    for layer_id, transformer_block in model.layers.named_children():
      if hasattr(transformer_block, "moe"):
        logging.warning(
            "Layer %s is an MoE block, but MoE is not yet supported by FSDP."
            " Skipping FSDP for this layer.",
            layer_id,
        )
      else:
        logging.info("Wrapping layer %s with Fairscale FSDP", layer_id)
        fsdp_block = data_parallel.FullyShardedDataParallel(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
        setattr(model.layers, layer_id, fsdp_block)

  # As an optimization, do not reshard_after_forward the last layers by
  # default since FSDP would prefetch them immediately after the forward pass
  if hasattr(model, "norm") and model.norm is not None:
    logging.info("Wrapping norm with Fairscale FSDP")
    model.norm = data_parallel.FullyShardedDataParallel(
        model.norm,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward_policy == "always"
    )
  if hasattr(model, "output") and model.output is not None:
    logging.info("Wrapping output with Fairscale FSDP")
    model.output = data_parallel.FullyShardedDataParallel(
        model.output,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward_policy == "always"
    )

  model = data_parallel.FullyShardedDataParallel(
      model,
      **fsdp_config,
      reshard_after_forward=False,
  )

  return model

