"""Parallelization implementation for Qwen3 models.

Uses internal/third_party/py/torchtitan/models/qwen3/infra/parallelize.py 
as a guide.

NOTE: This module was forked from the native implementation in torchtitan to
allow for potential changes to support TPU. As of 01/2026, we have reverted to
the native implementation for unit and training tests. This file is currently unused except
for training tests configured with `model_name="qwen3_tpu"` and
`use_fairscale=False`.
"""
from absl import logging
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor import parallel
from torch.distributed.fsdp import fully_shard, CPUOffload
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
    PrepareModuleInput,
)
import torch
import torchtitan.config
import torchtitan.distributed
from torchtitan.models.llama4.infra.parallelize import apply_compile


def parallelize_qwen3(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: torchtitan.config.JobConfig,
):
  """Apply tensor parallelism and FSDP to the model.

  TODO: Add support for activation checkpointing, torch.compile, async TP, and EP
  (like in reference torchtitan/models/llama3/infra/parallelize.py)
  """
  world_mesh = parallel_dims.world_mesh
  # TODO (from TT reference): TP currently cannot handle uneven seq_len
  # because we set `use_local_output=True` to use plain Tensors for legacy 
  # reasons. Need to revisit this.

  assert (
      job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
  ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

  model_compile_enabled = (
      job_config.compile.enable and "model" in job_config.compile.components
  )
  if parallel_dims.tp_enabled:
    # configuration for enable_float8_tensorwise_tp and enable async TP not used for now
    # EP also not supported for now
    tp_mesh = parallel_dims.get_mesh("tp")
    apply_non_moe_tp(
        model,
        tp_mesh,
        loss_parallel=not job_config.parallelism.disable_loss_parallel,
        # enable_float8_tensorwise_tp=False,
        # enable_async_tp=False,
      )

  if model_compile_enabled:
    logging.info("Applying torch.compile to model components.")
    apply_compile(model, job_config.compile, parallel_dims.ep_enabled)

  if parallel_dims.fsdp_enabled:
    # apply FSDP or HSDP, potentially with Context Parallel
    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    # FSDP skips MoE layers for now.
    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=job_config.training.enable_cpu_offload,
        reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
        # Uncomment once support for MoE/EP is added (like in TT reference).
        # ep_degree=parallel_dims.ep,
        # dp_mod_ep_mesh=(
        #     world_mesh[tuple(dp_mod_ep_mesh_dim_names)]
        #     if parallel_dims.ep_enabled
        #     else None
        # ),
        # gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
    )

      # Enable weight tying after applying parallelisms
    if model.model_args.enable_weight_tying:
      model.output.weight = model.tok_embeddings.weight

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool = False,
    enable_async_tp: bool = False,
    cp_enabled: bool = False,
):  # pylint: disable=unused-argument
  """Apply standard (1D) DTensor TP to Qwen3-style model's dense layers.

  NOTE: enable_float8_tensorwise_tp and enable_async_tp are ignored for now.

  Args:
    model: The PyTorch model to parallelize.
    tp_mesh: The DTensor device mesh.
    loss_parallel: Whether to parallelize the loss calculation.
    enable_float8_tensorwise_tp: Whether to use float8 tensorwise TP.
    enable_async_tp: Whether to use async TP.
    cp_enabled: Whether to use context parallelism. (ignored for now)

  Returns:
    The model with 1D DTensor TP applied to non-MoE layers.
  """
  logging.info(
      "Applying standard DTensor TP to non-MoE layers of %s.", model.__class__.__name__
  )
  world_size = tp_mesh.size()

  # Shard top-level modules (embedding and output projection)
  top_level_plan = {
      "tok_embeddings": RowwiseParallel(
          input_layouts=Replicate(),  # Explicitly state the input is Replicated
          output_layouts=Shard(1)
      ),
      "norm": SequenceParallel(),
      "output": ColwiseParallel(
         input_layouts=Shard(1),
          # Use Shard(-1) if loss_parallel, else Replicate to get full logits
          output_layouts=Shard(-1) if loss_parallel else Replicate(),
          use_local_output=not loss_parallel,
      ),
  }
  parallelize_module(model, tp_mesh, top_level_plan)

  # Shard transformer blocks, skipping MoE layers
  if hasattr(model, "layers"):
    for layer_id, transformer_block in model.layers.named_children():
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
      # Always shard the attention block
      layer_plan = {
          "attention_norm": SequenceParallel(),
          "attention": PrepareModuleInput(
              input_layouts=(Shard(1), Replicate(), None, None),
              desired_input_layouts=(Replicate(), Replicate(), None, None),
          ),  # pytype: disable=wrong-arg-types
          "attention.wq": ColwiseParallel(use_local_output=False),
          "attention.wk": ColwiseParallel(use_local_output=False),
          "attention.wv": ColwiseParallel(use_local_output=False),
          "attention.q_norm": SequenceParallel(sequence_dim=2),
          "attention.k_norm": SequenceParallel(sequence_dim=2),
          "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
      }

      # Conditionally shard the MLP block if it's not a Mixture of Experts
      if hasattr(transformer_block, "feed_forward"):
        logging.info("Layer %s is a dense block, applying TP to MLP.", layer_id)
        ffn_plan = {
            # Feed Forward Block
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),  # pytype: disable=wrong-arg-types
            "feed_forward.w1": ColwiseParallel(output_layouts=Shard(2)),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(output_layouts=Shard(2)),
        }
        layer_plan.update(ffn_plan)
      elif hasattr(transformer_block, "moe"):
        logging.info("Layer %s is an MoE block, skipping TP for MLP.", layer_id)

      # Apply the constructed plan to the current transformer block
      parallelize_module(
          module=transformer_block,
          device_mesh=tp_mesh,
          parallelize_plan=layer_plan,
      )

  return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    gradient_divide_factor: int | None = None,
):
    """
    Apply data parallelism (via DTensor-based FSDP) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.
        NOTE: EP is not supported in this implementation.
    """
    # Base FSDP configuration dictionary
    fsdp_config = {
        "mesh": dp_mesh,
    }

    if cpu_offload:
      fsdp_config["cpu_offload"] = CPUOffload(offload_params=True)

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
    if model.tok_embeddings is not None:
      fully_shard(
          model.tok_embeddings,
          **fsdp_config,
          reshard_after_forward=reshard_after_forward,
      )

    # Apply FSDP to each transformer block
    for layer_id, transformer_block in model.layers.items():
      if transformer_block.moe_enabled:
        logging.warning(
            "Layer %s is an MoE block, but MoE is not yet supported by FSDP."
            " Skipping FSDP for this layer.",
            layer_id,
        )
      else:
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # As an optimization, do not reshard_after_forward the last layers by
    # default since FSDP would prefetch them immediately after the forward pass
    if model.norm is not None:
      fully_shard(
          model.norm,
          **fsdp_config,
          reshard_after_forward=(reshard_after_forward_policy == "always"),
      )
    if model.output is not None:
      fully_shard(
          model.output,
          **fsdp_config,
          reshard_after_forward=(reshard_after_forward_policy == "always"),
      )

    fully_shard(model, **fsdp_config, reshard_after_forward=False)

    return model


