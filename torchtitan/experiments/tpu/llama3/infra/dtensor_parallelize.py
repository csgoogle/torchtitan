"""Parallelization implementations for Llama3 models.

Based on native implementation in internal/third_party/py/torchtitan/models/llama3/infra/parallelize.py.

NOTE: This module was forked from the native implementation in torchtitan to
allow for potential changes to support TPU. As of 01/2026, we have reverted to
the native implementation for unit and training tests. This file is currently unused except
for training tests configured with `model_name="llama3_tpu"` and
`use_fairscale=False`.
"""
from absl import logging
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, CPUOffload
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
    PrepareModuleInput,
)
import torchtitan.config
from torchtitan.config.job_config import Compile as CompileConfig
import torchtitan.distributed
from torchtitan.tools.logging import logger

TORCH_DTYPE_MAP = torchtitan.config.TORCH_DTYPE_MAP

def parallelize_llama(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: torchtitan.config.JobConfig,
):
  """Apply tensor parallelism to the model. FSDP not enabled for Llama3 models."""
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

  if parallel_dims.tp_enabled:
    # configuration for enable_float8_tensorwise_tp not used for now
    tp_mesh = parallel_dims.get_mesh("tp")
    apply_tp(
        model,
        tp_mesh,
        loss_parallel=not job_config.parallelism.disable_loss_parallel,
    )

  model_compile_enabled = (
      job_config.compile.enable and "model" in job_config.compile.components
  )
  if model_compile_enabled:
    logging.info("Applying torch.compile to model components.")
    apply_compile(model, job_config.compile)

  if parallel_dims.fsdp_enabled:
    # dp_mesh is the mesh for FSDP/HSDP
    names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(names)

    apply_fsdp(
        model,
        dp_mesh,
        # NOTE: param_ and reduce_ dtype are not used in this module's FSDP
        # implementation.
        param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=job_config.training.enable_cpu_offload,
        reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
    )
  return model


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool = False,
):  # pylint: disable=unused-argument
  """Apply Hybrid TP+SP a Llama-style model.

  NOTE: enable_float8_tensorwise_tp is ignored for now.

  Args:
    model: The PyTorch model to parallelize.
    tp_mesh: The DTensor device mesh.
    loss_parallel: Whether to parallelize the loss calculation.
    enable_float8_tensorwise_tp: Whether to use float8 tensorwise TP.

  Returns:
    The module with DTensor1D TP applied.
  """
  logging.info("Applying standard DTensor TP to %s.", model.__class__.__name__)

  world_size = tp_mesh.size()

  top_level_plan = {
      "tok_embeddings": RowwiseParallel(
          input_layouts=Replicate(),  # Explicitly state the input is Replicated
          output_layouts=Shard(1)
      ),
      # SequenceParallel() is the only way to ensure weights within norm
      # layers (e.g., RMSNorm) are Replicated across all ranks. Otherwise,
      # weights have to be broadcasted before/after parallelizing to ensure
      # that they are replicated.
      # https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html
      "norm": SequenceParallel(),
      "output": ColwiseParallel(
         input_layouts=Shard(1),
          # Use Shard(-1) if loss_parallel, else Replicate to get full logits
          output_layouts=Shard(-1) if loss_parallel else Replicate(),
          use_local_output=not loss_parallel,
      ),  # pytype: disable=wrong-arg-types
  }
  parallelize_module(model, tp_mesh, top_level_plan)

  # Parallelize the Transformer Blocks
  if hasattr(model, "layers"):
    for _, transformer_block in model.layers.named_children():
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
      layer_plan = {
          # Attention Block
          "attention_norm": SequenceParallel(),
          "attention": PrepareModuleInput(
              input_layouts=(Shard(1), None, None, None),
              desired_input_layouts=(Replicate(), None, None, None),
          ),  # pytype: disable=wrong-arg-types
          "attention.wq": ColwiseParallel(output_layouts=Shard(2)),
          "attention.wk": ColwiseParallel(output_layouts=Shard(2)),
          "attention.wv": ColwiseParallel(output_layouts=Shard(2)),
          "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
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

      parallelize_module(
          module=transformer_block,  # Call parallelize_module on the block
          device_mesh=tp_mesh,
          parallelize_plan=layer_plan,
      )
  return model


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(
            transformer_block, backend=compile_config.backend, fullgraph=True
        )
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype, # pylint: disable=unused-argument
    reduce_dtype: torch.dtype, # pylint: disable=unused-argument
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
): 
  """
  Apply data parallelism (via DTensor-based FSDP) to the model.

  Args:
      model (nn.Module): The model to apply data parallelism to.
      dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
      param_dtype (torch.dtype): The data type to use for model parameters (unused in this implementation).
      reduce_dtype (torch.dtype): The data type to use for reduction operations (unused in this implementation).
      pp_enabled (bool): Whether pipeline parallelism is enabled.
      cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
      reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
          Other options: "never", "always".
          - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
          - "always" will enable `reshard_after_forward` for all forward passes.
          - "never" will disable `reshard_after_forward` for all forward passes.
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
      # all-gathers, which can be expensive and non-overlapped.
      reshard_after_forward = not pp_enabled
    case _:
      raise ValueError(
          "Invalid reshard_after_forward_policy:"
          f" {reshard_after_forward_policy}."
      )

  # Apply FSDP to the embedding layer
  if model.tok_embeddings is not None:
    fully_shard(
        model.tok_embeddings,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )

  # Apply FSDP to each transformer block
  for transformer_block in model.layers.values():
    fully_shard(
        transformer_block,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
  # As an optimization, do not reshard_after_forward the last layers by default
  # since FSDP would prefetch them immediately after the forward pass
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

  fully_shard(model, **fsdp_config)

  return model
