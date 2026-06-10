# Copyright 2026 The TorchTitan Authors. All Rights Reserved.

"""GRPO utilities for loss and advantage computation."""

import typing
import torch
import torch.distributed.tensor as dist_tensor
import torch.nn.functional as F




def compute_grpo_advantages(
    rewards: torch.Tensor, group_size: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute advantages by normalizing rewards within groups."""
  batch_size = rewards.shape[0]
  assert (
      batch_size % group_size == 0
  ), f"Batch size {batch_size} must be divisible by group_size {group_size}"

  num_groups = batch_size // group_size
  rewards_grouped = rewards.view(num_groups, group_size)

  mean = rewards_grouped.mean(dim=1, keepdim=True)

  # Compute std manually: sqrt(mean((x - mean)^2))
  squared_diffs = (rewards_grouped - mean) ** 2
  var = squared_diffs.mean(dim=1, keepdim=True)
  std = torch.sqrt(var)

  advantages_grouped = (rewards_grouped - mean) / (std + 1e-4)
  advantages = advantages_grouped.view(-1)
  return advantages, std.view(-1)


def sync_model_weights(src_model, dst_model, parallel_dims):
  """Syncs weights from src_model to dst_model, handling DTensors."""
  state_dict = src_model.state_dict()
  tpu_state_dict = {}

  if hasattr(src_model, "device_mesh"):
    default_mesh = src_model.device_mesh
  else:
    try:
      default_mesh = parallel_dims.get_mesh("dp_replicate")
    except ValueError:
      default_mesh = parallel_dims.get_mesh("fsdp")

  for k, v in state_dict.items():
    if isinstance(v, dist_tensor.DTensor):
      tensor_to_wrap = v.full_tensor()
      mesh = v.device_mesh
    else:
      tensor_to_wrap = v
      mesh = default_mesh
    tpu_state_dict[k] = dist_tensor.distribute_tensor(
        tensor_to_wrap, device_mesh=mesh, placements=[dist_tensor.Replicate()]
    )

  dst_model.load_state_dict(tpu_state_dict)


def build_random_dataloader(
    job_config, vocab_size, device
) -> typing.Generator[tuple[dict[str, torch.Tensor], None], None, None]:

  """Yields the same random tokens every step."""
  local_batch_size = job_config.training.local_batch_size
  seq_len = job_config.training.seq_len
  x = torch.randint(0, vocab_size, (local_batch_size, seq_len), device=device)
  while True:
    yield {"input": x}, None


def compute_grpo_loss(
    model,
    prompt_ids: torch.Tensor,
    completed_ids: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor | None = None,
    ppo_clip_eps: float = 0.2,
    grpo_beta: float = 0.1,
) -> torch.Tensor:
  """Compute GRPO loss with KL penalty."""
  outputs = model(completed_ids)

  if isinstance(outputs, tuple):
    logits = outputs[0]
  else:
    logits = outputs

  prompt_len = prompt_ids.shape[1]
  gen_logits = logits[:, prompt_len - 1 : -1, :]
  gen_targets = completed_ids[:, prompt_len:]
  
  # TPU specific patch: Avoid log_softmax + gather which causes XLA compiler crash (IsFusibleUnalignedDUS)
  # due to unaligned DynamicUpdateSlice on large vocab sizes. Use cross_entropy instead.
  ce_loss = F.cross_entropy(
      gen_logits.reshape(-1, gen_logits.size(-1)), 
      gen_targets.reshape(-1), 
      reduction='none'
  )
  token_log_probs = -ce_loss.view(gen_targets.shape)

  # Importance sampling ratio uses old_log_probs
  if old_log_probs is None:
    old_log_probs = token_log_probs.detach()

  log_ratio = token_log_probs - old_log_probs
  log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
  ratio = torch.exp(log_ratio)

  # Expand advantages to match token level shape [batch_size, gen_len]
  advantages = advantages.unsqueeze(-1).expand_as(ratio)

  unclipped_loss = ratio * advantages
  clipped_ratio = torch.clamp(ratio, 1 - ppo_clip_eps, 1 + ppo_clip_eps)
  clipped_loss = clipped_ratio * advantages
  pg_loss = -torch.min(unclipped_loss, clipped_loss).mean()

  # KL penalty uses ref_log_probs (from separate ref model or saved from
  # generation)
  kl_log_ratio = token_log_probs - ref_log_probs
  kl = torch.exp(-kl_log_ratio) + kl_log_ratio - 1.0
  loss = pg_loss + grpo_beta * kl.mean()

  return loss
