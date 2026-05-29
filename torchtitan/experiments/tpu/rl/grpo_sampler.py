# Copyright 2026 The TorchTitan Authors. All Rights Reserved.

"""Sampler logic for GRPO on TPU."""

import torch
import torch.nn.functional as F

# Monkeypatch multinomial for TPU (generator not supported)
original_multinomial = torch.multinomial


def patched_multinomial(
    input, num_samples, replacement=False, *, generator=None, out=None
):
  if input.device.type == "tpu":
    return original_multinomial(input, num_samples, replacement, out=out)
  return original_multinomial(
      input, num_samples, replacement, generator=generator, out=out
  )


torch.multinomial = patched_multinomial


def logits_to_probs(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None
) -> torch.Tensor:
  """Converts logits to probabilities with temperature and top-k filtering."""
  logits = logits / max(temperature, 1e-5)
  if top_k is not None and top_k > 0:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    pivot = v.select(-1, -1).unsqueeze(-1)
    logits = torch.where(logits < pivot, -float("Inf"), logits)
  probs = F.softmax(logits, dim=-1)
  return probs


def multinomial_sample_one(
    probs: torch.Tensor, rng: torch.Generator | None = None
) -> torch.Tensor:
  """Samples one token from multinomial distribution."""
  return torch.multinomial(probs, num_samples=1, generator=rng)


def generate_next_token(
    model,
    x: torch.Tensor,
    max_seq_len: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Generates the next token and its log probability.
  
  Pads the input sequence to max_seq_len to satisfy divisible by 512
  constraint for SDPA on TPU.
  """
  cur_len = x.shape[1]
  pad_len = max_seq_len - cur_len
  if pad_len > 0:
    padded_x = F.pad(x, (0, pad_len), value=0)
  else:
    padded_x = x

  outputs = model(padded_x)
  if isinstance(outputs, tuple):
    logits = outputs[0]
  else:
    logits = outputs

  logits_last = logits[:, cur_len - 1, :]

  probs = logits_to_probs(logits_last, temperature, top_k)
  next_token = multinomial_sample_one(probs, rng=rng)

  # Compute log prob of the sampled token
  log_probs = F.log_softmax(logits_last, dim=-1)
  token_log_prob = log_probs.gather(1, next_token)

  return next_token, token_log_prob


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_seq_len: int,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Generates a sequence of tokens autoregressively."""
  if input_ids.ndim == 1:
    input_ids = input_ids.unsqueeze(0)
  rng = None
  if seed is not None:
    rng = torch.Generator(input_ids.device).manual_seed(seed)
  generated_tokens = input_ids.clone()
  all_token_log_probs = []
  for _ in range(max_new_tokens):
    next_token, token_log_prob = generate_next_token(
        model,
        x=generated_tokens,
        max_seq_len=max_seq_len,
        temperature=temperature,
        top_k=top_k,
        rng=rng,
    )
    generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
    all_token_log_probs.append(token_log_prob)

  token_log_probs = torch.cat(all_token_log_probs, dim=1)
  return generated_tokens, token_log_probs


@torch.no_grad()
def generate_fake(
    model,
    input_ids: torch.Tensor,
    max_seq_len: int,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    vocab_size: int = 151936,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Generates purely random tokens and log probs for testing."""
  batch_size = input_ids.shape[0]

  # Generate random token IDs for the completion
  random_completions = torch.randint(
      0, vocab_size, (batch_size, max_new_tokens), device=input_ids.device
  )
  generated_tokens = torch.cat([input_ids, random_completions], dim=1)

  # Generate random log probs for each generated token
  token_log_probs = -torch.abs(
      torch.randn(batch_size, max_new_tokens, device=input_ids.device)
  )

  return generated_tokens, token_log_probs
