"""
padding_packing_utils.py

Holds utilities and patching helpers related to static batch padding, sequence absorbing,
and varlen/masked attention execution for TPU.

=====================================================================
TPU vs GPU Compilation & Shape Semantics:
1. GPU Dynamic Shapes: GPUs run in PyTorch eager mode or use specialized kernels (like flash-attention's
   varlen API) that process fluctuating token shapes with zero compilation overhead.
2. TPU Static Compilation: TorchTPU translates operations into static JAX/HLO graphs.
   Any change in batch size, sequence counts, or total tokens triggers a new, slow JIT recompilation
   on TPU (taking up to 10+ minutes and causing compilation memory spikes).
3. The Static Padding Solution: To maintain high hardware throughput, we statically pad training
   batches to fixed bounds (e.g., exactly 4096 tokens across 64 sequences) and employ custom 
   masking in SDPA. XLA compiles the execution graph once, ensuring fast, zero-overhead subsequent steps.
=====================================================================
"""

import torch
import torch.nn.functional as F
from torchtitan.models.common.attention import VarlenAttention


def patch_varlen_attention(target_device: str) -> None:
    """
    Monkey-patch VarlenAttention for CPU and TPU execution.
    - CPU: Fallback to scaled_dot_product_attention per sequence.
    - TPU: Implement static token/sequence-safe padding SDPA since aten::_flash_attention_forward is not implemented.
    """

    if target_device == "cpu":
        def cpu_varlen_forward(self, xq, xk, xv, *, attention_masks, scale=None, **kwargs):
            cu_seqs = attention_masks.cu_seq_q.tolist()
            out = torch.zeros_like(xq)
            for i in range(len(cu_seqs) - 1):
                start, end = cu_seqs[i], cu_seqs[i+1]
                if start == end: continue
                q = xq[:, start:end, :, :].transpose(1, 2)
                k = xk[:, start:end, :, :].transpose(1, 2)
                v = xv[:, start:end, :, :].transpose(1, 2)
                
                if q.shape[1] != k.shape[1]:
                    num_repeat = q.shape[1] // k.shape[1]
                    k = k.repeat_interleave(num_repeat, dim=1)
                    v = v.repeat_interleave(num_repeat, dim=1)
                    
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=scale
                )
                out[:, start:end, :, :] = attn_out.transpose(1, 2)
            return out
        VarlenAttention.forward = cpu_varlen_forward
    else:
        def tpu_varlen_forward(self, xq, xk, xv, *, attention_masks, scale=None, **kwargs):
            total_tokens = xq.shape[1]
            cu_seqs = attention_masks.cu_seq_q
            
            positions = torch.arange(total_tokens, device=xq.device)
            seq_indices = (positions.unsqueeze(1) >= cu_seqs.unsqueeze(0)).sum(dim=1) - 1
            same_seq_mask = seq_indices.unsqueeze(1) == seq_indices.unsqueeze(0)
            causal_mask = positions.unsqueeze(1) >= positions.unsqueeze(0)
            
            mask = same_seq_mask & causal_mask
            mask = mask.unsqueeze(0).unsqueeze(0)
            
            q = xq.transpose(1, 2)
            k = xk.transpose(1, 2)
            v = xv.transpose(1, 2)
            
            if q.shape[1] != k.shape[1]:
                num_repeat = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(num_repeat, dim=1)
                v = v.repeat_interleave(num_repeat, dim=1)
                
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=scale
            )
            return attn_out.transpose(1, 2)
            
        VarlenAttention.forward = tpu_varlen_forward


def pad_train_batch_to_static(local_batch, device, pad_len: int, max_seqs: int):
    """
    Pads the TrainBatch's token_ids, advantages, and lengths to static sizes.
    Avoids constant Dynamo recompilation on TPU due to fluctuating batch layouts.
    """
    actual_seqs = len(local_batch.seq_lens)
    total_tokens = sum(local_batch.seq_lens)
    
    if total_tokens > pad_len:
        raise ValueError(f"Total tokens {total_tokens} exceeds pad_len {pad_len}")
        
    if actual_seqs > max_seqs:
        raise ValueError(f"Number of sequences {actual_seqs} exceeds max_seqs {max_seqs}")
        
    token_pad = pad_len - total_tokens
    seq_pad = max_seqs - actual_seqs
    
    dummy_seq_lens = []
    for _ in range(seq_pad):
        chunk = min(token_pad, 1024)
        dummy_seq_lens.append(chunk)
        token_pad -= chunk
        
    if token_pad > 0:
        raise ValueError(f"Could not absorb padding. token_pad remaining: {token_pad}")
        
    token_ids = local_batch.token_ids.to(device)
    advantages = local_batch.advantages.to(device)

    if pad_len - total_tokens > 0:
        token_ids = torch.cat([token_ids, torch.zeros((1, pad_len - total_tokens), dtype=token_ids.dtype, device=device)], dim=1)
        
    if seq_pad > 0:
        advantages = torch.cat([advantages, torch.zeros(seq_pad, dtype=advantages.dtype, device=device)], dim=0)
        local_batch.seq_lens.extend(dummy_seq_lens)
        local_batch.prompt_lens.extend(dummy_seq_lens)
        local_batch.response_lens.extend([0] * seq_pad)
        
    return token_ids, advantages, actual_seqs
