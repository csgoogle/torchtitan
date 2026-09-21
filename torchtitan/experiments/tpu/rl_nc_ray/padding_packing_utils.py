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
from torchtitan.models.common.attention import VarlenInnerAttention


def patch_create_varlen_metadata(max_seqs: int = 64) -> None:
    """
    Monkey-patch create_varlen_metadata_for_document to pad metadata to static shapes.
    Prevents XLA recompilation due to varying number of sequences in a microbatch.
    """
    import torchtitan.models.common.attention as attn
    
    if hasattr(attn, "_original_create_varlen_metadata"):
        return # Already patched
        
    attn._original_create_varlen_metadata = attn.create_varlen_metadata_for_document
    
    def padded_create_varlen_metadata(positions):
        metadata = attn._original_create_varlen_metadata(positions)
        cu_seq_q = metadata.cu_seq_q
        cu_seq_k = metadata.cu_seq_k
        
        pad_val_q = cu_seq_q[-1]
        pad_val_k = cu_seq_k[-1]
        
        seq_count = cu_seq_q.shape[0] - 1
        if seq_count > max_seqs:
            raise ValueError(f"Number of sequences {seq_count} exceeds max_seqs {max_seqs}")
            
        pad_len = max_seqs + 1 - cu_seq_q.shape[0]
        if pad_len > 0:
            padded_cu_seq_q = torch.cat([cu_seq_q, pad_val_q.repeat(pad_len)])
            padded_cu_seq_k = torch.cat([cu_seq_k, pad_val_k.repeat(pad_len)])
        else:
            padded_cu_seq_q = cu_seq_q
            padded_cu_seq_k = cu_seq_k
            
        return metadata._replace(cu_seq_q=padded_cu_seq_q, cu_seq_k=padded_cu_seq_k)

    attn.create_varlen_metadata_for_document = padded_create_varlen_metadata


def patch_varlen_attention(target_device: str) -> None:
    """
    Monkey-patch VarlenInnerAttention for CPU and TPU execution.
    - CPU: Fallback to scaled_dot_product_attention per sequence.
    - TPU: Implement static token/sequence-safe padding SDPA since aten::_flash_attention_forward is not implemented.
    """

    if target_device == "cpu":
        def cpu_varlen_forward(self, q_BLNH, k_BLNH, v_BLNH, *, attention_masks, scale=None, **kwargs):
            cu_seqs = attention_masks.cu_seq_q.tolist()
            chunks = []
            for i in range(len(cu_seqs) - 1):
                start, end = cu_seqs[i], cu_seqs[i+1]
                if start == end:
                    continue
                q = q_BLNH[:, start:end, :, :].transpose(1, 2)
                k = k_BLNH[:, start:end, :, :].transpose(1, 2)
                v = v_BLNH[:, start:end, :, :].transpose(1, 2)
                
                if q.shape[1] != k.shape[1]:
                    num_repeat = q.shape[1] // k.shape[1]
                    k = k.repeat_interleave(num_repeat, dim=1)
                    v = v.repeat_interleave(num_repeat, dim=1)
                    
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=scale
                )
                chunks.append(attn_out.transpose(1, 2))
                
            if len(chunks) == 0:
                return torch.zeros_like(q_BLNH)
            out = torch.cat(chunks, dim=1)
            if out.shape[1] < q_BLNH.shape[1]:
                pad = torch.zeros(
                    (q_BLNH.shape[0], q_BLNH.shape[1] - out.shape[1], q_BLNH.shape[2], q_BLNH.shape[3]),
                    dtype=q_BLNH.dtype,
                    device=q_BLNH.device,
                )
                out = torch.cat([out, pad], dim=1)
            return out
        VarlenInnerAttention.forward = cpu_varlen_forward
    else:
        def tpu_varlen_forward(self, q_BLNH, k_BLNH, v_BLNH, *, attention_masks, scale=None, **kwargs):
            B, L, N, H = q_BLNH.shape
            N_kv = k_BLNH.shape[2]
            T = B * L
            
            q_flat = q_BLNH.view(1, T, N, H)
            k_flat = k_BLNH.view(1, T, N_kv, H)
            v_flat = v_BLNH.view(1, T, N_kv, H)
            
            cu_seqs = attention_masks.cu_seq_q
            
            positions = torch.arange(T, device=q_BLNH.device)
            seq_indices = (positions.unsqueeze(1) >= cu_seqs.unsqueeze(0)).sum(dim=1) - 1
            same_seq_mask = seq_indices.unsqueeze(1) == seq_indices.unsqueeze(0)
            causal_mask = positions.unsqueeze(1) >= positions.unsqueeze(0)
            
            mask = same_seq_mask & causal_mask
            mask = mask.unsqueeze(0).unsqueeze(0)
            
            q = q_flat.transpose(1, 2)
            k = k_flat.transpose(1, 2)
            v = v_flat.transpose(1, 2)
            
            if q.shape[1] != k.shape[1]:
                num_repeat = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(num_repeat, dim=1)
                v = v.repeat_interleave(num_repeat, dim=1)
                
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=scale
            )
            return attn_out.transpose(1, 2).view(B, L, N, H)
            
        VarlenInnerAttention.forward = tpu_varlen_forward


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
