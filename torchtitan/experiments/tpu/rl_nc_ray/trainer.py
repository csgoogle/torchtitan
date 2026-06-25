"""
TPU-specific Ray Actor for RL Policy Training.

Inherits from `PolicyTrainer` but is wrapped with `@ray.remote` instead of Monarch's Actor framework. 
Runs exclusively on CPU to decouple from the TPU Generator, patching `VarlenAttention` to use CPU 
SDPA. Replaces `torchstore` weight pushing with `get_model_state_dict()`, extracting CPU tensors for 
Ray to send to the Generator. Implements DTensor workarounds for CPU Gloo checkpoint loading.

TODO (jialei): consider dropping support for CPU to make code (e.g., weight sync) simpler. 
"""

import copy
import logging
import os
import torch
import torch.nn.functional as F
import ray
import torch.distributed.checkpoint as dcp
import torch_tpu

import torchtitan.experiments.tpu.rl_nc_ray  # Ensure mock runs first

from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed.tensor import DTensor
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.types import TrainBatch
from torchtitan.experiments.tpu import utils as tpu_utils
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.common.attention import VarlenAttention
from torchtitan.tools import utils
from torchtitan.experiments.rl.actors.utils import (
    compute_logprobs,
    create_positions_from_seq_lens,
    create_varlen_metadata,
    extract_response_logprobs,
    verify_logprob_identity,
)

logger = logging.getLogger(__name__)

# TPU Hack Constants for Static Padding
PAD_LEN = 4096
MAX_SEQS = 64

@ray.remote
class RayTPUPolicyTrainer(PolicyTrainer):

    def __init__(self, *args, **kwargs):
        # Ray actors don't have torchrun env vars by default.
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        
        target_device = os.environ.get("TORCHTITAN_DEVICE_TYPE", "cpu")
        tpu_utils.set_device_type(target_device)
        
        import torch._dynamo
        # PyTorch Dynamo specializes on the tensor type. In TP overlap, collectives 
        # return AsyncCollectiveTensor or a plain Tensor depending on timing. 
        # Each switch triggers a recompile. We must increase the limit to prevent crashes.
        # See: https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/rl/models/vllm_wrapper.py#L207
        torch._dynamo.config.recompile_limit = 100
        
        if target_device == "tpu":
            # Patch std() to avoid aten::std.correction which is unimplemented on TPU
            orig_std = torch.Tensor.std
            def patched_std(self, *args, **kwargs):
                mean = self.mean()
                if kwargs.get("unbiased", True) and self.numel() > 1:
                    var = ((self - mean) ** 2).sum() / (self.numel() - 1)
                else:
                    var = ((self - mean) ** 2).mean()
                return torch.sqrt(var)
            torch.Tensor.std = patched_std
            
        # Monkey patch torchtitan utils before super().__init__ calls it
        if target_device == "cpu":
            dist_utils._get_distributed_backend = lambda x: "gloo"  

        if not hasattr(utils, "device_module"):
            utils.device_module = utils.get_device_module()  
            utils.device_type = target_device  
            
        # Manually initialize distributed for CPU to prevent XLA from hijacking it
        if target_device == "cpu" and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
            
        # TPU does not support fused AdamW natively, so we fallback to the foreach implementation
        config = args[0] if len(args) > 0 else kwargs.get("config")
        if config is not None:
            config.optimizer.implementation = "foreach"
            
        super().__init__(*args, **kwargs)
        print("@@@ Trainer: Finished init!")

    def _build_model(
        self,
        model_spec,
        config,
        device_type: str,
        hf_assets_path: str,
    ):
        target_device = os.environ.get("TORCHTITAN_DEVICE_TYPE", "cpu")
        print(f"@@@ Trainer Rank {os.environ.get('RANK', '0')}: Patching VarlenAttention for target_device={target_device}")
        
        if target_device == "cpu":
            # Monkey-patch `VarlenAttention` for CPU execution.
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
            # We MUST monkey-patch VarlenAttention for TPU because `aten::_flash_attention_forward` 
            # is not implemented for TPU.
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
            
        return super()._build_model(model_spec, config, device_type, hf_assets_path)

    def _load_initial_hf_weights(self, model, checkpoint_path: str) -> None:
        if os.environ.get("TORCHTITAN_DEVICE_TYPE", "cpu") != "cpu":
            return super()._load_initial_hf_weights(model, checkpoint_path)
            
        if self.sd_adapter is None:
            return

        storage_reader = self.sd_adapter.get_hf_storage_reader(checkpoint_path)
        hf_state_dict = self.sd_adapter.to_hf(model.state_dict())
        
        dcp.load(hf_state_dict, storage_reader=storage_reader)
        torchtitan_state_dict = self.sd_adapter.from_hf(hf_state_dict)

        # Fix DTensor copy issue on CPU manually
        model_sd = model.state_dict()
        with torch.no_grad():
            for k, v in torchtitan_state_dict.items():
                if k in model_sd:
                    orig_v = model_sd[k]
                    
                    def handle_dtensor_copy(target_dtensor, source_tensor):
                        target_local = target_dtensor.to_local() if hasattr(target_dtensor, "to_local") else target_dtensor
                        
                        if hasattr(source_tensor, "to_local"):
                            source_tensor = source_tensor.to_local()
                        elif isinstance(source_tensor, torch.nn.Parameter) and hasattr(source_tensor.data, "to_local"):
                            source_tensor = source_tensor.data.to_local()
                        elif isinstance(source_tensor, torch.nn.Parameter):
                            source_tensor = source_tensor.data
                            
                        if target_local.shape == source_tensor.shape:
                            target_local.copy_(source_tensor)
                        else:
                            for dim in range(len(target_dtensor.shape)):
                                if target_dtensor.shape[dim] != target_local.shape[dim]:
                                    shard_size = target_local.shape[dim]
                                    rank = int(os.environ.get("LOCAL_RANK", 0)) if os.environ.get("TORCHTITAN_DEVICE_TYPE", "cpu") == "cpu" else int(os.environ.get("RANK", 0))
                                    rank_offset = shard_size * rank
                                    indices = [slice(None)] * len(target_dtensor.shape)
                                    indices[dim] = slice(rank_offset, rank_offset + shard_size)
                                    target_local.copy_(source_tensor[tuple(indices)])
                                    return
                            # Fallback if no dim diff found but shape mismatch
                            target_local.copy_(source_tensor)

                    if isinstance(orig_v, DTensor):
                        handle_dtensor_copy(orig_v, v)
                    elif isinstance(orig_v, torch.nn.Parameter) and isinstance(orig_v.data, DTensor):
                        handle_dtensor_copy(orig_v.data, v)
                    else:
                        orig_v.copy_(v)
        logger.info(f"Loaded initial weights from {checkpoint_path}")



    def get_model_state_dict(self) -> dict:
        """Ray-native weight extraction. Returns state dict on CPU."""
        print(f"@@@ Trainer Rank {os.environ.get('RANK', '0')}: Extracting model state dict...")
        
        target_device = os.environ.get("TORCHTITAN_DEVICE_TYPE", "cpu")
        if target_device == "tpu":
            torch_tpu._internal.sync.synchronize(wait=True)
            print(f"@@@ Trainer Rank {os.environ.get('RANK', '0')}: Sync 1 done")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            print(f"@@@ Trainer Rank {os.environ.get('RANK', '0')}: Barrier 1 done")
            
            full_sd = get_model_state_dict(self.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
            print(f"@@@ Trainer Rank {os.environ.get('RANK', '0')}: get_model_state_dict done")
            
            cpu_sd = None
            if int(os.environ.get("RANK", "0")) == 0:
                cpu_sd = {}
                with torch.no_grad():
                    for k, v in full_sd.items():
                        if hasattr(v, "full_tensor"):
                            cpu_sd[k] = v.full_tensor().detach().cpu().clone().contiguous()
                        elif hasattr(v, "to_local"): # fallback
                            cpu_sd[k] = v.to_local().detach().cpu().clone().contiguous()
                        else:
                            cpu_sd[k] = v.detach().cpu().clone().contiguous()
                print("@@@ Trainer Rank 0: Finished extracting full model state dict.")
            
            torch_tpu._internal.sync.synchronize(wait=True)
            print(f"@@@ Trainer Rank {os.environ.get('RANK', '0')}: Sync 2 done")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            print(f"@@@ Trainer Rank {os.environ.get('RANK', '0')}: Barrier 2 done")
        else:
            cpu_sd = {}
            with torch.no_grad():
                for k, v in self.model.state_dict().items():
                    if hasattr(v, "to_local"):
                        cpu_sd[k] = v.to_local().detach().cpu().clone().contiguous()
                    else:
                        cpu_sd[k] = v.detach().cpu().clone().contiguous()
            print("@@@ Trainer: Finished extracting model state dict on CPU.")
            
        print(f"@@@ Trainer Rank {os.environ.get('RANK', '0')}: Returning cpu_sd!")
        return cpu_sd

    def forward_backward(self, train_data: list[TrainBatch]) -> dict:
        """
        Run forward pass, compute loss, and call backward.

        This method overrides the base PolicyTrainer.forward_backward to bridge GPU/Monarch
        with TPU/Ray execution. The core mathematical RL logic remains identical, but introduces
        the following mechanical differences:
        
        1. Synchronous vs Asynchronous: The base class uses `async def` for Monarch. We override 
           with a synchronous `def` because Ray actors natively wrap synchronous methods into 
           asynchronous futures via `.remote()`.
        2. Batch Copying: We deepcopy `train_data` so mutating the batch (e.g. by appending
           padding tokens) doesn't corrupt the original data structure in Ray shared memory.
        3. Static Padding: XLA Dynamo compiles a new graph for every unique sequence length,
           quickly hitting the recompilation limit on TPU. To prevent this, we pad `token_ids`,
           `advantages`, and lengths to statically sized bounds (e.g., exactly 4096 tokens and 
           64 sequences).
        4. Padding Slicing: Before passing outputs to `loss_fn`, we slice out the dummy 
           padding sequences to prevent them from polluting the mean loss calculations.
        """
        logger.debug(
            f"{os.getpid()=} PolicyTrainer forward_backward "
            f"step {self.policy_version}"
        )

        local_batch = copy.copy(train_data[self.dp_rank])
        device = self.device

        # [TPU HACK] Pad all tensors to a static batch size of 4096 tokens
        # to avoid Dynamo recompilations from varying sequence shapes across RL steps.
        target_device = os.environ.get("TORCHTITAN_DEVICE_TYPE", "cpu")
        if target_device == "tpu":
            actual_seqs = len(local_batch.seq_lens)
            total_tokens = sum(local_batch.seq_lens)
            
            if total_tokens > PAD_LEN:
                raise ValueError(f"Total tokens {total_tokens} exceeds pad_len {PAD_LEN}")
                
            if actual_seqs > MAX_SEQS:
                raise ValueError(f"Number of sequences {actual_seqs} exceeds MAX_SEQS {MAX_SEQS}")
                
            token_pad = PAD_LEN - total_tokens
            seq_pad = MAX_SEQS - actual_seqs
            
            dummy_seq_lens = []
            for _ in range(seq_pad):
                chunk = min(token_pad, 1024)
                dummy_seq_lens.append(chunk)
                token_pad -= chunk
                
            if token_pad > 0:
                raise ValueError(f"Could not absorb padding. token_pad remaining: {token_pad}")
                
            token_ids = local_batch.token_ids.to(device)
            advantages = local_batch.advantages.to(device)

            if PAD_LEN - total_tokens > 0:
                token_ids = torch.cat([token_ids, torch.zeros((1, PAD_LEN - total_tokens), dtype=token_ids.dtype, device=device)], dim=1)
                
            if seq_pad > 0:
                advantages = torch.cat([advantages, torch.zeros(seq_pad, dtype=advantages.dtype, device=device)], dim=0)
                local_batch.seq_lens.extend(dummy_seq_lens)
                local_batch.prompt_lens.extend(dummy_seq_lens)
                local_batch.response_lens.extend([0] * seq_pad)

            seq_lens = local_batch.seq_lens
            prompt_lens = local_batch.prompt_lens
            response_lens = local_batch.response_lens
        else:
            token_ids = local_batch.token_ids.to(device)
            advantages = local_batch.advantages.to(device)
            seq_lens = local_batch.seq_lens
            prompt_lens = local_batch.prompt_lens
            response_lens = local_batch.response_lens

        max_seq_len = max(seq_lens)
        rope_cache_len = self.model.freqs_cis.shape[0]
        if max_seq_len > rope_cache_len:
            raise ValueError(
                f"Episode length {max_seq_len} exceeds rope cache size "
                f"{rope_cache_len}. Increase model max_seq_len or reduce "
                f"generation max_tokens."
            )

        attention_masks = create_varlen_metadata(seq_lens, device)
        positions = create_positions_from_seq_lens(seq_lens, device)

        logits = self.model(
            token_ids, attention_masks=attention_masks, positions=positions
        )
        all_policy_logprobs = compute_logprobs(logits, token_ids)
        policy_logprobs = extract_response_logprobs(
            all_policy_logprobs, seq_lens, prompt_lens, response_lens
        )

        if target_device == "tpu":
            # Slice out the dummy sequences so they don't affect loss or crash mean()
            policy_logprobs = policy_logprobs[:actual_seqs]
            advantages = advantages[:actual_seqs]

        loss, loss_metrics = self.loss_fn(
            policy_logprobs=policy_logprobs,
            advantages=advantages,
        )

        verification_result = verify_logprob_identity(
            local_batch.token_logprobs,
            policy_logprobs,
        )

        logger.debug(
            f"Logprob verification: bitwise_identical={verification_result['logprob_bitwise_identical']}, "
            f"max_delta={verification_result['logprob_max_delta']:.6e}, "
            f"diff_mean={verification_result['logprob_diff_mean']:.6e}, "
            f"diff_max={verification_result['logprob_diff_max']:.6e}, "
            f"tokens_checked={verification_result['total_tokens_checked']}"
        )

        # Backward pass
        self.optimizers.zero_grad()
        loss.backward()

        return {
            "loss": loss.item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "logprob_diff_mean": verification_result["logprob_diff_mean"],
            "logprob_diff_max": verification_result["logprob_diff_max"],
            "logprob_max_delta": verification_result["logprob_max_delta"],
            "logprob_bitwise_identical": verification_result[
                "logprob_bitwise_identical"
            ],
            **loss_metrics,
        }
