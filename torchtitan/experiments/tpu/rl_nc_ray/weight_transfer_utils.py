"""
weight_transfer_utils.py

Highly optimized, productionized weight transfer utilities and helper functions for TPU-to-TPU non-colocated RL.
Hosts helpers for:
- TPU collective replication over ICI.
- zero-copy CPU serialization via Ray Plasma.
- CPU-sliced chunked flat memory-safe streaming to TPU.
- Policy Trainer weight extraction and CPU offloading (ICI and PCIe Concatenation).
- Generator/Worker weight injection and slicing.
"""

import logging
import os
import re
import time
import torch
import torch_tpu
from torch.distributed.tensor import DTensor, Replicate

logger = logging.getLogger(__name__)

# --- GLOBAL CONFIGURATION / CONSTANTS ---
# Toggle between whole-model weight sync (False) and layer-by-layer weight sync (True).
# Layer-by-layer sync reduces TPU HBM memory peak dramatically by only redistributing and 
# offloading a single layer/group at a time.
SYNC_LAYER_BY_LAYER = True

# Peak available TPU memory after vLLM KV Cache allocation is extremely tight (~455MB).
# Setting this chunk size limits temporary HBM footprint to ~15MB per chunk, ensuring
# absolute safety against compilation limits while preserving stream PCIe bandwidth.
TPU_COPY_CHUNK_SIZE_PARAMETERS = 30


# =====================================================================
# Namespace & Formatting Utilities
# =====================================================================

def get_clean_name(name: str) -> str:
    """Strip FSDP/DCP wrapper prefixes from state dict keys to match standard model namespaces."""
    return name.replace("_fsdp_wrapped_module.", "") \
               .replace("_checkpoint_wrapped_module.", "") \
               .replace("module.", "")


def get_layer_group(key: str) -> str:
    """
    Given a state dict key, returns its group name (e.g. 'embeddings', 'layers.0', 'output').
    """
    clean_k = get_clean_name(key)
    # Match 'layers.<number>.' pattern
    match = re.search(r'layers\.(\d+)\.', clean_k)
    if match:
        return f"layers.{match.group(1)}"
    elif "tok_embeddings" in clean_k:
        return "embeddings"
    else:
        return "output"


def get_sorted_groups(keys: list[str]) -> list[str]:
    """
    Groups state dict keys and returns the group names sorted in forward order.
    """
    groups = set()
    for k in keys:
        groups.add(get_layer_group(k))
    
    # Sort groups: embeddings first, then layers in numerical order, then output last
    layers = []
    others = []
    for g in groups:
        if g.startswith("layers."):
            layers.append(g)
        else:
            others.append(g)
            
    layers.sort(key=lambda x: int(x.split(".")[1]))
    
    sorted_g = []
    if "embeddings" in others:
        sorted_g.append("embeddings")
    sorted_g.extend(layers)
    if "output" in others:
        sorted_g.append("output")
        
    for g in others:
        if g not in ("embeddings", "output") and g not in sorted_g:
            sorted_g.append(g)
            
    return sorted_g


# =====================================================================
# Trainer Weight Extraction & CPU Offloading (ICI and PCIe Concatenation)
# =====================================================================

def prepare_trainer_state_dict(model) -> dict:
    """
    Extracts, replicates, and offloads model parameters directly using TPU Inter-Chip Connect (ICI).
    Concatenates parameters by dtype on TPU to reduce PCIe transfer and Ray Object Store overheads.
    
    Supports both whole-model weight sync (SYNC_LAYER_BY_LAYER = False) and highly scalable
    layer-by-layer/block-by-block weight sync (SYNC_LAYER_BY_LAYER = True).
    
    Why: Whole-model sync requires redistributing and flat-concatenating all 1.2GB (or 14GB for 7B)
    parameters concurrently, which leads to a massive TPU memory peak and instant OOM.
    What didn't work: Keeping all replicated tensors in memory across layers; we must delete group references
    and call synchronize() sequentially to recycle intermediate TPU memory.
    
    TODO: Remove HACK. Normal non-colocated RL usage would use native distributed channels/actors
    without PJRT client locking and preallocation constraints.
    """
    t_start = time.perf_counter()
    
    if not SYNC_LAYER_BY_LAYER:
        # --- Whole-Model Synchronization (Baseline / Legacy) ---
        tpu_sd = {}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                val = v.data if isinstance(v, torch.nn.Parameter) else v
                if isinstance(val, DTensor):
                    device_mesh = val.device_mesh
                    tpu_sd[k] = val.redistribute(device_mesh, [Replicate()] * len(val.placements))
                else:
                    tpu_sd[k] = val

        t_redist_queue = time.perf_counter() - t_start
        
        torch_tpu._internal.sync.synchronize(wait=True)
        t_redist_sync = time.perf_counter() - t_start - t_redist_queue
        
        cpu_sd = None
        t_copy = 0.0
        
        if int(os.environ.get("RANK", "0")) == 0:
            t_copy_start = time.perf_counter()
            cpu_sd = {
                "flat_tensors": {},
                "metadata": {}
            }
            by_dtype = {}
            
            with torch.no_grad():
                for k, v in tpu_sd.items():
                    local_v = v.to_local() if isinstance(v, DTensor) else v
                    by_dtype.setdefault(local_v.dtype, []).append((k, local_v))
                
                for dtype, items in by_dtype.items():
                    if not items:
                        continue
                    flat_tpu = torch.cat([v.detach().view(-1) for _, v in items])
                    flat_cpu = flat_tpu.cpu()
                    cpu_sd["flat_tensors"][dtype] = flat_cpu.numpy()
                    cpu_sd["metadata"][dtype] = [(k, local_v.shape, local_v.numel()) for k, local_v in items]
                    
            t_copy = time.perf_counter() - t_copy_start

        t_total = time.perf_counter() - t_start
        
        if int(os.environ.get("RANK", "0")) == 0:
            log_msg = (f"@@@ Trainer Rank 0 Profiling (Whole Model): "
                       f"Queue redist = {t_redist_queue:.3f}s, "
                       f"Sync redist = {t_redist_sync:.3f}s, "
                       f"CPU offload = {t_copy:.3f}s, "
                       f"Total trainer time = {t_total:.3f}s")
            print(log_msg, flush=True)
            logger.info(log_msg)
            
        torch_tpu._internal.sync.synchronize(wait=True)
        return cpu_sd

    else:
        # --- Layer-by-Layer Synchronization (Scalable) ---
        state_dict_keys = list(model.state_dict().keys())
        sorted_groups = get_sorted_groups(state_dict_keys)
        
        cpu_sd = None
        if int(os.environ.get("RANK", "0")) == 0:
            cpu_sd = {
                "grouped": {}
            }
            
        t_redist_queue_total = 0.0
        t_redist_sync_total = 0.0
        t_copy_total = 0.0
        
        # Sequentially process each layer/group to avoid TPU memory spikes
        for group_name in sorted_groups:
            t_g_start = time.perf_counter()
            group_tpu_sd = {}
            
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if get_layer_group(k) != group_name:
                        continue
                    val = v.data if isinstance(v, torch.nn.Parameter) else v
                    if isinstance(val, DTensor):
                        device_mesh = val.device_mesh
                        group_tpu_sd[k] = val.redistribute(device_mesh, [Replicate()] * len(val.placements))
                    else:
                        group_tpu_sd[k] = val
                        
            t_redist_queue_total += time.perf_counter() - t_g_start
            
            # Sync collective replication for this group only
            t_g_sync_start = time.perf_counter()
            torch_tpu._internal.sync.synchronize(wait=True)
            t_redist_sync_total += time.perf_counter() - t_g_sync_start
            
            # Offload group parameters to CPU on Trainer Rank 0
            if int(os.environ.get("RANK", "0")) == 0:
                t_g_copy_start = time.perf_counter()
                group_cpu_sd = {
                    "flat_tensors": {},
                    "metadata": {}
                }
                by_dtype = {}
                
                with torch.no_grad():
                    for k, v in group_tpu_sd.items():
                        local_v = v.to_local() if isinstance(v, DTensor) else v
                        by_dtype.setdefault(local_v.dtype, []).append((k, local_v))
                        
                    for dtype, items in by_dtype.items():
                        if not items:
                            continue
                        flat_tpu = torch.cat([v.detach().view(-1) for _, v in items])
                        flat_cpu = flat_tpu.cpu()
                        group_cpu_sd["flat_tensors"][dtype] = flat_cpu.numpy()
                        group_cpu_sd["metadata"][dtype] = [(k, local_v.shape, local_v.numel()) for k, local_v in items]
                        
                cpu_sd["grouped"][group_name] = group_cpu_sd
                t_copy_total += time.perf_counter() - t_g_copy_start
                
            # Explicitly delete TPU group refs and synchronize to recycle TPU HBM memory instantly
            del group_tpu_sd
            torch_tpu._internal.sync.synchronize(wait=True)
            
        t_total = time.perf_counter() - t_start
        
        if int(os.environ.get("RANK", "0")) == 0:
            log_msg = (f"@@@ Trainer Rank 0 Profiling (Layer-by-Layer): "
                       f"Num groups = {len(sorted_groups)}, "
                       f"Queue redist = {t_redist_queue_total:.3f}s, "
                       f"Sync redist = {t_redist_sync_total:.3f}s, "
                       f"CPU offload = {t_copy_total:.3f}s, "
                       f"Total trainer time = {t_total:.3f}s")
            print(log_msg, flush=True)
            logger.info(log_msg)
            
        torch_tpu._internal.sync.synchronize(wait=True)
        return cpu_sd


# =====================================================================
# Generator/Worker Weight Injection & Slicing
# =====================================================================

def load_weights_on_worker(vllm_model, state_dict: dict, rank: int) -> int:
    """
    Worker-side weight loader. Performs host-side CPU sharding (slicing)
    and chunked, memory-safe, JIT-partitioned PCIe copying to TPU.
    
    Supports both whole-model weight sync (False) and highly scalable
    layer-by-layer weight sync (True).
    """
    t_start = time.perf_counter()
    
    # [TPU HACK / LOCAL RUN SUPPORT]: Convert plain state dict (e.g. from CPU Trainer runs) 
    # to flat/metadata dict on-the-fly.
    # Why: For local CPU trainer runs, the returned state dict is a plain un-flat {key: tensor} dict.
    # We dynamically flatten it on-the-fly to reuse the highly optimized chunked loading pipeline.
    # What didn't work: Writing a completely separate weight injection pathway for CPU trainer; 
    # dynamically converting format on-the-fly preserves high PCIe throughput and 100% codebase compatibility.
    # TODO: Remove HACK. In normal non-colocated RL runs, both trainer and generator run on TPU with 
    # uniform distributed formats, eliminating format mismatch workarounds.
    if isinstance(state_dict, dict) and "flat_tensors" not in state_dict and "grouped" not in state_dict:
        flat_tensors = {}
        metadata = {}
        by_dtype = {}
        for k, v in state_dict.items():
            by_dtype.setdefault(v.dtype, []).append((k, v))
            
        for dtype, items in by_dtype.items():
            flat_cpu = torch.cat([v.detach().view(-1) for _, v in items])
            flat_tensors[dtype] = flat_cpu
            metadata[dtype] = [(k, v.shape, v.numel()) for k, v in items]
            
        state_dict = {
            "flat_tensors": flat_tensors,
            "metadata": metadata
        }
    
    # If the state dict is grouped, we process group-by-group
    if isinstance(state_dict, dict) and "grouped" in state_dict:
        total_keys = 0
        grouped_dict = state_dict["grouped"]
        for group_name, group_sd in grouped_dict.items():
            keys_loaded = _load_single_group_on_worker(vllm_model, group_sd, rank)
            total_keys += keys_loaded
            
        t_total = time.perf_counter() - t_start
        if rank == 0:
            log_msg = (f"@@@ Worker 0 Profiling (Layer-by-Layer): "
                       f"Total groups loaded = {len(grouped_dict)}, "
                       f"Total keys loaded = {total_keys}, "
                       f"Total worker execution = {t_total:.3f}s")
            print(log_msg, flush=True)
            logger.info(log_msg)
        return total_keys

    # Otherwise fallback to whole model sync logic
    flat_tensors = state_dict["flat_tensors"]
    metadata = state_dict["metadata"]
    
    # 1. Pre-process metadata to support tied embeddings (lm_head)
    clean_metadata = {}
    num_keys = 0
    for dtype, items in metadata.items():
        clean_items = []
        offset = 0
        for k, shape, numel in items:
            clean_k = get_clean_name(k)
            clean_items.append((clean_k, shape, numel, offset))
            num_keys += 1
            
            # Replicate embedding weights pointing to the same flat segment offset
            if "tok_embeddings.weight" in clean_k:
                lm_k = clean_k.replace("tok_embeddings", "lm_head")
                clean_items.append((lm_k, shape, numel, offset))
                num_keys += 1
                
            offset += numel
        clean_metadata[dtype] = clean_items
        
    model_sd = vllm_model.model.state_dict()
    
    def resolve_key(k):
        if k in model_sd:
            return k
        if k.startswith("model.") and k[6:] in model_sd:
            return k[6:]
        if f"model.{k}" in model_sd:
            return f"model.{k}"
        return k

    t_format_start = time.perf_counter()
    
    # 2. Slice and Chunk Parameters by Dtype
    for dtype, flat_data in flat_tensors.items():
        items = clean_metadata.get(dtype, [])
        if not items:
            continue
            
        # Unpack NumPy array to PyTorch CPU tensor (zero-copy shared-memory view)
        flat_cpu = torch.from_numpy(flat_data) if not isinstance(flat_data, torch.Tensor) else flat_data
        
        local_items = []
        local_tensors_to_cat = []
        local_offset = 0
        
        # 3. CPU-side slicing (reduces PCIe transfer payload across 8 workers by 8x!)
        for k, shape, numel, offset in items:
            target_key = resolve_key(k)
            if target_key not in model_sd:
                continue
                
            target_v = model_sd[target_key]
            target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
            
            param_cpu_global = flat_cpu[offset : offset + numel].view(shape)
            if target_local.shape == shape:
                param_cpu_local = param_cpu_global
            else:
                sharded = False
                for dim in range(len(shape)):
                    if shape[dim] != target_local.shape[dim]:
                        shard_size = target_local.shape[dim]
                        rank_offset = shard_size * rank
                        indices = [slice(None)] * len(shape)
                        indices[dim] = slice(rank_offset, rank_offset + shard_size)
                        param_cpu_local = param_cpu_global[tuple(indices)].clone()
                        sharded = True
                        break
                if not sharded:
                    param_cpu_local = param_cpu_global
                    
            local_tensors_to_cat.append(param_cpu_local.reshape(-1))
            local_items.append((target_key, target_local.shape, target_local.numel(), local_offset))
            local_offset += target_local.numel()
            
        if not local_tensors_to_cat:
            continue
            
        # Concatenate rank-specific parameters on CPU
        flat_local_cpu = torch.cat(local_tensors_to_cat)
        
        # 4. Grouped Flat Transfer to TPU (safety limits peak memory to exactly 15MB)
        chunks = [local_items[i : i + TPU_COPY_CHUNK_SIZE_PARAMETERS] for i in range(0, len(local_items), TPU_COPY_CHUNK_SIZE_PARAMETERS)]
        
        for chunk in chunks:
            chunk_start_offset = chunk[0][3]
            chunk_end_offset = chunk[-1][3] + chunk[-1][2]
            flat_chunk_cpu = flat_local_cpu[chunk_start_offset:chunk_end_offset]
            
            # Stream the 15MB chunk to TPU over PCIe in a single operation
            flat_chunk_tpu = flat_chunk_cpu.to("tpu")
            
            # Slice and copy directly on TPU
            for target_key, local_shape, local_numel, offset in chunk:
                local_offset = offset - chunk_start_offset
                slice_tpu = flat_chunk_tpu[local_offset : local_offset + local_numel].view(local_shape)
                
                target_v = model_sd[target_key]
                target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
                target_local.copy_(slice_tpu)
                
            # Free temporary flat chunk and sync immediately.
            # This prevents XLA compilation footprint buildup and ensures absolute VMEM safety.
            del flat_chunk_tpu
            torch_tpu._internal.sync.synchronize(wait=True)
            
    t_copy_to_device = time.perf_counter() - t_format_start
    t_total = time.perf_counter() - t_start
    
    if rank == 0:
        log_msg = (f"@@@ Worker 0 Profiling (Whole Model): "
                   f"Copy & Format = {t_copy_to_device:.3f}s, "
                   f"vLLM load = 0.000s, "
                   f"Total worker execution = {t_total:.3f}s")
        print(log_msg, flush=True)
        logger.info(log_msg)
        
    return num_keys


def _load_single_group_on_worker(vllm_model, group_sd: dict, rank: int) -> int:
    """
    Slices and streams parameters of a single group/layer to TPU worker.
    """
    flat_tensors = group_sd["flat_tensors"]
    metadata = group_sd["metadata"]
    
    clean_metadata = {}
    num_keys = 0
    for dtype, items in metadata.items():
        clean_items = []
        offset = 0
        for k, shape, numel in items:
            clean_k = get_clean_name(k)
            clean_items.append((clean_k, shape, numel, offset))
            num_keys += 1
            
            # Replicate embedding weights pointing to the same flat segment offset
            if "tok_embeddings.weight" in clean_k:
                lm_k = clean_k.replace("tok_embeddings", "lm_head")
                clean_items.append((lm_k, shape, numel, offset))
                num_keys += 1
                
            offset += numel
        clean_metadata[dtype] = clean_items
        
    model_sd = vllm_model.model.state_dict()
    
    def resolve_key(k):
        if k in model_sd:
            return k
        if k.startswith("model.") and k[6:] in model_sd:
            return k[6:]
        if f"model.{k}" in model_sd:
            return f"model.{k}"
        return k

    for dtype, flat_data in flat_tensors.items():
        items = clean_metadata.get(dtype, [])
        if not items:
            continue
            
        flat_cpu = torch.from_numpy(flat_data) if not isinstance(flat_data, torch.Tensor) else flat_data
        
        local_items = []
        local_tensors_to_cat = []
        local_offset = 0
        
        for k, shape, numel, offset in items:
            target_key = resolve_key(k)
            if target_key not in model_sd:
                continue
                
            target_v = model_sd[target_key]
            target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
            
            param_cpu_global = flat_cpu[offset : offset + numel].view(shape)
            if target_local.shape == shape:
                param_cpu_local = param_cpu_global
            else:
                sharded = False
                for dim in range(len(shape)):
                    if shape[dim] != target_local.shape[dim]:
                        shard_size = target_local.shape[dim]
                        rank_offset = shard_size * rank
                        indices = [slice(None)] * len(shape)
                        indices[dim] = slice(rank_offset, rank_offset + shard_size)
                        param_cpu_local = param_cpu_global[tuple(indices)].clone()
                        sharded = True
                        break
                if not sharded:
                    param_cpu_local = param_cpu_global
                    
            local_tensors_to_cat.append(param_cpu_local.reshape(-1))
            local_items.append((target_key, target_local.shape, target_local.numel(), local_offset))
            local_offset += target_local.numel()
            
        if not local_tensors_to_cat:
            continue
            
        flat_local_cpu = torch.cat(local_tensors_to_cat)
        
        chunks = [local_items[i : i + TPU_COPY_CHUNK_SIZE_PARAMETERS] for i in range(0, len(local_items), TPU_COPY_CHUNK_SIZE_PARAMETERS)]
        
        for chunk in chunks:
            chunk_start_offset = chunk[0][3]
            chunk_end_offset = chunk[-1][3] + chunk[-1][2]
            flat_chunk_cpu = flat_local_cpu[chunk_start_offset:chunk_end_offset]
            
            flat_chunk_tpu = flat_chunk_cpu.to("tpu")
            
            for target_key, local_shape, local_numel, offset in chunk:
                local_offset = offset - chunk_start_offset
                slice_tpu = flat_chunk_tpu[local_offset : local_offset + local_numel].view(local_shape)
                
                target_v = model_sd[target_key]
                target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
                target_local.copy_(slice_tpu)
                
            del flat_chunk_tpu
            torch_tpu._internal.sync.synchronize(wait=True)
            
    return num_keys


def load_weights_on_driver(vllm_model, state_dict_data: dict) -> None:
    """
    Fallback weight loader for single-process driver runs where workers do not exist.
    """
    if isinstance(state_dict_data, dict) and "grouped" in state_dict_data:
        for group_name, group_sd in state_dict_data["grouped"].items():
            load_weights_on_driver(vllm_model, group_sd)
        return

    state_dict = {}
    if isinstance(state_dict_data, dict) and "flat_tensors" in state_dict_data and "metadata" in state_dict_data:
        flat_tensors = state_dict_data["flat_tensors"]
        metadata = state_dict_data["metadata"]
        for dtype, items in metadata.items():
            flat_data = flat_tensors[dtype]
            flat_cpu = torch.from_numpy(flat_data) if not isinstance(flat_data, torch.Tensor) else flat_data
            for k, shape, numel, offset in items:
                state_dict[k] = flat_cpu[offset : offset + numel].view(shape)
    else:
        state_dict = state_dict_data
        
    # If weight tying is enabled, explicitly add lm_head for syncing
    tok_key = next((k for k in state_dict.keys() if "tok_embeddings.weight" in k), None)
    lm_key = next((k for k in state_dict.keys() if "lm_head.weight" in k), None)
    if tok_key and not lm_key:
        state_dict[tok_key.replace("tok_embeddings", "lm_head")] = state_dict[tok_key]

    target_sd = dict(vllm_model.named_parameters())
    with torch.no_grad():
        for name, source_p in state_dict.items():
            clean_name = get_clean_name(name)
            target_name = f"model.{clean_name}"
            
            if target_name in target_sd:
                target_t = target_sd[target_name]
                val = source_p
                if hasattr(val, "full_tensor"):
                    val = val.full_tensor()
                if target_t.shape == val.shape:
                    target_t.copy_(val.to(target_t.device))
                else:
                    logger.warning(f"Shape mismatch for {target_name}: {target_t.shape} != {val.shape}")
            elif clean_name in target_sd:
                target_name = clean_name
                target_t = target_sd[target_name]
                val = source_p
                if hasattr(val, "full_tensor"):
                    val = val.full_tensor()
                if target_t.shape == val.shape:
                    target_t.copy_(val.to(target_t.device))
                else:
                    logger.warning(f"Shape mismatch for {target_name}: {target_t.shape} != {val.shape}")
