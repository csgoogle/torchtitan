import contextlib
import json
import logging
import os
import unittest.mock

import torch
from torch.distributed import fsdp

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"

@contextlib.contextmanager
def mock_vllm_distributed(tp_size: int):
    """Mocks vLLM's UniProcExecutor to pass the correct LOCAL_RANK instead of 0.
    
    By default, when initializing vLLM with tensor_parallel_size=1, it uses 
    UniProcExecutor which forcibly assumes the process is running on LOCAL_RANK=0.
    In the collocated TPU PyTorch distributed environment (launched via torchrun),
    workers run on LOCAL_RANK=0, 1, 2, 3, etc. This mock intercepts the initialization
    to pass the actual LOCAL_RANK from environment variables, preventing conflicts
    and crashes.

    We will remove this with Ray-based multiprocessing in the future, 
    but for now this allows us to use vLLM in our collocated TPU setup.
    """
    if tp_size > 1:
        yield
        return
        
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    
    original_distributed_args = UniProcExecutor._distributed_args
    
    def mock_distributed_args(self):
        distributed_init_method, rank, local_rank = original_distributed_args(self)
        actual_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return distributed_init_method, actual_local_rank, actual_local_rank

    with unittest.mock.patch.object(UniProcExecutor, '_distributed_args', new=mock_distributed_args):
        yield

@contextlib.contextmanager
def suppress_vllm_tpu_dummy_init_crash():
    """
    Mocks vLLM's dummy weight initialization to suppress a TPU-specific crash.

    When `load_format="dummy"` is used, vLLM allocates dummy tensors and
    internally calls `torch._sync(param)` on them. On PyTorch TPU, calling
    `torch._sync()` on these newly created dummy functional tensors triggers
    an internal C++ assertion (`isFunctionalTensor(t) INTERNAL ASSERT FAILED`).

    This mock allows vLLM to allocate the dummy memory buffers it needs for
    profiling and KV cache allocation, but safely catches and suppresses the
    TPU crash that happens at the very end of the tensor creation.
    
    Once the vLLM engine finishes starting up, our `sync_weights()` method
    is called to overwrite these dummy buffers with the real initialized
    weights from the TorchTitan FSDP model, completely bypassing the issue.
    """
    import unittest.mock
    try:
        import vllm.model_executor.model_loader.weight_utils as vllm_weight_utils
    except ImportError:
        yield
        return

    original_init_dummy = vllm_weight_utils.initialize_single_dummy_weight
    
    def mock_init_dummy(param, low, high, seed):
        try:
            return original_init_dummy(param, low, high, seed)
        except RuntimeError as e:
            if "at::functionalization::impl::isFunctionalTensor(t) INTERNAL ASSERT FAILED" in str(e):
                pass
            else:
                raise e

    with unittest.mock.patch('vllm.model_executor.model_loader.weight_utils.initialize_single_dummy_weight', mock_init_dummy):
        yield

class VLLMSampler:
    """
    Handles sampling (generation) using vLLM for GRPO training on TPU.
    
    This class wraps the vLLM engine, dynamically configuring it to match the
    TorchTitan model architecture. During the training step, it pulls the FSDP 
    weights into its internal cache to generate rollouts, avoiding the overhead 
    of loading real weights from disk during initialization.
    """
    def __init__(self, job_config):
        self.job_config = job_config
        self.vllm_engine = None
        self._init_vllm()

    def _get_config_dict_from_vllm_registry(self):
        """Extract actual config dimensions for accurate KV cache allocation."""
        from torchtitan.experiments.rl.models.vllm_registry import VLLM_MODEL_NAME
        model_args = getattr(self.job_config.model_spec, "model", None) if hasattr(self.job_config, 'model_spec') else None
        layers = getattr(model_args, "layers", []) if model_args else []
        first_layer_attn = getattr(layers[0], "attention", None) if isinstance(layers, list) and layers else None

        return {
            "architectures": [VLLM_MODEL_NAME],
            "model_type": "llama",
            "vocab_size": getattr(model_args, "vocab_size", 256000) if model_args else 256000,
            "num_hidden_layers": len(layers) if isinstance(layers, list) and layers else getattr(model_args, "n_layers", 4) if model_args else 4,
            "num_attention_heads": getattr(first_layer_attn, "n_heads", 16) if first_layer_attn else 16,
            "num_key_value_heads": getattr(first_layer_attn, "n_kv_heads", 2) if first_layer_attn else 2,
            "hidden_size": getattr(model_args, "dim", 1024) if model_args else 1024,
            "max_position_embeddings": getattr(getattr(model_args, "rope", None), "max_seq_len", 1024) if model_args else 1024
        }

    def _get_tokenizer_name(self):
        """Determine tokenizer name from job configuration."""
        # Use the initial load path if provided, fallback to hf_assets_path
        if hasattr(self.job_config, 'checkpoint') and self.job_config.checkpoint.initial_load_path:
            return self.job_config.checkpoint.initial_load_path
            
        if hasattr(self.job_config, 'hf_assets_path') and self.job_config.hf_assets_path:
            return self.job_config.hf_assets_path
            
        return DEFAULT_TOKENIZER_NAME

    def _create_dummy_model_config(self):
        """Create a dummy config.json for vLLM to read dimensions from."""
        vllm_model_str = f"torchtitan-{self.job_config.model.name}-{self.job_config.model.flavor}"
        dummy_model_path = f"/tmp/{vllm_model_str}"
        os.makedirs(dummy_model_path, exist_ok=True)
        config_path = os.path.join(dummy_model_path, "config.json")
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(self._get_config_dict_from_vllm_registry(), f)
        return dummy_model_path

    def _init_vllm(self):
        tp_size = getattr(self.job_config.sampler, "vllm_tensor_parallel_size", 1)
        logger.info(f"Initializing vLLM engine (Data Parallel mode, TP={tp_size})...")
        
        os.environ['SKIP_JAX_PRECOMPILE'] = '1'
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0' if tp_size == 1 else '1'

        # Use model name from config.
        try:
            from torchtitan.experiments.rl.models.vllm_registry import register_model_to_vllm_model_registry
            if hasattr(self.job_config, 'model_spec'):
                compile_config = getattr(self.job_config, 'compile', None)
                register_model_to_vllm_model_registry(self.job_config.model_spec, compile_config=compile_config)
        except Exception as e:
            logger.warning(f"Could not register model to vLLM registry: {e}")

        dummy_model_path = self._create_dummy_model_config()
        tokenizer_name = self._get_tokenizer_name()

        try:
            from vllm import LLM
            from torchtitan.experiments.rl.models.vllm_registry import VLLM_MODEL_NAME
            
            with mock_vllm_distributed(tp_size), suppress_vllm_tpu_dummy_init_crash():
                self.vllm_engine = LLM(
                    model=dummy_model_path,
                    tokenizer=tokenizer_name,
                    enforce_eager=self.job_config.sampler.vllm_enforce_eager, 
                    max_model_len=self.job_config.sampler.vllm_max_model_len, 
                    tensor_parallel_size=tp_size,
                    load_format="dummy",
                    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
                    gpu_memory_utilization=self.job_config.sampler.vllm_gpu_memory_utilization,
                    disable_log_stats=True
                )
            logger.info("vLLM engine initialized successfully.")
        except Exception as e:
            import traceback
            logger.error(f"Failed to initialize vLLM: {e}")
            logger.error(traceback.format_exc())
            self.vllm_engine = None

    def sync_weights(self, source_model: torch.nn.Module):
        """Syncs the torchtitan model weights to the vLLM engine.
        
        How weight synchronization works:
        1. FSDP Gather (ICI Used): It uses FSDP's `summon_full_params` to gather the sharded 
           model weights across all FSDP ranks into full parameters. This triggers an 
           all-gather communication operation across the TPUs, which utilizes the high-speed 
           Inter-Chip Interconnect (ICI) network.
        2. Direct Copy (No CPU Transfer): It extracts the state dictionaries for both the 
           source model and the target vLLM model. It then performs a direct, in-place copy 
           (`target_p.copy_(val)`). Since both models reside on the TPU device, this copy 
           happens entirely on-device. It does NOT offload to the CPU, meaning it bypasses 
           the host CPU memory and avoids slow PCIe transfers.
        3. Memory Management: To prevent Out-Of-Memory (OOM) issues on the TPU during the 
           gathering phase, it eagerly clears references from the source state dictionary 
           immediately after each parameter is copied.
        
        TODO: Make this per-layer instead of summoning all parameters at once. Currently, 
        `summon_full_params(recurse=True)` gathers the entire model into HBM simultaneously, 
        which causes a massive memory spike. A per-layer iteration would keep the memory 
        overhead small and bounded.
        """
        if self.vllm_engine is None:
            return
            
        logger.info("Syncing weights to vLLM engine...")
        
        vllm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.get_model()
        
        with fsdp.FullyShardedDataParallel.summon_full_params(
            source_model, recurse=True, writeback=False
        ):
            # Use state_dict but iterate it immediately to avoid keeping large dict alive
            source_sd = source_model.state_dict()
            
            # vLLM model is a wrapper around the actual model
            target_model = getattr(vllm_model, "model", vllm_model)
            target_sd = target_model.state_dict()

            with torch.no_grad():
                for name, source_p in source_sd.items():
                    if name in target_sd:
                        target_p = target_sd[name]
                        val = source_p
                        if hasattr(val, "full_tensor"):
                            val = val.full_tensor()
                        elif hasattr(val, "to_local"):
                            val = val.to_local()
                        target_p.copy_(val)
                    source_sd[name] = None # Clear memory immediately
            del source_sd
            del target_sd

    def generate(self, prompt_ids_repeated: torch.Tensor, sampling_params) -> tuple[torch.Tensor, torch.Tensor]:
        if self.vllm_engine is None:
            raise RuntimeError("vLLM engine is not initialized.")
            
        sampling_params.logprobs = 1
        
        # TODO: avoid moving to CPU
        prompt_ids_list = prompt_ids_repeated.cpu().tolist()
        prompts = [{"prompt_token_ids": p} for p in prompt_ids_list]
        
        outputs = self.vllm_engine.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        
        completions = []
        token_log_probs_list = []
        
        for output in outputs:
            completion_output = output.outputs[0]
            completions.append(completion_output.token_ids)
            
            step_logprobs = []
            for i, token_id in enumerate(completion_output.token_ids):
                pos_logprobs = completion_output.logprobs[i]
                if token_id in pos_logprobs:
                    step_logprobs.append(pos_logprobs[token_id].logprob)
                else:
                    step_logprobs.append(0.0)
            token_log_probs_list.append(step_logprobs)
            
        completions_tensor = torch.tensor(completions, device=prompt_ids_repeated.device, dtype=torch.long)
        completed_ids = torch.cat([prompt_ids_repeated, completions_tensor], dim=1)
        token_log_probs = torch.tensor(token_log_probs_list, device=prompt_ids_repeated.device, dtype=torch.float32)
                                      
        return completed_ids, token_log_probs
