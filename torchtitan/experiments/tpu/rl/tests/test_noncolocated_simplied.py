"""
Non-colocated RL dummy example:
Trainer on CPU and only do forward pass to get loss
Sampler on TPU and only test with tp=1

To run:
    sudo fuser -k /dev/vfio/*
    export PYTHONPATH=$PYTHONPATH:.
    python torchtitan/experiments/tpu/rl/tests/test_noncolocated_simplied.py > non_colocated.log 2>&1 
"""
import os
import torch
import torch.nn.functional as F
import ray

from torchtitan.models.qwen3 import model_registry, Qwen3Model
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from transformers import AutoTokenizer
from safetensors.torch import load_file

@ray.remote(resources={"TPU": 1})
class VLLMWorker:
    def __init__(self, model_path: str, tp_size: int = 1):
        import json
        from vllm import LLM
        from torchtitan.models.qwen3 import model_registry
        from torchtitan.experiments.rl.models.vllm_registry import register_model_to_vllm_model_registry, VLLM_MODEL_NAME
        from torchtitan.config.configs import CompileConfig

        # Use the Torchtitan custom model registry for vLLM so parameter names match exactly
        model_spec = model_registry("0.6B")
        register_model_to_vllm_model_registry(model_spec, compile_config=CompileConfig(enable=False))

        def get_config_dict():
            return {
                "architectures": [VLLM_MODEL_NAME],
                "model_type": "llama",
                "vocab_size": 151936,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "hidden_size": 1024,
                "max_position_embeddings": 4096,
                "intermediate_size": 3072,
            }

        dummy_model_path = "/tmp/torchtitan-qwen3-0.6B-minimal"
        os.makedirs(dummy_model_path, exist_ok=True)
        config_path = os.path.join(dummy_model_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(get_config_dict(), f)

        # Initialize LLM with dummy weights
        self.llm = LLM(
            model=dummy_model_path,
            tokenizer=model_path,
            max_model_len=256,
            max_num_batched_tokens=256,
            max_num_seqs=32,
            tensor_parallel_size=tp_size,
            load_format="dummy",
            hf_overrides={"architectures": [VLLM_MODEL_NAME]},
            enforce_eager=False,
            enable_prefix_caching=False,
        )
    
    def update_weights(self, state_dict: dict):
        import torch_tpu
        vllm_model = self.llm.llm_engine.model_executor.driver_worker.get_model()
        
        target_sd = dict(vllm_model.named_parameters())
        
        # If weight tying is enabled, explicitly add lm_head for syncing
        tok_key = next((k for k in state_dict.keys() if "tok_embeddings.weight" in k), None)
        lm_key = next((k for k in state_dict.keys() if "lm_head.weight" in k), None)
        if tok_key and not lm_key:
            state_dict[tok_key.replace("tok_embeddings", "lm_head")] = state_dict[tok_key]

        def get_local_vllm_tensor(source_p, target_t_dtensor):
            val = source_p
            if hasattr(val, "full_tensor"):
                full_val = val.full_tensor()
            else:
                full_val = val
                
            if hasattr(target_t_dtensor, "device_mesh") and hasattr(target_t_dtensor, "placements"):
                from torch.distributed.tensor import DTensor, Replicate
                val_dtensor = DTensor.from_local(
                    full_val.to(target_t_dtensor.device),
                    device_mesh=target_t_dtensor.device_mesh,
                    placements=[Replicate()] * target_t_dtensor.device_mesh.ndim,
                )
                val_dtensor_sharded = val_dtensor.redistribute(
                    device_mesh=target_t_dtensor.device_mesh,
                    placements=target_t_dtensor.placements,
                )
                return val_dtensor_sharded.to_local()
            else:
                if hasattr(target_t_dtensor, "to_local"):
                    target_local = target_t_dtensor.to_local()
                else:
                    target_local = target_t_dtensor
                    
                if target_local.shape == full_val.shape:
                    return full_val
                    
                for dim in range(len(target_local.shape)):
                    if target_local.shape[dim] != full_val.shape[dim]:
                        shard_size = target_local.shape[dim]
                        rank_offset = 0  # tp_size=1
                        indices = [slice(None)] * len(target_local.shape)
                        indices[dim] = slice(rank_offset, rank_offset + shard_size)
                        return full_val[tuple(indices)]
                return full_val

        synced_count = 0
        with torch.no_grad():
            for name, source_p in state_dict.items():
                clean_name = name.replace("_fsdp_wrapped_module.", "") \
                                 .replace("_checkpoint_wrapped_module.", "") \
                                 .replace("module.", "")
                
                target_name = f"model.{clean_name}"
                
                if target_name in target_sd:
                    target_t_dtensor = target_sd[target_name]
                    target_t = target_t_dtensor.to_local() if hasattr(target_t_dtensor, "to_local") else target_t_dtensor
                    val = get_local_vllm_tensor(source_p, target_t_dtensor)
                    
                    if target_t.shape == val.shape:
                        target_t.copy_(val.to(target_t.device))
                        synced_count += 1
                    else:
                        print(f"Shape mismatch for {target_name}: target {target_t.shape} != source {val.shape}")
                else:
                    if clean_name in target_sd:
                        target_name = clean_name
                        target_t_dtensor = target_sd[target_name]
                        target_t = target_t_dtensor.to_local() if hasattr(target_t_dtensor, "to_local") else target_t_dtensor
                        
                        val = get_local_vllm_tensor(source_p, target_t_dtensor)
                        if target_t.shape == val.shape:
                            target_t.copy_(val.to(target_t.device))
                            synced_count += 1
                        else:
                            print(f"Shape mismatch for {target_name}: target {target_t.shape} != source {val.shape}")
                    
        # Synchronize XLA device to ensure all copy operations finish
        torch_tpu._internal.sync.synchronize(wait=True)
        return synced_count

    def generate(self, prompts: list[str]) -> list[str]:
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

@ray.remote(num_cpus=1)
class TrainerWorker:
    def __init__(self, model_path: str):
        from torchtitan.models.qwen3 import model_registry, Qwen3Model
        from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
        from transformers import AutoTokenizer
        from safetensors.torch import load_file
        import torch
        import os
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("@@@ [TrainerWorker] Initializing Torchtitan Model on CPU for Trainer forward pass...")
        model_spec = model_registry("0.6B")
        model_args = model_spec.model
        self.model = Qwen3Model(model_args)
        # Using bfloat16 to perfectly align with safe_tensors format
        self.model.to(torch.bfloat16)
        self.model.init_states(buffer_device=torch.device("cpu"))
        
        print(f"@@@ [TrainerWorker] Loading real weights from {model_path} into CPU model...")
        adapter = Qwen3StateDictAdapter(model_args, None)
        hf_state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        torchtitan_state_dict = adapter.from_hf(hf_state_dict)
        
        # Strict=False because we don't have DTensors here, just normal tensors
        self.model.load_state_dict(torchtitan_state_dict, strict=False)
        self.model.eval()
        
    def get_weights(self):
        """Returns the CPU state dict to be sent over Ray to vLLM worker."""
        # CPU Model's weights mapped to basic CPU tensors to send over Ray
        # Using state_dict() so all necessary keys (including buffers/tied weights) are provided if needed
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
        
    def evaluate_loss(self, prompt: str, generated_text: str) -> float:
        import torch
        import torch.nn.functional as F
        
        # Ensure generated text is not empty, fallback to space to avoid crashes
        if not generated_text:
            generated_text = " "

        final_text = prompt + generated_text
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt")
        full_tokens = self.tokenizer.encode(final_text, return_tensors="pt")
        
        prompt_len = prompt_tokens.shape[1]
        
        labels = full_tokens.clone()
        labels[0, :prompt_len] = -100
        
        with torch.no_grad():
            logits = self.model(full_tokens)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                ignore_index=-100
            )
        return loss.item()
        
    def get_token_ids(self, text: str):
        return self.tokenizer.encode(text)


def get_prompt():
    return """You are a helpful assistant. Solve the problem step by step.
When you have your final answer, state it as [ANSWER] <number>.

Example:
User: What is the total digit sum of [12, 345, 67]?
Assistant: Break each number into digits:
12 -> 1, 2
345 -> 3, 4, 5
67 -> 6, 7
Sum all digits: 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28
[ANSWER] 28

What is the total digit sum of [10, 93, 53]?"""

def main():
    # Keep main process on CPU
    os.environ["JAX_PLATFORMS"] = "cpu"
    ray.init()
    
    model_path = "/home/jialeic_google_com/work/rl_poc/torchtitan/assets/hf/Qwen3-0.6B"
    
    print("@@@ Creating Trainer Worker on CPU via Ray...")
    trainer_worker = TrainerWorker.options(
        runtime_env={
            "env_vars": {
                "JAX_PLATFORMS": "cpu"
            }
        }
    ).remote(model_path)

    print("\n@@@ Creating vLLM Worker on TPU via Ray with DUMMY weights...")
    vllm_worker = VLLMWorker.options(
        runtime_env={
            "env_vars": {
                "JAX_PLATFORMS": "tpu",
                "SKIP_JAX_PRECOMPILE": "1",
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0"
            }
        }
    ).remote(model_path, tp_size=1)
    
    # Wait for models to initialize
    ray.get([trainer_worker.evaluate_loss.remote("Test", " Test"), vllm_worker.generate.remote(["Test"])])
    
    prompt = get_prompt()
    
    print("\nEvaluating prompt loss on CPU Trainer...")
    loss_prompt = ray.get(trainer_worker.evaluate_loss.remote(prompt, ""))
    print(f"CPU Prompt Loss: {loss_prompt}")
    
    # --- PHASE 1: GENERATE WITH DUMMY WEIGHTS ---
    print("\n" + "="*50)
    print("PHASE 1: GENERATING WITH DUMMY WEIGHTS")
    print("="*50)
    nonsense_texts = ray.get(vllm_worker.generate.remote([prompt]))
    nonsense_text = nonsense_texts[0]
    print(f"Output:\n{nonsense_text!r}\n")
    
    loss_nonsense = ray.get(trainer_worker.evaluate_loss.remote(prompt, nonsense_text))
    print(f"@@@ Trainer CPU Model Loss (Dummy): {loss_nonsense:.4f}")
    assert loss_nonsense > 5.0, "Expected high loss for dummy weights!"
    
    # --- PHASE 2: SYNC WEIGHTS ---
    print("\n" + "="*50)
    print("PHASE 2: SYNCING REAL WEIGHTS TO vLLM")
    print("="*50)
    cpu_sd = ray.get(trainer_worker.get_weights.remote())
    synced_count = ray.get(vllm_worker.update_weights.remote(cpu_sd))
    print(f"@@@ Successfully synced {synced_count} parameter tensors over Ray.")
    
    # --- PHASE 3: GENERATE WITH REAL WEIGHTS ---
    print("\n" + "="*50)
    print("PHASE 3: GENERATING WITH REAL WEIGHTS")
    print("="*50)
    sensible_texts = ray.get(vllm_worker.generate.remote([prompt]))
    sensible_text = sensible_texts[0]
    print(f"Output:\n{sensible_text!r}\n")
    
    token_ids = ray.get(trainer_worker.get_token_ids.remote(sensible_text))
    print(f"Token IDs:\n{token_ids}")
    
    loss_sensible = ray.get(trainer_worker.evaluate_loss.remote(prompt, sensible_text))
    print(f"@@@ Trainer CPU Model Loss (Real): {loss_sensible:.4f}")
    assert loss_sensible < 2.0, "Expected low loss for real weights!"
    
    print("@@@ Success!")

if __name__ == "__main__":
    main()
