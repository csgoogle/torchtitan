import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["SKIP_JAX_PRECOMPILE"] = "1"
os.environ["JAX_PLATFORMS"] = "tpu"

import torch
import torch_tpu
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm import SamplingParams

def main():
    engine_args = EngineArgs(
        model="Qwen/Qwen3-0.6B", 
        enforce_eager=False, 
        gpu_memory_utilization=0.5,
        max_model_len=1024,
        disable_log_stats=True,
        tensor_parallel_size=1,
    )
    print("Initializing vLLM Engine...")
    engine = LLMEngine.from_engine_args(engine_args)

    prompt = "You are a helpful assistant. Solve the problem step by step."
    sampling_params = SamplingParams(temperature=1.0, max_tokens=50)
    engine.add_request("0", {"prompt": prompt}, sampling_params)

    print("Starting generation loop...")
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            if output.finished:
                print(f"Output: {output.outputs[0].text}")

if __name__ == "__main__":
    main()
