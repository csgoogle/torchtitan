import argparse
import os
import ray

from vllm import EngineArgs, LLMEngine, SamplingParams
from tpu_inference.core import disagg_utils
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def create_parser():
    parser = argparse.ArgumentParser(description="Minimal vLLM TPU Ray 1-chip worker test")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="Tensor parallel degree")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path")
    return parser


@ray.remote(resources={"TPU": 1})
class RayVLLM1ChipWorker:
    """Ray Worker representing a single vLLM rank on 1 TPU chip."""

    def __init__(self, rank: int, world_size: int, master_addr: str, master_port: int, model: str):
        os.environ['SKIP_JAX_PRECOMPILE'] = '1'
        os.environ["VLLM_TARGET_DEVICE"] = "tpu"
        os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
        os.environ["VLLM_TPU_MULTIHOST_BACKEND"] = "ray"
        
        # Configure distributed environment variables for external_launcher
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        # Configure torch_tpu slicebuilder addresses and topology for TPU backend
        if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
            ports = [master_port + i for i in range(world_size)]
            os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join([f"{master_addr}:{p}" for p in ports])
        if "TORCH_TPU_TOPOLOGY" not in os.environ:
            os.environ["TORCH_TPU_TOPOLOGY"] = "2,2,1"  # Default 4-chip VM topology

        # Permanent monkeypatching for disagg engine core in the Ray worker process
        if disagg_utils.is_disagg_enabled():
            logger.info(f"Rank {rank}: Disagg enabled - Monkeypatching vLLM EngineCore with DisaggEngineCore in Ray worker")
            from tpu_inference.core.core_tpu import DisaggEngineCore, DisaggEngineCoreProc
            import vllm.v1.engine.core
            vllm.v1.engine.core.EngineCore = DisaggEngineCore
            vllm.v1.engine.core.EngineCoreProc = DisaggEngineCoreProc

        logger.info(f"Rank {rank}: Initializing vLLM LLMEngine ({model}, TP={world_size}) via external_launcher...")
        engine_args = EngineArgs(
            model=model,
            tensor_parallel_size=world_size,
            trust_remote_code=True,
            enforce_eager=True,
            distributed_executor_backend="external_launcher",
            disable_log_stats=True,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

    def generate(self, prompts: list[str]) -> list[dict]:
        logger.info(f"Rank {os.environ['RANK']}: Adding {len(prompts)} requests to LLMEngine...")
        for i, prompt in enumerate(prompts):
            self.engine.add_request(str(i), prompt, self.sampling_params)

        all_outputs = []
        while self.engine.has_unfinished_requests():
            all_outputs.extend(self.engine.step())

        # Sort outputs by request ID to maintain original prompt order
        all_outputs.sort(key=lambda o: int(o.request_id))

        results = []
        for output in all_outputs:
            results.append({
                "prompt": output.prompt,
                "generated_text": output.outputs[0].text
            })
        return results


def main():
    if not ray.is_initialized():
        ray.init()

    parser = create_parser()
    args = parser.parse_args()
    world_size = args.tensor_parallel_size

    master_addr = ray.util.get_node_ip_address()
    master_port = 29600

    logger.info(f"Deploying {world_size} RayVLLM1ChipWorker actors (each requesting TPU: 1, Model: {args.model})...")
    workers = [
        RayVLLM1ChipWorker.remote(
            rank=i,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
            model=args.model
        ) for i in range(world_size)
    ]

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The colors of the rainbow are",
        "The future of AI is",
    ]

    logger.info(f"Submitting {len(prompts)} prompts to all 1-chip vLLM workers collectively...")
    # In external_launcher mode, all ranks must execute generate collectively in sync
    futures = [worker.generate.remote(prompts) for worker in workers]
    results = ray.get(futures)

    # All workers produce identical results, so we print Rank 0's output
    print("-" * 50)
    for output in results[0]:
        print(f"Prompt: {output['prompt']!r}\nGenerated text: {output['generated_text']!r}")
        print("-" * 50)


if __name__ == "__main__":
    main()
