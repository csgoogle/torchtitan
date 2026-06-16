import argparse
import os
import ray

from vllm import LLM, SamplingParams
from tpu_inference.core import disagg_utils
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def create_parser():
    parser = argparse.ArgumentParser(description="Minimal vLLM TPU Ray test")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="Tensor parallel degree")
    parser.add_argument("--data-parallel-size", type=int, default=1, help="Data parallel degree")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path")
    return parser


@ray.remote
class RayVLLMTPUActor:
    """Minimal Ray Actor wrapping vLLM execution on TPUs."""

    def __init__(self, model: str, tensor_parallel_size: int, data_parallel_size: int):
        os.environ['SKIP_JAX_PRECOMPILE'] = '1'

        # Permanent monkeypatching for disagg engine core in the Ray worker process
        if disagg_utils.is_disagg_enabled():
            logger.info("Disagg enabled: Monkeypatching vLLM EngineCore with DisaggEngineCore in Ray worker")
            from tpu_inference.core.core_tpu import DisaggEngineCore, DisaggEngineCoreProc
            import vllm.v1.engine.core
            vllm.v1.engine.core.EngineCore = DisaggEngineCore
            vllm.v1.engine.core.EngineCoreProc = DisaggEngineCoreProc

        logger.info(f"Initializing vLLM LLM Engine ({model}, TP={tensor_parallel_size}, DP={data_parallel_size}) on TPU host inside Ray Actor...")
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            trust_remote_code=True,
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

    def generate(self, prompts: list[str]) -> list[dict]:
        logger.info(f"Generating completions for {len(prompts)} prompts...")
        outputs = self.llm.generate(prompts, self.sampling_params)

        results = []
        for output in outputs:
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
    total_chips = args.tensor_parallel_size * args.data_parallel_size

    logger.info(f"Deploying RayVLLMTPUActor to TPU host (requesting TPU: {total_chips}, Model: {args.model})...")
    vllm_actor = RayVLLMTPUActor.options(resources={"TPU": total_chips}).remote(
        model=args.model, 
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size
    )

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The colors of the rainbow are",
        "The future of AI is",
    ]

    logger.info(f"Submitting {len(prompts)} prompts to remote TPU vLLM actor...")
    outputs = ray.get(vllm_actor.generate.remote(prompts))

    print("-" * 50)
    for output in outputs:
        print(f"Prompt: {output['prompt']!r}\nGenerated text: {output['generated_text']!r}")
        print("-" * 50)


if __name__ == "__main__":
    main()
