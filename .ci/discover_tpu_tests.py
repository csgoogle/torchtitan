import json
import math
import sys
from pathlib import Path

# Number of shards to split the tests into
NUM_SHARDS = 4

# Tests that are temporarily disabled
DISABLED_TESTS = {
    "torchtitan.experiments.tpu.kernels.tests.test_splash_attention",
    "torchtitan.experiments.tpu.kernels.tests.test_tokamax_splash_attention",
    "torchtitan.experiments.tpu.kernels.tests.test_linear_softmax_cross_entropy_loss",
    "torchtitan.experiments.tpu.kernels.tests.test_gmm",
    "torchtitan.experiments.tpu.kernels.tests.test_fill_indices",
    "torchtitan.experiments.tpu.kernels.tests.test_segment_matmul",
    "torchtitan.experiments.tpu.qwen3.tests.test_qwen3",
    "torchtitan.experiments.tpu.redacted.tests.test_kernels",
    # TODO: Unknown command line flag 'device'
    "torchtitan.experiments.tpu.conformer.tests.test_jax_ctc_loss",
    # TODO: unrecognized arguments: --device=tpu
    "torchtitan.experiments.tpu.rl.tests.test_grpo",
    "torchtitan.experiments.tpu.rl.tests.test_grpo_sampler",
    "torchtitan.experiments.tpu.rl.tests.test_grpo_token",
    # TODO: ModuleNotFoundError: No module named 'internal'
    "torchtitan.experiments.tpu.tests.test_profiler",
    # TODO: SyntaxError: invalid syntax (from jaxtyping)
    "torchtitan.experiments.tpu.redacted.tests.test_train_minimal_distributed",
    # TODO: Unknown command line flag 'device'
    "torchtitan.experiments.tpu.tests.test_workarounds",
    # TODO: AssertionError: test_deepseek_complete_model_training_steps_cpu_device_parity_MOE
    "torchtitan.experiments.tpu.deepseek_v3.tests.test_deepseek_v3",
}

def discover_tests():
    base_dir = Path("torchtitan/experiments/tpu")
    
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    test_modules = []
    
    # Find all test files recursively
    for path in base_dir.rglob("test_*.py"):
        # Convert path to module string (e.g. torchtitan.experiments.tpu.tests.test_dtensor)
        # We replace '/' with '.' and remove the '.py' extension
        module = str(path.with_suffix("")).replace("/", ".")
        
        if module not in DISABLED_TESTS:
            test_modules.append(module)
            
    # Sort for deterministic behavior
    test_modules.sort()
    return test_modules

def chunk_tests(tests, num_shards):
    tests_per_shard = math.ceil(len(tests) / num_shards)
    shards = []
    
    for i in range(num_shards):
        start_idx = i * tests_per_shard
        end_idx = min((i + 1) * tests_per_shard, len(tests))
        shard_tests = tests[start_idx:end_idx]
        
        if shard_tests:
            # Store tests as a space-separated string so bash can iterate easily
            shards.append({
                "shard": f"shard_{i+1}",
                "modules": " ".join(shard_tests)
            })
            
    return shards

def main():
    tests = discover_tests()
    
    if not tests:
        print("Error: No tests found", file=sys.stderr)
        sys.exit(1)
        
    shards = chunk_tests(tests, NUM_SHARDS)
    
    # We output a JSON array of shard objects to stdout
    print(json.dumps(shards))

if __name__ == "__main__":
    main()
