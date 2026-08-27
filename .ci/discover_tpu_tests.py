import ast
import json
import math
import sys
from pathlib import Path

NUM_DISTRIBUTED_TPU_SHARDS = 4
NUM_SINGLE_TPU_SHARDS = 2
NUM_CPU_SHARDS = 1

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
    # RL Ray non-colocated tests (require multi-host Ray TPU cluster)
    "torchtitan.experiments.tpu.rl_nc_ray.tests.test_mini_rl",
    "torchtitan.experiments.tpu.rl_nc_ray.tests.test_varlen_attn_tpu",
    # TODO: ModuleNotFoundError: No module named 'internal'
    "torchtitan.experiments.tpu.tests.test_profiler",
    # TODO: add copybara handling for this test
    "torchtitan.experiments.tpu.kernels.tests.test_kernel_name_determinism",
    # TODO: SyntaxError: invalid syntax (from jaxtyping)
    "torchtitan.experiments.tpu.redacted.tests.test_train_minimal_distributed",
    # TODO: Unknown command line flag 'device'
    "torchtitan.experiments.tpu.tests.test_workarounds",
    # TODO: AssertionError: test_deepseek_complete_model_training_steps_cpu_device_parity_MOE
    "torchtitan.experiments.tpu.deepseek_v3.tests.test_deepseek_v3",
}

def classify_test(file_path):
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read(), filename=str(file_path))
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                else:
                    continue
                
                if "BaseDistributedDeviceTest" in base_name:
                    return "distributed"
                elif "BaseAcceleratorDeviceTest" in base_name:
                    return "single_tpu"
                elif base_name in ["TestCase"]:
                    return "cpu"
    return "cpu" # Default fallback

def discover_tests():
    base_dir = Path("torchtitan/experiments/tpu")
    
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    categorized_tests = {
        "distributed": [],
        "single_tpu": [],
        "cpu": []
    }
    
    # Find all test files recursively
    for path in base_dir.rglob("test_*.py"):
        # Convert path to module string (e.g. torchtitan.experiments.tpu.tests.test_dtensor)
        # We replace '/' with '.' and remove the '.py' extension
        module = str(path.with_suffix("")).replace("/", ".")
        
        if module not in DISABLED_TESTS:
            category = classify_test(path)
            categorized_tests[category].append(module)
            
    # Sort for deterministic behavior
    for category in categorized_tests:
        categorized_tests[category].sort()
        
    return categorized_tests

def chunk_tests(tests, num_shards):
    if not tests:
        return []
        
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
    
    if not any(tests.values()):
        print("Error: No tests found", file=sys.stderr)
        sys.exit(1)
        
    matrix = {
        "distributed_shards": chunk_tests(tests["distributed"], NUM_DISTRIBUTED_TPU_SHARDS),
        "single_tpu_shards": chunk_tests(tests["single_tpu"], NUM_SINGLE_TPU_SHARDS),
        "cpu_shards": chunk_tests(tests["cpu"], NUM_CPU_SHARDS)
    }
    
    # We output a JSON object of shard arrays to stdout
    print(json.dumps(matrix))

if __name__ == "__main__":
    main()
