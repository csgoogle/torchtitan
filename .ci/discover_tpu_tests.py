"""Discover test modules across the whole repository and shard them for CI.

Every ``test_*.py`` file in the repository is a candidate; files that define no
tests (shared helpers such as ``test_utils.py``) are skipped. Each discovered
test is classified along two independent axes:

* hardware -- ``distributed`` (multi-chip TPU), ``single_tpu``, or ``cpu``,
  derived from the base classes the test's TestCase inherits from.
* runner -- how the file is invoked:

  - ``module``: absltest/unittest files with a ``__main__`` block, run as
    ``python -m <module>``. These are the tests written in G3.
  - ``pytest``: bare pytest-style files with no entry point, plus everything
    under ``PYTEST_ONLY_DIRS``, run as ``pytest <path>``. These are the tests
    from upstream under tests/unit_tests.

The resulting shards are emitted as JSON for the GitHub Actions matrix.
"""

import ast
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

NUM_DISTRIBUTED_TPU_SHARDS = 4
NUM_SINGLE_TPU_SHARDS = 2
NUM_CPU_SHARDS = 2
NUM_PYTEST_SHARDS = 2

# Trees run under pytest even when a file has a unittest entry point. pytest
# collects unittest.TestCase natively, so routing the whole unit test suite
# through one runner keeps behavior uniform. Paths are relative to REPO_ROOT.
PYTEST_ONLY_DIRS = ("tests/unit_tests",)

# Directory names pruned during the walk. These hold no importable tests.
EXCLUDED_DIR_NAMES = {
    ".ci",
    ".claude",
    ".git",
    ".github",
    ".pytest_cache",
    "__pycache__",
    "assets",
    "build",
    "dist",
    "outputs",
    "node_modules",
}

# Tests that are temporarily disabled.
DISABLED_TESTS = {
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
    "torchtitan.experiments.tpu.kernels.tests.test_kernel_name_determinism_distributed",
    # TODO: SyntaxError: invalid syntax (from jaxtyping)
    "torchtitan.experiments.tpu.redacted.tests.test_train_minimal_distributed",
    # TODO: Unknown command line flag 'device'
    "torchtitan.experiments.tpu.tests.test_workarounds",
    # TODO: AssertionError: test_deepseek_complete_model_training_steps_cpu_device_parity_MOE
    "torchtitan.experiments.tpu.deepseek_v3.tests.test_deepseek_v3",
    # TODO: RuntimeError: create_process_group(): PjRtClient is not initialized.
    "torchtitan.experiments.tpu.graph_trainer.tests.test_layer_stack_distributed",
    # TODO: imports torchtitan.scripts; this fork keeps scripts/ at the repo root.
    "tests.unit_tests.test_download_hf_assets",
    "tests.unit_tests.test_tokenizer",
    # TODO: mark_dynamic() no longer sets _dynamo_dynamic_range, so the copy assertion is stale.
    "torchtitan.experiments.graph_trainer.tests.test_trace_module",
    # Missing optional dependency renderers
    "torchtitan.experiments.rl.tests.test_alphabet_sort",
    "torchtitan.experiments.rl.tests.test_async_controller",
    "torchtitan.experiments.rl.tests.test_rollout_recorder",
    "torchtitan.experiments.rl.tests.test_rollout_utils",
    "torchtitan.experiments.rl.tests.test_search_r1",
    "torchtitan.experiments.rl.tests.test_weight_sync",
    # Missing optional dependency torchstore
    "torchtitan.experiments.rl.tests.test_engine_loop",
    # Missing optional dependency monarch
    "torchtitan.experiments.rl.tests.test_shutdown",
    # Missing optional dependency vllm
    "torchtitan.experiments.rl.tests.test_generator",
    "torchtitan.experiments.rl.tests.test_bitwise_parity",
    # Requires CUDA; upstream runs these tests only on CUDA
    "torchtitan.experiments.graph_trainer.tests.test_passes",
    "torchtitan.experiments.graph_trainer.tests.test_profiler",
    "torchtitan.experiments.graph_trainer.tests.test_cpu_offload",
    "torchtitan.experiments.graph_trainer.tests.test_numerics",
    # Requires optional dependency fla from requirements-vlm.txt
    "tests.unit_tests.test_rope",
    # Missing optional dependency torchvision
    "tests.unit_tests.test_mm_dataset_checkpointing",
    # Missing optional dependency autoparallel
    "torchtitan.experiments.graph_trainer.tests.test_autoparallel",
    # Missing optional dependency torchax
    "torchtitan.experiments.torchax.conformer.tests.test_conformer_equivalence",
    "torchtitan.experiments.torchax.tests.test_train_distributed",
    # TODO: get_peak_flops() rejects 'cpu', so MFU setup raises; make MFU optional.
    "torchtitan.experiments.jax.tests.test_train_distributed",
}


def _iter_test_files():
    """Yield every ``test_*.py`` path in the repository, pruning excluded dirs."""
    stack = [REPO_ROOT]
    while stack:
        directory = stack.pop()
        for entry in sorted(directory.iterdir()):
            if entry.is_dir():
                if entry.name not in EXCLUDED_DIR_NAMES:
                    stack.append(entry)
            elif entry.name.startswith("test_") and entry.suffix == ".py":
                yield entry


def _is_test_node(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name.startswith("test")
    if isinstance(node, ast.ClassDef):
        return node.name.startswith("Test") or any(
            _is_test_node(child) for child in node.body
        )
    return False


def _base_name(base):
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def classify_test(tree):
    """Return the hardware category a parsed test module needs."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = _base_name(base)
            if base_name is None:
                continue
            if "BaseDistributedDeviceTest" in base_name:
                return "distributed"
            if "BaseAcceleratorDeviceTest" in base_name:
                return "single_tpu"
    return "cpu"


def _is_pytest_only(relative):
    """True if the path lives in a tree that always runs under pytest."""
    return str(relative).startswith(PYTEST_ONLY_DIRS)


def has_main_block(tree):
    """True if the module has a top-level ``if __name__ == "__main__":`` guard."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        for sub in ast.walk(node.test):
            if isinstance(sub, ast.Name) and sub.id == "__name__":
                return True
    return False


def discover_tests():
    """Bucket every discovered test by hardware category and runner.

    The distributed, single_tpu and cpu buckets hold dotted module paths; the
    pytest bucket holds file paths, matching how each runner is invoked.
    """
    categorized_tests = {
        "distributed": [],
        "single_tpu": [],
        "cpu": [],
        "pytest": [],
    }

    for path in _iter_test_files():
        relative = path.relative_to(REPO_ROOT)
        module = str(relative.with_suffix("")).replace("/", ".")
        if module in DISABLED_TESTS:
            continue

        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            print(f"Warning: skipping unparsable {relative}: {exc}", file=sys.stderr)
            continue

        # Skip helpers (e.g. test_utils.py) that match the glob but define no tests.
        if not any(_is_test_node(node) for node in ast.walk(tree)):
            continue

        category = classify_test(tree)
        if category == "distributed" and not path.name.endswith("_distributed.py"):
            sys.exit(
                f"Error: Test '{relative}' is classified as distributed (inherits from "
                "BaseDistributedDeviceTest) but its filename does not end with '_distributed.py'. "
                "Please rename it to end with '_distributed.py'."
            )

        if category == "cpu":
            # Categorize tests without a `__main__` entry point to run under pytest.
            if _is_pytest_only(relative) or not has_main_block(tree):
                categorized_tests["pytest"].append(str(relative))
            else:
                categorized_tests["cpu"].append(module)
        else:
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
    if not (REPO_ROOT / "torchtitan").is_dir():
        sys.exit(f"Error: {REPO_ROOT} is not a torchtitan checkout")

    tests = discover_tests()
    if not any(tests.values()):
        sys.exit("Error: No tests found")


    matrix = {
        "distributed_shards": chunk_tests(tests["distributed"], NUM_DISTRIBUTED_TPU_SHARDS),
        "single_tpu_shards": chunk_tests(tests["single_tpu"], NUM_SINGLE_TPU_SHARDS),
        "cpu_shards": chunk_tests(tests["cpu"], NUM_CPU_SHARDS),
        "pytest_shards": chunk_tests(tests["pytest"], NUM_PYTEST_SHARDS),
    }
    
    # We output a JSON object of shard arrays to stdout
    print(json.dumps(matrix))

if __name__ == "__main__":
    main()
