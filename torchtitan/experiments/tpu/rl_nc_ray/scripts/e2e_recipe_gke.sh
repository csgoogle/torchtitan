#!/bin/bash
# Submit to GKE with vLLM explicitly enabled

export RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1

MODEL_SIZE=${1:-"0.6b"}
shift

case "${MODEL_SIZE,,}" in
  "0.6b")
    CONFIG="rl_grpo_qwen3_0_6b"
    HF_ASSETS_PATH="/data/assets/assets/hf/Qwen3-0.6B"
    ;;
  "1.7b")
    CONFIG="rl_grpo_qwen3_1_7b"
    HF_ASSETS_PATH="/data/assets/assets/hf/Qwen3-1.7B"
    ;;
  "4b")
    CONFIG="rl_grpo_qwen3_4b"
    HF_ASSETS_PATH="/data/assets/assets/hf/Qwen3-4B"
    ;;
  *)
    echo "Unknown model size: $MODEL_SIZE. Supported: 0.6b, 1.7b, 4b"
    exit 1
    ;;
esac

ray job submit \
  --address="http://127.0.0.1:8265" \
  --working-dir . \
  --runtime-env-json '{"env_vars": {"PYTHONPATH": "torchtitan/experiments/tpu/rl_nc_ray/dummy_modules:.", "PYTHONUNBUFFERED": "1"}}' \
  -- \
  python3 -m torchtitan.experiments.tpu.rl_nc_ray.grpo \
    --module rl \
    --config "$CONFIG" \
    --generator.parallelism.tensor-parallel-degree 8 \
    --trainer.parallelism.tensor-parallel-degree 2 \
    --trainer.parallelism.data_parallel_shard_degree 4 \
    --num_prompts_per_step 4 \
    --hf_assets_path "$HF_ASSETS_PATH" \
    --num_steps 10 \
    --log_samples \
    "$@"

